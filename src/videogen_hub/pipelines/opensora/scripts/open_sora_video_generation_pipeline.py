import os
import time
from pprint import pformat
from typing import Union, List

import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from mmengine.runner import set_random_seed
from mmengine import Config as mmengine_config
from tqdm import tqdm

from videogen_hub.pipelines.opensora.opensora.acceleration.parallel_states import set_sequence_parallel_group
from videogen_hub.pipelines.opensora.opensora.datasets import save_sample
from videogen_hub.pipelines.opensora.opensora.datasets.aspect import get_image_size, get_num_frames
from videogen_hub.pipelines.opensora.opensora.models.text_encoder.t5 import text_preprocessing
from videogen_hub.pipelines.opensora.opensora.registry import MODELS, SCHEDULERS, build_module
from videogen_hub.pipelines.opensora.opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from videogen_hub.pipelines.opensora.opensora.utils.misc import all_exists, create_logger, is_distributed, \
    is_main_process

logger = logging.get_logger(__name__)


class OpenSoraVideoGenerationPipelineOutput(BaseOutput):
    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class OpenSoraVideoGenerationPipeline(DiffusionPipeline):
    def __init__(self, config):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            import colossalai
            from colossalai.cluster import DistCoordinator
        except ImportError:
            colossalai = None

        if is_distributed() and colossalai is not None:
            colossalai.launch_from_torch({})
            self.coordinator = DistCoordinator()
            self.enable_sequence_parallelism = self.coordinator.world_size > 1
            if self.enable_sequence_parallelism:
                set_sequence_parallel_group(dist.group.WORLD)
        else:
            self.coordinator = None
            self.enable_sequence_parallelism = False

        set_random_seed(seed=config.get("seed", 1024))
        self.logger = create_logger()
        self.logger.info("Inference configuration:\n %s", pformat(config))

        self.verbose = config.get("verbose", 1)
        self.progress_wrap = tqdm if self.verbose == 1 else (lambda x: x)

        # Build models
        self.text_encoder = build_module(config.text_encoder, MODELS, device=self.device)
        self.vae = build_module(config.vae, MODELS).to(self.device, self.dtype).eval()

        self.image_size = config.get("image_size", None)
        if self.image_size is None:
            self.resolution = config.get("resolution", None)
            self.aspect_ratio = config.get("aspect_ratio", None)
            assert self.resolution is not None and self.aspect_ratio is not None, "resolution and aspect_ratio must be provided if image_size is not provided"
            self.image_size = get_image_size(self.resolution, self.aspect_ratio)
        self.num_frames = get_num_frames(config.num_frames)

        self.input_size = (self.num_frames, *self.image_size)
        self.latent_size = self.vae.get_latent_size(self.input_size)
        self.model = build_module(
            config.model,
            MODELS,
            input_size=self.latent_size,
            in_channels=self.vae.out_channels,
            caption_channels=self.text_encoder.output_dim,
            model_max_length=self.text_encoder.model_max_length,
            enable_sequence_parallelism=self.enable_sequence_parallelism,
        ).to(self.device, self.dtype).eval()

        self.text_encoder.y_embedder = self.model.y_embedder  # HACK: for classifier-free guidance
        self.scheduler = build_module(config.scheduler, SCHEDULERS)

    def prepare_inputs(self, prompts):
        config = self.config
        if prompts is None:
            prompts = config.get("prompt", None)
            start_idx = config.get("start_index", 0)
            if prompts is None:
                if config.get("prompt_path", None) is not None:
                    prompts = load_prompts(config.get("prompt_path"), start_idx, config.get("end_index", None))
                else:
                    prompts = [config.get("prompt_generator", "")] * 1_000_000  # endless loop

        reference_path = config.get("reference_path", [""] * len(prompts))
        mask_strategy = config.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        return prompts, reference_path, mask_strategy

    def __call__(self, config: mmengine_config, return_dict=True):
        config = self.config
        torch.set_grad_enabled(False)
        prompts = config.get("prompt", None)

        prompts, reference_path, mask_strategy = self.prepare_inputs(prompts)

        fps = config.get("fps", 24)
        save_fps = config.get("save_fps", fps // config.get("frame_interval", 1))
        multi_resolution = config.get("multi_resolution", None)
        batch_size = config.get("batch_size", 1)
        num_sample = config.get("num_sample", 1)
        loop = config.get("loop", 1)
        condition_frame_length = config.get("condition_frame_length", 5)
        condition_frame_edit = config.get("condition_frame_edit", 0.0)
        align = config.get("align", None)

        save_dir = config.get("save_dir", "./samples/samples/")
        os.makedirs(save_dir, exist_ok=True)
        sample_name = config.get("sample_name", None)
        prompt_as_path = config.get("prompt_as_path", False)
        video_clips = []
        for i in self.progress_wrap(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i: i + batch_size]
            ms = mask_strategy[i: i + batch_size]
            refs = reference_path[i: i + batch_size]

            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts
            refs = collect_references_batch(refs, self.vae, self.image_size)

            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), self.image_size, self.num_frames, fps, self.device, self.dtype
            )
            start_idx = 0

            for k in range(num_sample):
                save_paths = [
                    get_save_path_name(
                        save_dir,
                        sample_name=sample_name,
                        sample_idx=start_idx + idx,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    )
                    for idx in range(len(batch_prompts))
                ]

                if prompt_as_path and all_exists(save_paths):
                    continue

                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                if config.get("llm_refine", False):
                    if not self.enable_sequence_parallelism or (self.enable_sequence_parallelism and is_main_process()):
                        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                            batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                    if self.enable_sequence_parallelism:
                        self.coordinator.block_all()
                        prompt_segment_length = [len(prompt_segment_list) for prompt_segment_list in
                                                 batched_prompt_segment_list]
                        batched_prompt_segment_list = [prompt_segment for prompt_segment_list in
                                                       batched_prompt_segment_list for prompt_segment in
                                                       prompt_segment_list]
                        broadcast_obj_list = [batched_prompt_segment_list] * self.coordinator.world_size
                        dist.broadcast_object_list(broadcast_obj_list)
                        batched_prompt_segment_list = []
                        segment_start_idx = 0
                        all_prompts = broadcast_obj_list[0]
                        for num_segment in prompt_segment_length:
                            batched_prompt_segment_list.append(
                                all_prompts[segment_start_idx: segment_start_idx + num_segment])
                            segment_start_idx += num_segment

                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes=config.get("aes", None),
                        flow=config.get("flow", None),
                        camera_motion=config.get("camera_motion", None),
                    )

                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                video_clips = []
                for loop_i in range(loop):
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)
                    if loop_i > 0:
                        refs, ms = append_generated(
                            self.vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    z = torch.randn(len(batch_prompts), self.vae.out_channels, *self.latent_size, device=self.device,
                                    dtype=self.dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = self.scheduler.sample(
                        self.model,
                        self.text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=self.device,
                        additional_args=model_args,
                        progress=self.verbose >= 2,
                        mask=masks,
                    )
                    samples = self.vae.decode(samples.to(self.dtype), num_frames=self.num_frames)
                    video_clips.append(samples)

                if is_main_process():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        if self.verbose >= 2:
                            self.logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[i][idx] for i in range(loop)]
                        for i in range(1, loop):
                            video[i] = video[i][:, dframe_to_frame(condition_frame_length):]
                        video = torch.cat(video, dim=1)
                        save_path = save_sample(
                            video,
                            fps=save_fps,
                            save_path=save_path,
                            verbose=self.verbose >= 2,
                        )
                        if save_path.endswith(".mp4") and config.get("watermark", False):
                            time.sleep(1)  # prevent loading previous generated video
                            add_watermark(save_path)
            start_idx += len(batch_prompts)

        if not return_dict:
            return video_clips

        return OpenSoraVideoGenerationPipelineOutput(frames=video_clips)
