import os
import imageio
from PIL import Image
from typing import List

import torch
import torch.nn.functional as F

from diffusers import IFSuperResolutionPipeline, VideoToVideoSDPipeline
from diffusers.utils.torch_utils import randn_tensor


class ShowOnePipeline():
    def __init__(self):
        from .showone.pipelines import TextToVideoIFPipeline, TextToVideoIFInterpPipeline, TextToVideoIFSuperResolutionPipeline
        from .showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
        from .showone.pipelines.pipeline_t2v_sr_pixel_cond import TextToVideoIFSuperResolutionPipeline_Cond
        from huggingface_hub import snapshot_download
        from huggingface_hub import login

        self.tensor2vid = tensor2vid
        # Base Model
        # When using "showlab/show-1-base-0.0", it's advisable to increase the number of inference steps (e.g., 100)
        # and opt for a larger guidance scale (e.g., 12.0) to enhance visual quality.
        path = "./checkpoints/showlab/show-1-base"
        pretrained_model_path = snapshot_download(
            repo_id="showlab/show-1-base", 
            local_dir=path
        )
        
        self.pipe_base = TextToVideoIFPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe_base.enable_model_cpu_offload()

        # Interpolation Model
        path = "./checkpoints/showlab/show-1-interpolation"
        pretrained_model_path = snapshot_download(
            repo_id="showlab/show-1-interpolation", 
            local_dir=path
        )

        self.pipe_interp_1 = TextToVideoIFInterpPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe_interp_1.enable_model_cpu_offload()

        # Super-Resolution Model 1
        # Image super-resolution model from DeepFloyd https://huggingface.co/DeepFloyd/IF-II-L-v1.0
        # pretrained_model_path = "./checkpoints/DeepFloyd/IF-II-L-v1.0"
        access_token = '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'
        login(token=access_token)
        
        path = "./checkpoints/DeepFloyd/IF-II-L-v1.0"
        pretrained_model_path = snapshot_download(
            repo_id="DeepFloyd/IF-II-L-v1.0", 
            local_dir=path
        )

        self.pipe_sr_1_image = IFSuperResolutionPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=None,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe_sr_1_image.enable_model_cpu_offload()

        path = "./checkpoints/showlab/show-1-sr1"
        pretrained_model_path = snapshot_download(
            repo_id="showlab/show-1-sr1", 
            local_dir=path
        )

        self.pipe_sr_1_cond = TextToVideoIFSuperResolutionPipeline_Cond.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float16
        )
        self.pipe_sr_1_cond.enable_model_cpu_offload()

        # Super-Resolution Model 2
        path = "./checkpoints/showlab/show-1-sr2"
        pretrained_model_path = snapshot_download(
            repo_id="showlab/show-1-sr2", 
            local_dir=path
        )

        self.pipe_sr_2 = VideoToVideoSDPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float16
        )
        self.pipe_sr_2.enable_model_cpu_offload()
        self.pipe_sr_2.enable_vae_slicing()

    def inference(self, prompt: str = "A burning lamborghini driving on rainbow.",
                  negative_prompt="low resolution, blur",
                  output_size: List[int] = [240, 560],
                  initial_num_frames: int = 8,
                  scaling_factor: int = 4,
                  seed: int = 42):

        # Inference
        # Text embeds
        prompt_embeds, negative_embeds = self.pipe_base.encode_prompt(prompt)

        # Keyframes generation (8x64x40, 2fps)
        video_frames = self.pipe_base(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_frames=initial_num_frames,
            height=40,
            width=64,
            num_inference_steps=75,
            guidance_scale=9.0,
            generator=torch.manual_seed(seed),
            output_type="pt"
        ).frames

        # Frame interpolation (8x64x40, 2fps -> 29x64x40, 7.5fps)
        
        bsz, channel, num_frames_1, height, width = video_frames.shape
        
        k = scaling_factor
        
        new_num_frames = (k-1) * (num_frames_1 - 1) + num_frames_1
        new_video_frames = torch.zeros((bsz, channel, new_num_frames, height, width),
                                       dtype=video_frames.dtype, device=video_frames.device)
        new_video_frames[:, :, torch.arange(0, new_num_frames, k), ...] = video_frames
        init_noise = randn_tensor((bsz, channel, k+1, height, width), dtype=video_frames.dtype,
                                  device=video_frames.device, generator=torch.manual_seed(seed))

        for i in range(num_frames_1 - 1):
            batch_i = torch.zeros((bsz, channel, k+1, height, width), dtype=video_frames.dtype,
                                  device=video_frames.device)
            batch_i[:, :, 0, ...] = video_frames[:, :, i, ...]
            batch_i[:, :, -1, ...] = video_frames[:, :, i + 1, ...]
            batch_i = self.pipe_interp_1(
                pixel_values=batch_i,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_frames=batch_i.shape[2],
                height=40,
                width=64,
                num_inference_steps=75,
                guidance_scale=4.0,
                generator=torch.manual_seed(seed),
                output_type="pt",
                init_noise=init_noise,
                cond_interpolation=True,
            ).frames

            new_video_frames[:, :, i * k:i * k + k + 1, ...] = batch_i

        video_frames = new_video_frames

        # Super-resolution 1 (29x64x40 -> 29x256x160)
        bsz, channel, num_frames_2, height, width = video_frames.shape
        window_size, stride = 8, 7
        new_video_frames = torch.zeros(
            (bsz, channel, num_frames_2, height * 4, width * 4),
            dtype=video_frames.dtype,
            device=video_frames.device)
        for i in range(0, num_frames_2 - window_size + 1, stride):
            batch_i = video_frames[:, :, i:i + window_size, ...]
            all_frame_cond = None

            if i == 0:
                first_frame_cond = self.pipe_sr_1_image(
                    image=video_frames[:, :, 0, ...],
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    height=height * 4,
                    width=width * 4,
                    num_inference_steps=70,
                    guidance_scale=4.0,
                    noise_level=150,
                    generator=torch.manual_seed(seed),
                    output_type="pt"
                ).images
                first_frame_cond = first_frame_cond.unsqueeze(2)
            else:
                first_frame_cond = new_video_frames[:, :, i:i + 1, ...]

            batch_i = self.pipe_sr_1_cond(
                image=batch_i,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                first_frame_cond=first_frame_cond,
                height=height * 4,
                width=width * 4,
                num_inference_steps=125,
                guidance_scale=7.0,
                noise_level=250,
                generator=torch.manual_seed(seed),
                output_type="pt"
            ).frames
            new_video_frames[:, :, i:i + window_size, ...] = batch_i

        video_frames = new_video_frames

        # Super-resolution 2 (29x256x160 -> 29x576x320)
        video_frames = [Image.fromarray(frame).resize((output_size[1], output_size[0])) for frame in self.tensor2vid(video_frames.clone())]
        video_frames = self.pipe_sr_2(
            prompt,
            negative_prompt=negative_prompt,
            video=video_frames,
            strength=0.8,
            num_inference_steps=50,
            generator=torch.manual_seed(seed),
            output_type="pt"
        ).frames

        output = video_frames.squeeze()

        return output
