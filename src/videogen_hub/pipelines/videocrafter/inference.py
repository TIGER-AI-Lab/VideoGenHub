import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from .funcs import load_model_checkpoint, load_image_batch, get_filelist, save_videos
from .funcs import batch_ddim_sampling
from .utils import instantiate_from_config

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt", )
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="eta for ddim sampling (0.0 yields deterministic sampling)", )
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None,
                        help="temporal consistency guidance")
    ## for conditional i2v only
    # parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    return parser


class VideoCrafterPipeline():
    def __init__(self, arg_list, device, rank: int = 0, gpu_num: int = 1):
        """
        Initialize the pipeline of videocrafter.
        It is always on one GPU.
        Args:
            arg_list: The parameters needed for the model.
            device:
            rank:
            gpu_num:
        """
        parser = get_parser()
        self.args = parser.parse_args(args=arg_list)

        self.gpu_no, self.gpu_num = rank, gpu_num
        
        inference_t2v_512_v2_config = """
            model:
            target: lvdm.models.ddpm3d.LatentDiffusion
            params:
                linear_start: 0.00085
                linear_end: 0.012
                num_timesteps_cond: 1
                timesteps: 1000
                first_stage_key: video
                cond_stage_key: caption
                cond_stage_trainable: false
                conditioning_key: crossattn
                image_size:
                - 40
                - 64
                channels: 4
                scale_by_std: false
                scale_factor: 0.18215
                use_ema: false
                uncond_type: empty_seq
                use_scale: true
                scale_b: 0.7
                unet_config:
                target: lvdm.modules.networks.openaimodel3d.UNetModel
                params:
                    in_channels: 4
                    out_channels: 4
                    model_channels: 320
                    attention_resolutions:
                    - 4
                    - 2
                    - 1
                    num_res_blocks: 2
                    channel_mult:
                    - 1
                    - 2
                    - 4
                    - 4
                    num_head_channels: 64
                    transformer_depth: 1
                    context_dim: 1024
                    use_linear: true
                    use_checkpoint: true
                    temporal_conv: true
                    temporal_attention: true
                    temporal_selfatt_only: true
                    use_relative_position: false
                    use_causal_attention: false
                    temporal_length: 16
                    addition_attention: true
                    fps_cond: true
                first_stage_config:
                target: lvdm.models.autoencoder.AutoencoderKL
                params:
                    embed_dim: 4
                    monitor: val/rec_loss
                    ddconfig:
                    double_z: true
                    z_channels: 4
                    resolution: 512
                    in_channels: 3
                    out_ch: 3
                    ch: 128
                    ch_mult:
                    - 1
                    - 2
                    - 4
                    - 4
                    num_res_blocks: 2
                    attn_resolutions: []
                    dropout: 0.0
                    lossconfig:
                    target: torch.nn.Identity
                cond_stage_config:
                target: lvdm.modules.encoders.condition.FrozenOpenCLIPEmbedder
                params:
                    freeze: true
                    layer: penultimate
            """

        config = OmegaConf.load(inference_t2v_512_v2_config)

        # data_config = config.pop("data", OmegaConf.create())
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda(self.gpu_no)
        print("About to load model")
        assert os.path.exists(self.args.ckpt_path), f"Error: checkpoint [{self.args.ckpt_path}] Not Found!"
        self.model = load_model_checkpoint(model, self.args.ckpt_path)
        self.model.eval()

    def run_inference(self, prompt, video_length, height, width, **kwargs):
        """
        https://github.com/AILab-CVC/VideoCrafter
        Generate video from the provided text prompt.
        Args:
            prompt: The provided text prompt.
            video_length: The length (num of frames) of the generated video.
            height: The height of the video frame.
            width: The width of the video frame.
            **kwargs:

        Returns:
            The generated video represented as tensor with shape (1, 1, channels, height, width, num of frames)

        """
        ## step 1: model config
        ## -----------------------------------------------------------------
        ## sample shape
        assert (self.args.height % 16 == 0) and (
                self.args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        ## latent noise shape
        h, w = height // 8, width // 8
        frames = video_length
        channels = self.model.channels

        ## step 2: load data
        ## -----------------------------------------------------------------
        prompt_list = [prompt]
        num_samples = len(prompt_list)
        # filename_list = [f"{id + 1:04d}" for id in range(num_samples)]

        gpu_num = self.gpu_num
        gpu_no = self.gpu_no
        samples_split = num_samples // gpu_num
        residual_tail = num_samples % gpu_num
        print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
        indices = list(range(samples_split * gpu_no, samples_split * (gpu_no + 1)))
        if gpu_no == 0 and residual_tail != 0:
            indices = indices + list(range(num_samples - residual_tail, num_samples))
        prompt_list_rank = [prompt_list[i] for i in indices]

        # # conditional input
        # if self.args.mode == "i2v":
        #     ## each video or frames dir per prompt
        #     cond_inputs = get_filelist(self.args.cond_input, ext='[mpj][pn][4gj]')  # '[mpj][pn][4gj]'
        #     assert len(
        #         cond_inputs) == num_samples, f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
        #     filename_list = [f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)]
        #     cond_inputs_rank = [cond_inputs[i] for i in indices]

        # filename_list_rank = [filename_list[i] for i in indices]

        ## step 3: run over samples
        ## -----------------------------------------------------------------
        # start = time.time()
        n_rounds = len(prompt_list_rank) // self.args.bs
        n_rounds = n_rounds + 1 if len(prompt_list_rank) % self.args.bs != 0 else n_rounds
        for idx in range(0, n_rounds):
            print(f'[rank:{gpu_no}] batch-{idx + 1} ({self.args.bs})x{self.args.n_samples} ...')
            idx_s = idx * self.args.bs
            idx_e = min(idx_s + self.args.bs, len(prompt_list_rank))
            batch_size = idx_e - idx_s
            # filenames = filename_list_rank[idx_s:idx_e]
            noise_shape = [batch_size, channels, frames, h, w]
            fps = torch.tensor([self.args.fps] * batch_size).to(self.model.device).long()

            prompts = prompt_list_rank[idx_s:idx_e]
            if isinstance(prompts, str):
                prompts = [prompts]
            # prompts = batch_size * [""]
            text_emb = self.model.get_learned_conditioning(prompts)

            if self.args.mode == 'base':
                cond = {"c_crossattn": [text_emb], "fps": fps}
            # elif self.args.mode == 'i2v':
            #     # cond_images = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            #     cond_images = load_image_batch(cond_inputs_rank[idx_s:idx_e], (self.args.height, self.args.width))
            #     cond_images = cond_images.to(self.model.device)
            #     img_emb = self.model.get_image_embeds(cond_images)
            #     imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            #     cond = {"c_crossattn": [imtext_cond], "fps": fps}
            else:
                raise NotImplementedError

            ## inference
            batch_samples = batch_ddim_sampling(self.model, cond, noise_shape, self.args.n_samples,
                                                self.args.ddim_steps,
                                                self.args.ddim_eta,
                                                self.args.unconditional_guidance_scale, **kwargs)
            return batch_samples
