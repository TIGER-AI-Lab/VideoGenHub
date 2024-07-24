import argparse
import os

import imageio
import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import CLIPTokenizer, CLIPTextModel

from pipeline_videogen import VideoGenPipeline
from videogen_hub.pipelines.lavie.lavie_src.base.download import find_model
from videogen_hub.pipelines.lavie.lavie_src.base.models import get_models


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(
        device)  # huge

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()

    if args.sample_method == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(sd_path,
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start,
                                                  beta_end=args.beta_end,
                                                  beta_schedule=args.beta_schedule)
    elif args.sample_method == 'eulerdiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
                                                           subfolder="scheduler",
                                                           beta_start=args.beta_start,
                                                           beta_end=args.beta_end,
                                                           beta_schedule=args.beta_schedule)
    elif args.sample_method == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(sd_path,
                                                  subfolder="scheduler",
                                                  beta_start=args.beta_start,
                                                  beta_end=args.beta_end,
                                                  beta_schedule=args.beta_schedule)
    else:
        raise NotImplementedError

    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder_one,
                                         tokenizer=tokenizer_one,
                                         scheduler=scheduler,
                                         unet=unet).to(device)
    videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    video_grids = []
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt,
                                   video_length=args.video_length,
                                   height=args.image_size[0],
                                   width=args.image_size[1],
                                   num_inference_steps=args.num_sampling_steps,
                                   guidance_scale=args.guidance_scale).video
        imageio.mimwrite(args.output_folder + prompt.replace(' ', '_') + '.mp4', videos[0], fps=8,
                         quality=9)  # highest quality is 10, lowest is 0

    print('save path {}'.format(args.output_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    config = args.config

    # Overwrite config with command line arguments
    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    main(OmegaConf.load(config))
