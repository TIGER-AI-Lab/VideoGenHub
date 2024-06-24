# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from einops import rearrange
from PIL import Image
import numpy as np
from torchvision import transforms

from videogen_hub.pipelines.seine.datasets_seine import video_transforms
from natsort import natsorted


def get_input(args):
    input_path = args.input_path
    transform_video = transforms.Compose([
        video_transforms.ToTensorVideo(),  # TCHW
        video_transforms.ResizeVideo((args.image_h, args.image_w)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if input_path is not None:
        print(f'loading video from {input_path}')
        if os.path.isdir(input_path):
            file_list = os.listdir(input_path)
            video_frames = []
            if args.mask_type.startswith('onelast'):
                num = int(args.mask_type.split('onelast')[-1])
                # get first and last frame
                first_frame_path = os.path.join(input_path, natsorted(file_list)[0])
                last_frame_path = os.path.join(input_path, natsorted(file_list)[-1])
                first_frame = torch.as_tensor(
                    np.array(Image.open(first_frame_path), dtype=np.uint8, copy=True)).unsqueeze(0)
                last_frame = torch.as_tensor(
                    np.array(Image.open(last_frame_path), dtype=np.uint8, copy=True)).unsqueeze(0)
                for i in range(num):
                    video_frames.append(first_frame)
                # add zeros to frames
                num_zeros = args.num_frames - 2 * num
                for i in range(num_zeros):
                    zeros = torch.zeros_like(first_frame)
                    video_frames.append(zeros)
                for i in range(num):
                    video_frames.append(last_frame)
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)  # f,c,h,w
                video_frames = transform_video(video_frames)
            else:
                for file in file_list:
                    if file.endswith('jpg') or file.endswith('png'):
                        image = torch.as_tensor(np.array(Image.open(file), dtype=np.uint8, copy=True)).unsqueeze(0)
                        video_frames.append(image)
                    else:
                        continue
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)  # f,c,h,w
                video_frames = transform_video(video_frames)
            return video_frames, n
        elif os.path.isfile(input_path):
            _, full_file_name = os.path.split(input_path)
            file_name, extension = os.path.splitext(full_file_name)
            if extension == '.jpg' or extension == '.png':
                print("loading the input image")
                video_frames = []
                num = int(args.mask_type.split('first')[-1])
                first_frame = torch.as_tensor(np.array(Image.open(input_path), dtype=np.uint8, copy=True)).unsqueeze(0)
                for i in range(num):
                    video_frames.append(first_frame)
                num_zeros = args.num_frames - num
                for i in range(num_zeros):
                    zeros = torch.zeros_like(first_frame)
                    video_frames.append(zeros)
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)  # f,c,h,w
                video_frames = transform_video(video_frames)
                return video_frames, n
            else:
                raise TypeError(f'{extension} is not supported !!')
        else:
            raise ValueError('Please check your path input!!')
    else:
        raise ValueError('Need to give a video or some images')


def auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device, ):
    b, f, c, h, w = video_input.shape
    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8

    # prepare inputs
    if args.use_fp16:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, dtype=torch.float16,
                        device=device)  # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, device=device)  # b,c,f,h,w

    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:, :, 0, :], size=(latent_h, latent_w)).unsqueeze(1)

    # classifier_free_guidance
    if args.do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 2)
        mask = torch.cat([mask] * 2)
        z = torch.cat([z] * 2)
        prompt_all = [prompt] + [args.negative_prompt]

    else:
        masked_video = masked_video
        mask = mask
        z = z
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt,
                        class_labels=None,
                        cfg_scale=args.cfg_scale,
                        use_fp16=args.use_fp16, )  # tav unet

    # Sample video:
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    samples, _ = samples.chunk(2, dim=0)  # [1, 4, 16, 32, 32]
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous()  # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample  # [16, 3, 256, 256]
    return video_clip
