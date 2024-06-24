import argparse
import glob
import os
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from videogen_hub.common.lvdm.models.samplers.ddim import DDIMSampler
from videogen_hub.common.lvdm.models.samplers.ddim_multiplecond import \
    DDIMSampler as DDIMSampler_multicond
from videogen_hub.pipelines.dynamicrafter.utils import instantiate_from_config


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_data_prompts(data_dir, video_size=(256, 256), video_frames=16, interp=False):
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file) - 1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist

    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2 * idx]).convert('RGB')
            image2 = Image.open(file_list[2 * idx + 1]).convert('RGB')
            frame_tensor = processing_image((image1, image2), video_size, video_frames, True)
            _, filename = os.path.split(file_list[idx * 2])
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            frame_tensor = processing_image(image, video_size, video_frames, False)
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)

    return filename_list, data_list, prompt_list


def processing_image(image, video_size=(256, 256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if interp:
        image1, image2 = image
        image_tensor1 = transform(image1).unsqueeze(1)  # [c,1,h,w]
        image_tensor2 = transform(image2).unsqueeze(1)  # [c,1,h,w]
        frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
        frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
        frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
    else:
        image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
        frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
    return frame_tensor


def save_results(prompt, samples, filename, fakedir, fps=8, loop=False):
    filename = filename.split('.')[0] + '.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
        if loop:
            video = video[:-1, ...]

        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in
                       video]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264',
                                   options={'crf': '10'})  ## crf indicates the quality


def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop:  # remove the last frame
            video = video[:, :, :-1, ...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i, ...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0)  # thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'),
                                f'{filename.split(".")[0]}_sample{i}.mp4')
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.,
                           unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False,
                           multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform',
                           guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    img = videos[:, :, 0]  # bchw
    img_emb = model.embedder(img)  ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos)  # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
            img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
        else:
            img_cat_cond = z[:, :, :1, :, :]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=noise_shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             cfg_img=cfg_img,
                                             mask=cond_mask,
                                             x0=cond_z0,
                                             fs=fs,
                                             timestep_spacing=timestep_spacing,
                                             guidance_rescale=guidance_rescale,
                                             **kwargs
                                             )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt", )
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="eta for ddim sampling (0.0 yields deterministic sampling)", )
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3,
                        help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False,
                        help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform",
                        help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0,
                        help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False,
                        help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")

    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False,
                        help="generate generative frame interpolation or not")
    return parser


class DynamiCrafterPipeline():
    def __init__(self, args):
        """
        Initialize the parameters from args
        Args:
            args: is a list consisting of arguments needed for parser.
            e.g. ["--ckpt_path", <the model path>, ......]
        """
        parser = get_parser()
        self.args = parser.parse_args(args)

    def run_inference(self, input_image):
        """
        Run inference from the input_image.
        This input image can either be a tensor or a string as the path of the image file.
        Args:
            input_image: tensor or string.

        Returns: a tensor representing the generated video of shape (num_frames, channels, height, width)

        """
        args = self.args
        seed_everything(args.seed)
        ## model config
        config = OmegaConf.load(self.args.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.cuda()
        model.perframe_ae = args.perframe_ae
        assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, args.ckpt_path)
        model.eval()

        ## run over data
        assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        assert args.bs == 1, "Current implementation only support [batch size = 1]!"
        ## latent noise shape
        h, w = args.height // 8, args.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = args.video_length
        print(f'Inference with {n_frames} frames')
        noise_shape = [args.bs, channels, n_frames, h, w]

        # fakedir = os.path.join(args.savedir, "samples")
        # fakedir_separate = os.path.join(args.savedir, "samples_separate")

        # os.makedirs(fakedir, exist_ok=True)
        # os.makedirs(fakedir_separate, exist_ok=True)

        ## prompt file setting

        if type(input_image) == str:
            args.prompt_dir = input_image
            assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
            filename_list, data_list, prompt_list = load_data_prompts(args.prompt_dir,
                                                                      video_size=(args.height, args.width),
                                                                      video_frames=n_frames, interp=args.interp)
        else:
            input_pil = (transforms.ToPILImage())(input_image)
            frame_tensor = processing_image(input_pil, (args.height, args.width), n_frames, args.interp)
            data_list, prompt_list = [frame_tensor], [args.text_input]

        num_samples = len(prompt_list)
        # print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
        # indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
        # indices = list(range(0, num_samples))
        # prompt_list_rank = [prompt_list[i] for i in indices]
        # data_list_rank = [data_list[i] for i in indices]
        # filename_list_rank = [filename_list[i] for i in indices]

        # start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            # for idx, indice in tqdm(enumerate(range(0, len(prompt_list), args.bs)), desc='Sample Batch'):
            prompts = prompt_list[0]
            videos = data_list[0]
            # filenames = filename_list[0]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps,
                                                   args.ddim_eta, \
                                                   args.unconditional_guidance_scale, args.cfg_img, args.frame_stride,
                                                   args.text_input, args.multiple_cond_cfg, args.loop, args.interp,
                                                   args.timestep_spacing, args.guidance_rescale)

            output = batch_samples.squeeze().permute(1, 0, 2, 3)
            return output
            # save each example individually
            # for nn, samples in enumerate(batch_samples):
            #     ## samples : [n_samples,c,t,h,w]
            #     prompt = prompts[nn]
            #     filename = filenames[nn]
            #     # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
            #     save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)

        # print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")
