from typing import List

import torchvision
from diffusers import AutoencoderKL
from diffusers.utils import is_xformers_available
from omegaconf import OmegaConf
from torchvision.utils import save_image

from videogen_hub import MODEL_PATH
from videogen_hub.pipelines.seine.diffusion import create_diffusion
from videogen_hub.pipelines.seine.models import get_models
from videogen_hub.pipelines.seine.models.clip import TextEmbedder
from videogen_hub.pipelines.seine.utils import mask_generation_before
from videogen_hub.pipelines.seine.with_mask_sample import *


class SEINEPipeline:
    def __init__(self, seine_path: str = os.path.join(MODEL_PATH, "SEINE", "seine.pt"),
                 pretrained_model_path: str = os.path.join(MODEL_PATH, "SEINE", "stable-diffusion-v1-4"),
                 config_path: str = "src/videogen_hub/pipelines/seine/sample_i2v.yaml"):
        """
        Load the configuration file and set the paths of models.
        Args:
            seine_path: The path of the downloaded seine pretrained model.
            pretrained_model_path: The path of the downloaded stable diffusion pretrained model.
            config_path: The path of the configuration file.
        """
        self.config = OmegaConf.load(config_path)
        self.config.ckpt = seine_path
        self.config.pretrained_model_path = pretrained_model_path

    def infer_one_video(self, input_image,
                        text_prompt: List = [],
                        output_size: List = [240, 560],
                        num_frames: int = 16,
                        num_sampling_steps: int = 250,
                        seed: int = 42,
                        save_video: bool = False):
        """
        Generate video based on provided input_image and text_prompt.
        Args:
            input_image: The input image to generate video.
            text_prompt: The text prompt to generate video.
            output_size: The size of the generated video. Defaults to [240, 560].
            num_frames: number of frames of the generated video. Defaults to 16.
            num_sampling_steps: number of sampling steps to generate the video. Defaults to 250.
            seed: The random seed for video generation. Defaults to 42.
            save_video: save the video to the path in config if it is True. Not save if it is False. Defaults to False.

        Returns:
            The generated video as tensor with shape (num_frames, channels, height, width).

        """

        self.config.image_size = output_size
        self.config.num_frames = num_frames
        self.config.num_sampling_steps = num_sampling_steps
        self.config.seed = seed
        self.config.text_prompt = text_prompt
        print(input_image, type(input_image) == str)
        if type(input_image) == str:
            self.config.input_path = input_image
        else:
            assert torch.is_tensor(input_image)
            assert len(input_image.shape) == 3
            assert input_image.shape[0] == 3
            save_image(input_image, "src/videogen_hub/pipelines/seine/input_image.png")

        args = self.config

        # Setup PyTorch:
        if args.seed:
            torch.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

        if args.ckpt is None:
            raise ValueError("Please specify a checkpoint path using --ckpt <path>")

        # Load model:
        latent_h = args.image_size[0] // 8
        latent_w = args.image_size[1] // 8
        args.image_h = args.image_size[0]
        args.image_w = args.image_size[1]
        args.latent_h = latent_h
        args.latent_w = latent_w
        print('loading model')
        model = get_models(args).to(device)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # load model
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['ema']
        model.load_state_dict(state_dict)
        print('loading succeed')

        model.eval()
        pretrained_model_path = args.pretrained_model_path
        diffusion = create_diffusion(str(args.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
        text_encoder = TextEmbedder(pretrained_model_path).to(device)
        if args.use_fp16:
            print('Warnning: using half percision for inferencing!')
            vae.to(dtype=torch.float16)
            model.to(dtype=torch.float16)
            text_encoder.to(dtype=torch.float16)

        # prompt:
        prompt = args.text_prompt
        if prompt is None or prompt == []:
            prompt = args.input_path.split('/')[-1].split('.')[0].replace('_', ' ')
        else:
            prompt = prompt[0]
        prompt_base = prompt.replace(' ', '_')
        prompt = prompt + args.additional_prompt

        if save_video:
            if not os.path.exists(os.path.join(args.save_path)):
                os.makedirs(os.path.join(args.save_path))

        video_input, researve_frames = get_input(args)  # f,c,h,w
        video_input = video_input.to(device).unsqueeze(0)  # b,f,c,h,w

        mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device)  # b,f,c,h,w
        masked_video = video_input * (mask == 0)

        video_clip = auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model,
                                     device, )
        video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3,
                                                                                                               1)

        if save_video:
            save_video_path = os.path.join(args.save_path, prompt_base + '.mp4')
            torchvision.io.write_video(save_video_path, video_, fps=8)
            print(f'save in {save_video_path}')

        return video_.permute(0, 3, 1, 2)
