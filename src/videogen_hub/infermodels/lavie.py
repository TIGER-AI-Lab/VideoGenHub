import os

import torch

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from videogen_hub.pipelines.lavie.lavie_src.base.pipelines.pipeline_videogen import VideoGenPipeline
from videogen_hub.pipelines.lavie.lavie_src.base.download import find_model
from videogen_hub.pipelines.lavie.lavie_src.base.models.unet import UNet3DConditionModel
from diffusers.schedulers import DDPMScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


class LaVie(BaseT2vInferModel):
    def __init__(self, device="cuda"):
        """
        1. Download all necessary models from huggingface.
        2. Initializes the LaVie model with a specific model path and device.

        Args:
            model_path (str, optional): The path to the model checkpoints. Defaults to "MODEL_PATH/lavie".
            device (str, optional): The device to run the model on. Defaults to "cuda".
        """

        # Put the source code imports here to avoid dependency version issues

        torch.set_grad_enabled(False)
        self.resolution = [320, 512]
        self.device = device
        self.model_path = os.path.join(MODEL_PATH, "lavie")

        config = {
            "model_config": {
                "use_compile": False,
                "use_fp16": True,
                "run_time": 0,
                "guidance_scale": 7.5,
                "num_sampling_steps": 50
            },
            "scheduler_config": {
                "sample_method": "ddpm",
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "linear"
            }
        }
        self.config = OmegaConf.create(config)

    def load_pipeline(self):
        if self.pipeline is None:
            self.download_models()
            sd_path = os.path.join(self.model_path, "stable-diffusion-v1-4")
            unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(self.device, dtype=torch.float16)
            state_dict = find_model(os.path.join(self.model_path, "lavie_base.pt"))
            unet.load_state_dict(state_dict)

            vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(self.device)
            tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
            text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder",
                                                             torch_dtype=torch.float16).to(self.device)  # huge

            scheduler = DDPMScheduler.from_pretrained(sd_path,
                                                      subfolder="scheduler",
                                                      beta_start=self.config.scheduler_config.beta_start,
                                                      beta_end=self.config.scheduler_config.beta_end,
                                                      beta_schedule=self.config.scheduler_config.beta_schedule)

            self.pipeline = VideoGenPipeline(vae=vae,
                                             text_encoder=text_encoder_one,
                                             tokenizer=tokenizer_one,
                                             scheduler=scheduler,
                                             unet=unet).to(self.device)
            self.pipeline.enable_xformers_memory_efficient_attention()
        self.to(self.device)
        return self.pipeline

    def download_models(self):
        model_paths = []
        mp = snapshot_download(repo_id="Vchitect/LaVie", local_dir=self.model_path,
                               ignore_patterns=["*fp16*", "*non_ema.bin", "diffusion_pytorch_model.bin", "diffusion_pytorch_model_fp16.bin"])
        model_paths.append(mp)
        mp = snapshot_download(repo_id="CompVis/stable-diffusion-v1-4",
                               local_dir=os.path.join(self.model_path, "stable-diffusion-v1-4"),
                               ignore_patterns=["*fp16*", "*non_ema.bin", "diffusion_pytorch_model.bin", "diffusion_pytorch_model_fp16.bin"])
        model_paths.append(mp)
        mp = snapshot_download(repo_id="stabilityai/stable-diffusion-x4-upscaler",
                               local_dir=os.path.join(self.model_path, "stable-diffusion-x4-upscaler"),
                               ignore_patterns=["*fp16*", "*non_ema.bin", "diffusion_pytorch_model.bin", "diffusion_pytorch_model_fp16.bin"]
        )
        model_paths.append(mp)
        return model_paths

    def infer_one_video(self,
                        prompt: str = None,
                        negative_prompt: str = None,
                        size: list = None,
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42,
                        unload: bool = True
                        ):
        """
        Generates a single video based on the provided prompt and parameters.

        Args:
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            negative_prompt (str, optional): The negative text prompt to generate the video from. Defaults to None.
            size (list, optional): The size of the video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.
            unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True

        Returns:
            torch.Tensor: The generated video as a tensor.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.load_pipeline()
        videos = self.pipeline(prompt,
                               negative_prompt=negative_prompt,
                               video_length=seconds * fps,
                               height=size[0],
                               width=size[1],
                               num_inference_steps=self.config.model_config.num_sampling_steps,
                               guidance_scale=self.config.model_config.guidance_scale).video
        if unload:
            self.to("cpu")
        return videos[0]
