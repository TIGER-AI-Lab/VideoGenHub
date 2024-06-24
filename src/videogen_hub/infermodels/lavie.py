import os, sys
import torch

from videogen_hub import MODEL_PATH


class LaVie():
    def __init__(self, model_path=os.path.join(MODEL_PATH, "lavie"), device="cuda"):
        """
        1. Download all necessary models from huggingface.
        2. Initializes the LaVie model with a specific model path and device.

        Args:
            model_path (str, optional): The path to the model checkpoints. Defaults to "MODEL_PATH/lavie".
            device (str, optional): The device to run the model on. Defaults to "cuda".
        """

        # Put the source code imports here to avoid dependency version issues
        from videogen_hub.pipelines.lavie.lavie_src.base.pipelines.pipeline_videogen import VideoGenPipeline
        from videogen_hub.pipelines.lavie.lavie_src.base.download import find_model
        from videogen_hub.pipelines.lavie.lavie_src.base.models.unet import UNet3DConditionModel
        from diffusers.schedulers import DDPMScheduler
        from diffusers.models import AutoencoderKL
        from transformers import CLIPTokenizer, CLIPTextModel
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf

        snapshot_download(repo_id="Vchitect/LaVie", local_dir=model_path)
        snapshot_download(repo_id="CompVis/stable-diffusion-v1-4", local_dir=os.path.join(model_path, "/stable-diffusion-v1-4"))
        snapshot_download(repo_id="stabilityai/stable-diffusion-x4-upscaler",
                          local_dir=os.path.join(model_path, "/stable-diffusion-x4-upscaler"))

        torch.set_grad_enabled(False)
        self.device = device

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

        sd_path = os.path.join(model_path, "stable-diffusion-v1-4")
        unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(device, dtype=torch.float16)
        state_dict = find_model(os.path.join(model_path, "lavie_base.pt"))
        unet.load_state_dict(state_dict)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
        tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder",
                                                         torch_dtype=torch.float16).to(device)  # huge

        scheduler = DDPMScheduler.from_pretrained(sd_path,
                                                  subfolder="scheduler",
                                                  beta_start=self.config.scheduler_config.beta_start,
                                                  beta_end=self.config.scheduler_config.beta_end,
                                                  beta_schedule=self.config.scheduler_config.beta_schedule)

        self.videogen_pipeline = VideoGenPipeline(vae=vae,
                                                  text_encoder=text_encoder_one,
                                                  tokenizer=tokenizer_one,
                                                  scheduler=scheduler,
                                                  unet=unet).to(device)
        self.videogen_pipeline.enable_xformers_memory_efficient_attention()

    def infer_one_video(self,
                        prompt: str = None,
                        size: list = [320, 512],
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        """
        Generates a single video based on the provided prompt and parameters.

        Args:
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            size (list, optional): The size of the video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """
        if seed is not None:
            torch.manual_seed(seed)
        videos = self.videogen_pipeline(prompt,
                                        video_length=seconds * fps,
                                        height=size[0],
                                        width=size[1],
                                        num_inference_steps=self.config.model_config.num_sampling_steps,
                                        guidance_scale=self.config.model_config.guidance_scale).video
        return videos[0]
