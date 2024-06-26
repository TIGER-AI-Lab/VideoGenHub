import os

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel
from videogen_hub.pipelines.opensora.scripts.open_sora_video_generation_pipeline import OpenSoraVideoGenerationPipeline
from mmengine import Config as mmConfig


class Svd(BaseI2vInferModel):
    def __init__(self, version="xt1", device="cuda"):
        """
        1. Download the pretrained model and put it inside MODEL_PATH/modelscope
        2. Create Pipeline
        Note: it seems that the model needed from model_dir cannot support cpu
        Args:
            device: 'gpu' or 'cpu' the device to use the model
        """
        self.device = device
        self.resolution = [480, 854]
        self.version = version
        self.model_dir = os.path.join(MODEL_PATH, "svd")
        self.model_path = None

        self.pipeline = None  # Initialize the pipeline to None
        self.model_dict = models = {
            "xt": {
                "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
                "filename": "svd_xt.safetensors",
                "frames": 25
            },
            "xt11": {
                "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
                "filename": "svd_xt_1_1.safetensors",
                "frames": 25
            },
            "svd128": {
                "repo": "ECNU-CILab/ExVideo-SVD-128f-v1",
                "filename": "model.fp16.safetensors",
                "frames": 128
            }
        }

    def download_models(self, model=None):
        if model:
            model_dict = {model: self.model_dict[model]}
        else:
            model_dict = self.model_dict
        model_paths = []
        for model, model_info in model_dict.items():
            model_path = hf_hub_download(
                repo_id=model_info["repo"],
                local_dir=self.model_dir,
                filename=model_info["filename"],
            )
            model_paths.append(model_path)
        if model:
            self.model_path = model_paths[0]
        return model_paths

    def load_pipeline(self):
        if self.pipeline is None:
            self.download_models()
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            )
            self.pipeline.enable_model_cpu_offload()
            # TODO: Test if this makes things faster
            self.pipeline.unet.enable_forward_chunking()
        self.to(self.device)
        return self.pipeline

    def infer_one_video(
            self,
            input_image: str = None,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
            unload: bool = True,
            motion_bucket_id: int = 180,
            noise_aug_strength: float = 0.1,
    ):
        """
        Generates a single video based on the provided prompt and parameters.
        The generated video always has resolution 256x256

        Args:
            input_image (str, optional): The input image path or tensor to use as the basis for video generation.
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            negative_prompt (str, optional): The negative text prompt to generate the video from. Defaults to None.
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            size (list, optional): The size of the video as [height, width]. Defaults to {self.resolution}.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.
            unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True
            motion_bucket_id (int, optional): The motion bucket id to use for video generation. Defaults to 180.
            noise_aug_strength (float, optional): The strength of the noise augmentation. Defaults to 0.1.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """
        if not size:
            size = self.resolution
        self.load_pipeline()
        # Load the conditioning image
        image = load_image(input_image)
        generator = torch.manual_seed(seed)
        num_frames = fps * seconds
        video = self.pipeline(
            image,
            height=size[0],
            width=size[1],
            num_frames=num_frames,
            output_type="np",
            decode_chunk_size=8,
            generator=generator
        ).frames[0]
        video = torch.from_numpy(video).squeeze(0).permute(0, 3, 1, 2)
        if unload:
            self.to("cpu")
        return video
