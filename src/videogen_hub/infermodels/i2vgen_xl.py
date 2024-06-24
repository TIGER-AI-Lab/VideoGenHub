import os

import torch
from PIL import Image
from diffusers import I2VGenXLPipeline
from huggingface_hub import snapshot_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel


class I2VGenXL(BaseI2vInferModel):
    def __init__(self):
        """
        Initializes the I2VGenXL model using the ali-vilab/i2vgen-xl checkpoint from the Hugging Face Hub.

        Args: None
        """
        self.resolution = [320, 512]
        self.model_path = os.path.join(MODEL_PATH, "i2vgen-xl")

    def download_models(self) -> str:
        model_path = snapshot_download("ali-vilab/i2vgen-xl", local_dir=self.model_path,
                                       ignore_patterns=["*fp16*", "*png"])
        self.model_path = model_path
        return model_path

    def load_pipeline(self):
        self.download_models()
        if self.pipeline is None:
            self.pipeline = I2VGenXLPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            )
        return self.pipeline

    def infer_one_video(
            self,
            input_image: Image.Image,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
    ):
        f"""
        Generates a single video based on a textual prompt and first frame image, using either a provided image or an image path as the starting point. The output is a tensor representing the video.

        Args:
            input_image (Image.Image): The input image path or tensor to use as the basis for video generation.
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            negative_prompt (str, optional): The negative text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to {self.resolution}.
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.

        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        if size is None:
            size = self.resolution

        self.load_pipeline()
        return self.pipeline(
            prompt=prompt,
            image=input_image,
            height=size[0],
            width=size[1],
            target_fps=fps,
            num_frames=seconds * fps,
            generator=torch.manual_seed(seed),
        )
