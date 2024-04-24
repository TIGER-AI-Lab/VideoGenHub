from typing import Union
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from PIL import *


class I2VGenXL:
    def __init__(self):
        """
        1. Download the pretrained model and put it inside checkpoints/videocrafter2
        2. Create Pipeline.
        """

        from diffusers import I2VGenXLPipeline

        self.pipeline = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
        )

    def infer_one_video(
        self,
        input_image: Image.Image,
        prompt: str = None,
        size: list = [320, 512],
        seconds: int = 2,
        fps: int = 8,
        seed: int = 42,
    ):
        """
        Generates a single video based on a textual prompt and first frame image, using either a provided image or an image path as the starting point. The output is a tensor representing the video.

        Args:
            input_image (Union[str, torch.Tensor]): The input image path or tensor to use as the basis for video generation.
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.

        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        return self.pipeline(
            prompt=prompt,
            image=input_image,
            height=size[0],
            width=size[1],
            target_fps=fps,
            num_frames=seconds * fps,
            generator=torch.manual_seed(seed),
        )