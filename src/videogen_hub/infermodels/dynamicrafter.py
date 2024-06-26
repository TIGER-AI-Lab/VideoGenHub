import os

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel
from videogen_hub.pipelines.dynamicrafter.inference import DynamiCrafterPipeline, load_model


class DynamiCrafter(BaseI2vInferModel):
    def __init__(self, version: str = "1024", device="cuda"):
        """
        Initializes the DynamiCrafter model using the Doubiiu/DynamiCrafter_{version} checkpoint from the Hugging Face Hub.
        and load them to "MODEL_DIR/dynamicrafter_{version}_v1"

        Args:
            version (str, optional): The resolution of the video to generate. Choose from '256', '512', or '1024'. Defaults to '256'.
        """
        self.model_path = os.path.join(MODEL_PATH, f"dynamicrafter_{version}_v1")
        self.device = device
        self.version = version

        if version == "256":
            (self.height, self.width) = 256, 256
            self.fs = 3

        elif version == "512":
            (self.height, self.width) = 320, 512
            self.fs = 24

        elif version == "1024":
            (self.height, self.width) = 576, 1024
            self.fs = 10

        else:
            raise ValueError("Invalid input. Please enter 256, 512, or 1024.")

        self.resolution = [self.height, self.width]

    def load_pipeline(self):
        if self.pipeline is None:
            self.model_path = self.download_models(self.version)
            self.pipeline = DynamiCrafterPipeline(self.model_path, device=self.device)
        self.to(self.device)
        return self.pipeline

    def download_models(self, version=None):
        if version:
            if version == "256":
                model_path = hf_hub_download(
                    repo_id="Doubiiu/DynamiCrafter",
                    filename="model.ckpt",
                    local_dir=os.path.join(MODEL_PATH, "dynamicrafter_256_v1"),
                )
            elif version == "512":
                model_path = hf_hub_download(
                    repo_id="Doubiiu/DynamiCrafter_512",
                    filename="model.ckpt",
                    local_dir=os.path.join(MODEL_PATH, "dynamicrafter_512_v1"),
                )
            elif version == "1024":
                model_path = hf_hub_download(
                    repo_id="Doubiiu/DynamiCrafter_1024",
                    filename="model.ckpt",
                    local_dir=os.path.join(MODEL_PATH, "dynamicrafter_1024_v1"),
                )
            else:
                raise ValueError("Invalid input. Please enter 256, 512, or 1024.")
            return model_path
        model_paths = []
        mp = hf_hub_download(
            repo_id="Doubiiu/DynamiCrafter",
            filename="model.ckpt",
            local_dir=os.path.join(MODEL_PATH, "dynamicrafter_256_v1"),
        )
        model_paths.append(mp)

        mp = hf_hub_download(
            repo_id="Doubiiu/DynamiCrafter_512",
            filename="model.ckpt",
            local_dir=os.path.join(MODEL_PATH, "dynamicrafter_512_v1"),
        )
        model_paths.append(mp)

        mp = hf_hub_download(
            repo_id="Doubiiu/DynamiCrafter_1024",
            filename="model.ckpt",
            local_dir=os.path.join(MODEL_PATH, "dynamicrafter_1024_v1"),
        )
        model_paths.append(mp)
        return model_paths

    def infer_one_video(
            self,
            input_image,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
            unload: bool = True
    ):
        """
        Generates a single video based on a textual prompt and first frame image, using either a provided image or an image path as the starting point. The output is a tensor representing the video.

        Args:
            input_image (PIL.Image.Image or str): The input image to use as the basis for video generation.
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            negative_prompt (str, optional): The negative text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.
            unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True

        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        if size is not None:
            self.height, self.width = size

        self.load_pipeline()

        video_length = seconds * fps

        video = self.pipeline(
            input_image=input_image,
            height=self.height,
            width=self.width,
            bs=1,
            video_length=video_length,
            seed=seed,
            text_input=prompt,
            interp=None,
            n_samples=1,
            ddim_steps=50,
            ddim_eta=0.0,
            unconditional_guidance_scale=7.5,
            cfg_img=None,
            frame_stride=self.fs,
            multiple_cond_cfg=None,
            loop=False,
            timestep_spacing=None,
            guidance_rescale=None
        )
        if unload:
            self.to("cpu")
        return video
