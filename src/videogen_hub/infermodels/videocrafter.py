import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from videogen_hub.pipelines.videocrafter.inference import VideoCrafterPipeline


class VideoCrafter2(BaseT2vInferModel):
    def __init__(self, device="cuda"):
        self.device = device
        self.pipeline = None
        self.resolution = [320, 512]
        self.model_path = os.path.join(MODEL_PATH, "videocrafter2")

    def download_models(self) -> str:
        self.model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                          filename="model.ckpt",
                                          local_dir=os.path.join(MODEL_PATH, "videocrafter2"))
        return self.model_path

    def load_pipeline(self):
        if self.pipeline is None:
            self.download_models()
            config_path = str(Path(__file__).parent.parent.absolute())
            config_path = os.path.join(config_path, 'pipelines/videocrafter/inference_t2v_512_v2.0.yaml')

            arg_list = ['--mode', 'base',
                        '--ckpt_path', self.model_path,
                        '--config', config_path,
                        '--n_samples', '1',
                        '--bs', '1',
                        '--unconditional_guidance_scale', '12.0',
                        '--ddim_steps', '50',
                        '--ddim_eta', '1.0',
                        '--fps', '8']

            self.pipeline = VideoCrafterPipeline(arg_list, self.device, 0, 1)
        self.to(self.device)
        return self.pipeline

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
            torch.Tensor: The generated video as a tensor, the shape being [num_frames, 3, height, width]

        """
        if size is None:
            size = self.resolution
        self.load_pipeline()
        torch.manual_seed(seed)
        video = self.pipeline.run_inference(prompt,
                                            video_length=seconds * fps,
                                            height=size[0],
                                            width=size[1])

        video = video.squeeze(0, 1).cpu().permute(1, 0, 2, 3)
        if unload:
            self.to("cpu")
        return video
