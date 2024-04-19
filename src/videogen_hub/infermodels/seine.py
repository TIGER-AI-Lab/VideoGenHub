from typing import Union
import torch
from huggingface_hub import snapshot_download, hf_hub_download


class SEINE():
    def __init__(self):
        """
        1. Download the pretrained model and put it inside checkpoints/videocrafter2
        2. Create Pipeline.
        """
        from videogen_hub.pipelines.seine.SEINEPipeline import SEINEPipeline

        seine_path = hf_hub_download(repo_id="Vchitect/SEINE", filename="seine.pt", local_dir="./checkpoints/SEINE")
        pretrained_model_path = snapshot_download(repo_id="CompVis/stable-diffusion-v1-4",
                                                  local_dir="./checkpoints/SEINE/stable-diffusion-v1-4")

        self.pipeline = SEINEPipeline(seine_path, pretrained_model_path,
                                      'src/videogen_hub/pipelines/seine/sample_i2v.yaml')

    def infer_one_video(self, input_image: Union[str, torch.Tensor],
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
        video = self.pipeline.infer_one_video(input_image=input_image,
                                              text_prompt=prompt,
                                              output_size=size,
                                              num_frames=seconds * fps,
                                              seed=seed)
        return video
