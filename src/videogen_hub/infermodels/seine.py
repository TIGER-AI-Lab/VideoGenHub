import os

import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download

from videogen_hub import MODEL_PATH


class SEINE():
    def __init__(self):
        """
        1. Download the pretrained model and put it inside MODEL_PATH/SEINE
        2. Create Pipeline.
        """
        from videogen_hub.pipelines.seine.SEINEPipeline import SEINEPipeline

        seine_path = hf_hub_download(repo_id="Vchitect/SEINE", filename="seine.pt", local_dir=os.path.join(MODEL_PATH, "SEINE"))
        pretrained_model_path = snapshot_download(repo_id="CompVis/stable-diffusion-v1-4",
                                                  local_dir=os.path.join(MODEL_PATH, "SEINE", "stable-diffusion-v1-4"))

        self.pipeline = SEINEPipeline(seine_path, pretrained_model_path,
                                      'src/videogen_hub/pipelines/seine/sample_i2v.yaml')

    def infer_one_video(self,
                        input_image: Image.Image,
                        prompt: str = None,
                        size: list = [320, 512],
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        """
        Generates a single video based on a textual prompt and first frame image, using either a provided image or an image path as the starting point. The output is a tensor representing the video.
    
        Args:
            input_image (PIL.Image.Image): The input image to use as the basis for video generation.
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.
    
        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        video = self.pipeline.infer_one_video(input_image=input_image,
                                              text_prompt=prompt,
                                              output_size=size,
                                              num_frames=seconds * fps,
                                              seed=seed)
        return video
