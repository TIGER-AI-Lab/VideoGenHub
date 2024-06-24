import os

import torch
from huggingface_hub import snapshot_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from modelscope.pipelines import pipeline
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from decord import VideoReader
from decord import cpu
import io


class ModelScope(BaseT2vInferModel):
    def __init__(self, device="gpu"):
        """
        1. Download the pretrained model and put it inside checkpoints/modelscope
        2. Create Pipeline
        Note: it seems that the model needed from model_dir cannot support cpu
        Args:
            device: 'gpu' or 'cpu' the device to use the model
        """
        self.model_path = os.path.join(MODEL_PATH, "modelscope")
        self.device = device
        self.resolution = [256, 256]
        self.model = None

    def download_models(self):
        model_dir = snapshot_download(
            repo_id="ali-vilab/modelscope-damo-text-to-video-synthesis",
            local_dir=self.model_path,

        )
        self.model_path = model_dir
        return model_dir

    def load_pipeline(self):
        self.download_models()
        # Store both model and pipeline in self so we can manipulate them later if desired
        if not self.pipeline or not self.model:
            self.model = Model.from_pretrained(self.model_path)
            self.pipeline = pipeline("text-to-video-synthesis", model=self.model, device=self.device)
        self.pipeline.model.config.model.model_args.max_frames = self.fps * self.seconds
        self.model.config.model.model_args.max_frames = self.fps * self.seconds

    def infer_one_video(
            self,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42
    ):
        """
        Generates a single video based on the provided prompt and parameters.
        The generated video always has resolution 256x256

        Args:
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            negative_prompt (str, optional): The negative text prompt to generate the video from. Defaults to None.
            size (list, optional): The resolution of the video as [height, width]. Defaults to None.
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """

        # if not size:
        #     size = self.resolution
        #
        torch.manual_seed(seed)
        self.fps = fps
        self.seconds = seconds
        self.load_pipeline()

        test_text = {
            "text": prompt,
        }
        output_video_path = self.pipeline(test_text,)[OutputKeys.OUTPUT_VIDEO]
        result = io.BytesIO(output_video_path)
        result = VideoReader(result, ctx=cpu(0))
        result = torch.from_numpy(result.get_batch(range(len(result))).asnumpy())
        return result
