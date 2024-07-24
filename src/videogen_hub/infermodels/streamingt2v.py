import os

from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from videogen_hub.pipelines.streamingt2v.streamingt2v_pipeline import pipeline


class StreamingT2V(BaseI2vInferModel):
    def __init__(self, device="cuda"):
        """
        Initializes the StreamingT2V model.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
        """
        self.device = device
        self.model_path = os.path.join(MODEL_PATH, "streamingtv2")
        self.resolution = [320, 512]
        self.pipeline = None

    def download_models(self) -> str:
        ckpt_file_streaming_t2v = hf_hub_download(repo_id="PAIR/StreamingT2V",
                                                  filename="streaming_t2v.ckpt",
                                                  local_dir=os.path.join(MODEL_PATH, "streamingtv2"))
        self.model_path = ckpt_file_streaming_t2v
        return self.model_path

    def load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = pipeline
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
            unload: bool = True
    ):
        """
        Generates a single video based on the provided prompt and parameters.

        Args:
            input_image (str, optional): The input image path or tensor to use as the basis for video generation.
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

        if not size:
            size = self.resolution

        if not input_image:
            input_image = ""

        self.load_pipeline()

        video = self.pipeline(input_image, prompt, negative_prompt, size, seconds, fps, seed)
        if unload:
            self.to("cpu")
        return video
