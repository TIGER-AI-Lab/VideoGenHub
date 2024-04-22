import torch
import torchvision

# from modelscope.outputs import OutputKeys


class ModelScope:
    def __init__(self, device="gpu", revision="v1.1.0"):
        """
        1. Download the pretrained model and put it inside checkpoints/modelscope
        2. Create Pipeline
        Note: it seems that the model needed from model_dir cannot support cpu
        Args:
            device: 'gpu' or 'cpu' the device to use the model
        """

        from modelscope.pipelines import pipeline
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.models import Model

        model_dir = snapshot_download(
            "damo/text-to-video-synthesis",
            revision=revision,
            cache_dir="./checkpoints/modelscope",
        )
        model = Model.from_pretrained(model_dir)
        self.pipeline = pipeline("text-to-video-synthesis", model=model, device=device)

    def infer_one_video(
        self, prompt: str = None, seconds: int = 2, fps: int = 8, seed: int = 42
    ):
        """
        Generates a single video based on the provided prompt and parameters.
        The generated video always has resolution 256x256

        Args:
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """
        torch.manual_seed(seed)
        self.pipeline.model.config.model.model_args.max_frames = fps * seconds

        test_text = {
            "text": prompt,
        }
        output_video_path = self.pipeline(
            test_text,
        )[OutputKeys.OUTPUT_VIDEO]
        result = torchvision.io.read_video(output_video_path, output_format="TCHW")[0]

        return result
