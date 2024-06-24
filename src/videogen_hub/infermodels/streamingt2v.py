import os

from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH


class StreamingT2V:
    def __init__(self, device="cuda"):
        """
        Initializes the StreamingT2V model.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
        """

        from videogen_hub.pipelines.streamingt2v.streamingt2v_pipeline import pipeline
        # https://huggingface.co/spaces/PAIR/StreamingT2V/resolve/main/t2v_enhanced/checkpoints/streaming_t2v.ckpt?download=true
        model_url = "https://huggingface.co/spaces/PAIR/StreamingT2V/resolve/main/t2v_enhanced/checkpoints/streaming_t2v.ckpt?download=true"
        # Download the file
        ckpt_file_streaming_t2v = hf_hub_download(repo_id="PAIR/StreamingT2V",
                                                  filename="streaming_t2v.ckpt",
                                                  local_dir=os.path.join(MODEL_PATH, "streamingtv2"))

        self.pipeline = pipeline

    def infer_one_video(
            self,
            prompt: str = None,
            size: list = [320, 512],
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
    ):
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

        return self.pipeline(prompt, size, seconds, fps, seed)
