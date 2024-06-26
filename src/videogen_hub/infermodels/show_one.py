import os

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from videogen_hub.pipelines.show_1.run_inference import ShowOnePipeline
from huggingface_hub import snapshot_download


class ShowOne(BaseT2vInferModel):
    def __init__(self, device="cuda"):
        self.resolution = [320, 512]
        self.pipeline = None
        self.model_path = os.path.join(MODEL_PATH, "showlab")
        self.device = device

    def download_models(self) -> str:
        base_path = snapshot_download(
            repo_id="showlab/show-1-base",
            local_dir=os.path.join(MODEL_PATH, "showlab", "show-1-base"),
            local_dir_use_symlinks=False
        )

        interp_path = snapshot_download(
            repo_id="showlab/show-1-interpolation",
            local_dir=os.path.join(MODEL_PATH, "showlab", "show-1-interpolation"),

        )

        deepfloyd_path = snapshot_download(
            repo_id="DeepFloyd/IF-II-L-v1.0",
            local_dir=os.path.join(MODEL_PATH, "DeepFloyd/IF-II-L-v1.0"),

        )

        sr1_path = snapshot_download(
            repo_id="showlab/show-1-sr1",
            local_dir=os.path.join(MODEL_PATH, "showlab", "show-1-sr1"),

        )

        sr2_path = snapshot_download(
            repo_id="showlab/show-1-sr2",
            local_dir=os.path.join(MODEL_PATH, "showlab", "show-1-sr2"),

        )
        model_paths = [base_path, interp_path, deepfloyd_path, sr1_path, sr2_path]
        return model_paths

    def load_pipeline(self):
        if self.pipeline is None:
            model_paths = self.download_models()
            base_path, interp_path, deepfloyd_path, sr1_path, sr2_path = model_paths
            self.pipeline = ShowOnePipeline(base_path, interp_path, deepfloyd_path, sr1_path, sr2_path)
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
        Generates a single video based on a textual prompt. The output is a tensor representing the video.
        Since the initial_num_frames is set to be 8 as shown in paper in the pipeline,
        we need the (number of frames - 1) divisible by 7 to manage interpolation.
    
        Args:
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            negative_prompt (str, optional): A negative text prompt that guides the video generation. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.
            unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True
    
        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        if size is None:
            size = self.resolution

        self.load_pipeline()

        num_frames = fps * seconds

        assert (num_frames - 1) % 7 == 0
        scaling_factor = (num_frames - 1) // 7
        video = self.pipeline.inference(prompt=prompt,
                                        negative_prompt="",
                                        output_size=size,
                                        initial_num_frames=8,
                                        scaling_factor=scaling_factor,
                                        seed=seed)

        if unload:
            self.to("cpu")

        return video
