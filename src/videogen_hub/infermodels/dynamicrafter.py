import os

from PIL import Image
from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH


class DynamiCrafter:
    def __init__(self, version: str = "256"):
        """
        Initializes the DynamiCrafter model using the Doubiiu/DynamiCrafter_{version} checkpoint from the Hugging Face Hub.
        and load them to "MODEL_DIR/dynamicrafter_{version}_v1"

        Args:
            version (str, optional): The resolution of the video to generate. Choose from '256', '512', or '1024'. Defaults to '256'.
        """
        from videogen_hub.pipelines.dynamicrafter.inference import DynamiCrafterPipeline

        if version == "256":
            (self.height, self.width) = 256, 256
            self.fs = 3
            self.model_path = hf_hub_download(
                repo_id="Doubiiu/DynamiCrafter",
                filename="model.ckpt",
                local_dir=os.path.join(MODEL_PATH, "dynamicrafter_256_v1"),
            )

        elif version == "512":
            (self.height, self.width) = 320, 512
            self.fs = 24
            self.model_path = hf_hub_download(
                repo_id="Doubiiu/DynamiCrafter_512",
                filename="model.ckpt",
                local_dir=os.path.join(MODEL_PATH, "dynamicrafter_512_v1"),
            )

        elif version == "1024":
            (self.height, self.width) = 576, 1024
            self.fs = 10
            self.model_path = hf_hub_download(
                repo_id="Doubiiu/DynamiCrafter_1024",
                filename="model.ckpt",
                local_dir=os.path.join(MODEL_PATH, "dynamicrafter_1024_v1"),
            )
        else:
            raise ValueError("Invalid input. Please enter 256, 512, or 1024.")

        self.arg_list = [
            "--ckpt_path",
            self.model_path,
            "--config",
            f"src/videogen_hub/pipelines/dynamicrafter/configs/inference_{version}_v1.0.yaml",
            "--n_samples",
            "1",
            "--bs",
            "1",
            "--height",
            str(self.height),
            "--width",
            str(self.width),
            "--text_input",
            "--unconditional_guidance_scale",
            "7.5",
            "--ddim_steps",
            "50",
            "--ddim_eta",
            "1.0",
            "--video_length",
            "16",
            "--frame_stride",
            str(self.fs),
        ]

        self.pipeline = DynamiCrafterPipeline(self.arg_list)

    def infer_one_video(
            self,
            input_image: Image.Image,
            prompt: str = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
    ):
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
        self.pipeline.args.seed = seed
        self.pipeline.args.text_input = prompt
        self.pipeline.args.video_length = fps * seconds
        video = self.pipeline.run_inference(input_image)

        return video
