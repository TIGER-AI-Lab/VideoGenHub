import torch


class VideoCrafter2():
    def __init__(self, device="cuda"):
        """
        1. Download the pretrained model and put it inside checkpoints/videocrafter2
        2.
        Args:
            device:
        """
        from videogen_hub.pipelines.videocrafter.download import fetch_checkpoint
        from videogen_hub.pipelines.videocrafter.inference import VideoCrafterPipeline

        model_path = fetch_checkpoint('videocrafter2')

        config = {
            "mode": "base",
            "ckpt_path": model_path,
            "config": "videogen_hub/pipelines/videocrafter/inference_t2v_512_v2.0.yaml",
            "n_samples": 1,
            "bs": 1,
            "unconditional_guidance_scale": 12.0,
            "ddim_steps": 50,
            "ddim_eta": 1.0,
            "fps": 8
        }

        self.pipeline = VideoCrafterPipeline(config, device, 0, 1)

    def infer_one_video(self,
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
        torch.manual_seed(seed)
        video = self.pipeline.run_inference(prompt,
                                            video_length=seconds * fps,
                                            height=size[0],
                                            width=size[1])

        raise video
