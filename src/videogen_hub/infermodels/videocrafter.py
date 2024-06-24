import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH


class VideoCrafter2:
    def __init__(self, device="cuda"):
        """
        1. Download the pretrained model and put it inside MODEL_PATH/videocrafter2
        2. Create Pipeline
        Args:
            device: 'cuda' or 'cpu' the device to use the model
        """
        from videogen_hub.pipelines.videocrafter.inference import VideoCrafterPipeline

        model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                     filename="model.ckpt",
                                     local_dir=os.path.join(MODEL_PATH, "videocrafter2"))
        config_path = str(Path(__file__).parent.parent.absolute())
        config_path = os.path.join(config_path, 'pipelines/videocrafter/inference_t2v_512_v2.0.yaml')

        arg_list = ['--mode', 'base',
                    '--ckpt_path', model_path,
                    '--config', config_path,
                    '--n_samples', '1',
                    '--bs', '1',
                    '--unconditional_guidance_scale', '12.0',
                    '--ddim_steps', '50',
                    '--ddim_eta', '1.0',
                    '--fps', '8']

        self.pipeline = VideoCrafterPipeline(arg_list, device, 0, 1)

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
            torch.Tensor: The generated video as a tensor, the shape being [num_frames, 3, height, width]

        """
        torch.manual_seed(seed)
        video = self.pipeline.run_inference(prompt,
                                            video_length=seconds * fps,
                                            height=size[0],
                                            width=size[1])

        return video.squeeze(0, 1).cpu().permute(1, 0, 2, 3)
