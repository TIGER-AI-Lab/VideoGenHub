<<<<<<< HEAD
class CogVideo:
    def __init__(self, device="cuda"):

        raise NotImplementedError
=======
import sys

from src.videogen_hub.pipelines.cogvideo.cogvideo_pipeline import pipeline

class CogVideo():
    def __init__(self, device="cuda"):
        
        
        import argparse

        # Manually creating an args object
        self.args = argparse.Namespace(
            generate_frame_num=5,
            coglm_temperature2=0.89,
            use_guidance_stage1=True,
            use_guidance_stage2=False,  # Assuming this is not set
            guidance_alpha=3.0,
            stage_1=False,  # Assuming this is not set
            stage_2=False,  # Assuming this is not set
            both_stages=True,
            parallel_size=1,
            stage1_max_inference_batch_size=-1,
            multi_gpu=False,  # Assuming this is not set
            device=3
        )
        
>>>>>>> 0f3e69d57f673956c2ecb77b4ac393597ee2d687

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
<<<<<<< HEAD
        raise NotImplementedError
=======
        
        return pipeline(self.args, raw_text=prompt, height=size[0], width=size[1], duration=seconds)
        
        # raise NotImplementedError

    
>>>>>>> 0f3e69d57f673956c2ecb77b4ac393597ee2d687
