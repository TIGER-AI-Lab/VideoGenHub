import torch

class CogVideoX:
    def __init__(self, weight="THUDM/CogVideoX-2b", device="cuda"):
        """
        Initializes the CogVideo model with a specific device.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
        """
        from diffusers import CogVideoXPipeline

        self.pipe = CogVideoXPipeline.from_pretrained(weight, device_map="balanced")
        self.device = device

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
        #self.pipe.to(self.device)
        video = self.pipe(prompt=prompt, 
                        guidance_scale=6,
                        num_frames=seconds * fps, 
                        #height=size[0],
                        #width=size[1],
                        num_inference_steps=50,
                        generator=torch.manual_seed(seed)).frames[0]
        from videogen_hub.utils import images_to_tensor
        video = video[:-1] # drop the last frame
        video = images_to_tensor(video) # parse it back to tensor (T, C, H, W)
        
        return video

class CogVideoX5B(CogVideoX):
    def __init__(self, device="cuda"):
        # Require diffusers>=0.30.1
        super().__init__(weight="THUDM/CogVideoX-5b", device=device)

    def infer_one_video(
        self,
        prompt: str = None,
        size: list = [320, 512],
        seconds: int = 2,
        fps: int = 8,
        seed: int = 42,
    ):
        return super().infer_one_video(prompt, size, seconds, fps, seed)    