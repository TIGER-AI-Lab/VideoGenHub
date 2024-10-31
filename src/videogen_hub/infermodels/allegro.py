import torch

class Allegro:
    def __init__(self, weight="rhymes-ai/Allegro", device="cuda"):
        """
        pip install git+https://github.com/huggingface/diffusers.git
        Initializes the Allegro model with a specific device.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
        """
        from diffusers import AutoencoderKLAllegro, AllegroPipeline

        vae = AutoencoderKLAllegro.from_pretrained(weight, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = AllegroPipeline.from_pretrained(weight, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.pipe.vae.enable_tiling()
        # NotImplementedError: Decoding without tiling has not been implemented yet.
        self.device = device

    def infer_one_video(
        self,
        prompt: str = None,
        size: list = [720, 1280],
        seconds: int = 2,
        fps: int = 15,
        seed: int = 42,
    ):
        """
        Generates a single video based on the provided prompt and parameters.

        Args:
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            size (list, optional): The size of the video as [height, width]. 
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """
        #self.pipe.to(self.device)
        video = self.pipe(prompt, 
                          guidance_scale=7.5, 
                          max_sequence_length=512, 
                          num_inference_steps=20, # Default 100 but its way too long
                          generator=torch.manual_seed(seed)
                          ).frames[0]
        
        from videogen_hub.utils import images_to_tensor
        video = images_to_tensor(video) # parse it back to tensor (T, C, H, W)

        return video