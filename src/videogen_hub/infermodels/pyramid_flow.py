import torch
from torchvision.transforms.functional import pil_to_tensor
from huggingface_hub import snapshot_download
from videogen_hub import MODEL_PATH

class PyramidFlow:
    def __init__(self, device="cuda"):
        """
        1. Download the pretrained model and put it inside MODEL_PATH/modelscope
        2. Create Pipeline
        Note: it seems that the model needed from model_dir cannot support cpu
        Args:
            device: 'gpu' or 'cpu' the device to use the model
        """

        from videogen_hub.pipelines.pyramid_flow.pyramid_dit import PyramidDiTForVideoGeneration
        from diffusers.utils import load_image, export_to_video
        snapshot_download("rain1011/pyramid-flow-sd3", local_dir=MODEL_PATH, local_dir_use_symlinks=False, repo_type='model')

        torch.cuda.set_device(0)
        self.model_dtype, self.torch_dtype = 'bf16', torch.bfloat16   # Use bf16 (not support fp16 yet)

        self.model = PyramidDiTForVideoGeneration(
            MODEL_PATH,                                         # The downloaded checkpoint dir
            self.model_dtype,
            model_variant='diffusion_transformer_768p',     # 'diffusion_transformer_384p'
        )

        self.model.vae.to(device)
        self.model.dit.to(device)
        self.model.text_encoder.to(device)
        self.model.vae.enable_tiling()


    def infer_one_video(
        self,
        prompt: str = None,
        size: list = [768, 1280],
        seconds: int = 5,
        fps: int = 24,
        seed: int = 42,
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
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=self.torch_dtype):
            frames = self.model.generate(
                prompt=prompt,
                num_inference_steps=[20, 20, 20],
                video_num_inference_steps=[10, 10, 10],
                height=768,    #  768
                width=1280, # 1280
                temp=16,                    # temp=16: 5s, temp=31: 10s
                guidance_scale=9.0,         # The guidance for the first frame, set it to 7 for 384p variant
                video_guidance_scale=5.0,   # The guidance for the other video latent
                output_type="pil",
                save_memory=True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
                generator=torch.manual_seed(seed)
            )
            # 5s
        # Turn frames from a list of PIL images into torch tensor
        frames_tensor = torch.stack([pil_to_tensor(frame) for frame in frames])
        # (time, channel, height, width)
        return frames_tensor