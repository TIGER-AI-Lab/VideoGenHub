
import os
from huggingface_hub import snapshot_download
import torch

from videogen_hub import MODEL_PATH

class LTXVideo():
    def __init__(self, device="cuda"):
        # requires approximately 60GB VRAM 
        from videogen_hub.pipelines.ltx_video.pipelines.pipeline_ltx_video import (
            LTXVideoPipeline
        )
        from transformers import T5EncoderModel, T5Tokenizer
        from videogen_hub.pipelines.ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
        from videogen_hub.pipelines.ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
        from videogen_hub.pipelines.ltx_video import load_unet, load_vae, load_scheduler

        _DIR = snapshot_download("Lightricks/LTX-Video", local_dir=MODEL_PATH, local_dir_use_symlinks=False, repo_type='model')
        self.unet_dir = os.path.join(_DIR, "unet")
        self.vae_dir = os.path.join(_DIR, "vae") 
        self.scheduler_dir = os.path.join(_DIR, "scheduler")

        self.vae = load_vae(self.vae_dir)
        self.unet = load_unet(self.unet_dir)
        self.scheduler = load_scheduler(self.scheduler_dir)
        self.patchifier = SymmetricPatchifier(patch_size=1)
        
        self.text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        )
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.to(device)
            
        self.tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
        )

        # Initialize pipeline
        submodel_dict = {
            "transformer": self.unet,
            "patchifier": self.patchifier, 
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "scheduler": self.scheduler,
            "vae": self.vae,
        }

        self.pipe = LTXVideoPipeline(**submodel_dict)

    def infer_one_video(self,
                        prompt: str = None,
                        size: list = [480, 704],
                        seconds: int = 5,
                        fps: int = 24,
                        seed: int = 42):

        from videogen_hub.pipelines.ltx_video.utils.conditioning_method import ConditioningMethod
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

        height = size[0]
        width = size[1]
        num_frames = seconds * fps

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1 

        video = self.pipe(
            height=height_padded,
            width=width_padded,
            #guidance_scale=3,
            num_frames=num_frames_padded,
            frame_rate=25,
            num_inference_steps=40,
            output_type="pt",
            callback_on_step_end=None,
            num_images_per_prompt=1,
            prompt=prompt,
            negative_prompt="",
            generator=generator,
            vae_per_channel_normalize=True,
            is_video=True,
            conditioning_method=ConditioningMethod.UNCONDITIONAL
        ).images[0]
        # C, F, H, W -> F, C, H, W
        video = video.permute(1, 0, 2, 3)
        print(video.shape) # [F, C, H, W]
        return video


