
import os
from huggingface_hub import snapshot_download, hf_hub_download
import warnings
import torch

from videogen_hub import MODEL_PATH

class Mochi1():
    def __init__(self, device="cuda"):
        from diffusers import MochiPipeline
        self.pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()


    def infer_one_video(self,
                        prompt: str = None,
                        size: list = [480, 848],
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        frames = self.pipe(prompt, 
                           num_inference_steps=28, 
                           guidance_scale=3.5,
                           generator=torch.manual_seed(seed)).frames[0]
        return frames

class Mochi1_source():
    def __init__(self, device="cuda"):
        
        warnings.warn(
            "Mochi1 is deprecated and will be removed in a future version. "
            "Please use the official Mochi model from genmo/mochi instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # requires approximately 60GB VRAM 
        from videogen_hub.pipelines.genmo.mochi_preview.pipelines import (
            DecoderModelFactory,
            DitModelFactory,
            MochiSingleGPUPipeline,
            T5ModelFactory,
            linear_quadratic_schedule,
        )

        MOCHI_DIR = snapshot_download('genmo/mochi-1-preview', 
                                      local_dir = os.path.join(MODEL_PATH, 'mochi-1-preview'))

        self.pipe = MochiSingleGPUPipeline(
        text_encoder_factory=T5ModelFactory(),
        dit_factory=DitModelFactory(model_path=os.path.join(MOCHI_DIR, "dit.safetensors"), model_dtype="bf16"),
        decoder_factory=DecoderModelFactory(model_path=os.path.join(MOCHI_DIR, "decoder.safetensors")),
        cpu_offload=True,
        decode_type="tiled_full"
        )
        self.scheduler = linear_quadratic_schedule(64, 0.025)

    def infer_one_video(self,
                        prompt: str = None,
                        size: list = [480, 848],
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        warnings.warn(
            "This method is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        video = self.pipe(
            height=size[0],
            width=size[1],
            num_frames=31,
            num_inference_steps=64,
            sigma_schedule=self.scheduler,
            cfg_schedule=[4.5] * 64,
            batch_cfg=False,
            prompt=prompt,
            negative_prompt="",
            seed=seed,
        )
        return video
