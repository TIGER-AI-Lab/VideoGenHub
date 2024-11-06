
import os
from huggingface_hub import snapshot_download, hf_hub_download

from videogen_hub import MODEL_PATH

class Mochi1():
    def __init__(self, device="cuda"):
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


