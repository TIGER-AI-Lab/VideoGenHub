# Adapted from https://github.com/luosiallen/latent-consistency-model
from __future__ import annotations

import os
import random
from omegaconf import OmegaConf

import numpy as np

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from .utils.lora import collapse_lora, monkeypatch_remove_lora
from .utils.lora_handler import LoraHandler
from .utils.common_utils import load_model_checkpoint
from .utils.utils import instantiate_from_config
from .scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from .pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline

import torch

DESCRIPTION = """# T2V-Turbo 🚀
We provide T2V-Turbo (VC2) distilled from [VideoCrafter2](https://ailab-cvc.github.io/videocrafter2/) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master) and [InternVid2 Stage 2 Model](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4).

You can download the the models from [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2). Check out our [Project page](https://t2v-turbo.github.io) 😄
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"

"""
Operation System Options:
    If you are using MacOS, please set the following (device="mps") ;
    If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
    If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"  # Linux & Windows

"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = (
    torch.float16
)  # torch.float16 works as well, but pictures seem to be a bit worse


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


class T2VTurboVC2Pipeline1:
    def __init__(self, config, device, unet_dir, base_model_dir):
        config = OmegaConf.create(config)
        model_config = config.pop("model", OmegaConf.create())
        pretrained_t2v = instantiate_from_config(model_config)
        pretrained_t2v = load_model_checkpoint(pretrained_t2v, base_model_dir)

        unet_config = model_config["params"]["unet_config"]
        unet_config["params"]["time_cond_proj_dim"] = 256
        unet = instantiate_from_config(unet_config)

        unet.load_state_dict(
            pretrained_t2v.model.diffusion_model.state_dict(), strict=False
        )

        use_unet_lora = True
        lora_manager = LoraHandler(
            version="cloneofsimo",
            use_unet_lora=use_unet_lora,
            save_for_webui=True,
            unet_replace_modules=["UNetModel"],
        )
        lora_manager.add_lora_to_model(
            use_unet_lora,
            unet,
            lora_manager.unet_replace_modules,
            lora_path=unet_dir,
            dropout=0.1,
            r=64,
        )
        unet.eval()
        collapse_lora(unet, lora_manager.unet_replace_modules)
        monkeypatch_remove_lora(unet)

        pretrained_t2v.model.diffusion_model = unet
        scheduler = T2VTurboScheduler(
            linear_start=model_config["params"]["linear_start"],
            linear_end=model_config["params"]["linear_end"],
        )
        self.pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
        self.pipeline.to(device)

    def inference(
            self,
            prompt: str,
            height: int = 320,
            width: int = 512,
            seed: int = 0,
            guidance_scale: float = 7.5,
            num_inference_steps: int = 4,
            num_frames: int = 16,
            fps: int = 16,
            randomize_seed: bool = False,
            param_dtype="torch.float16"
    ):
        seed = randomize_seed_fn(seed, randomize_seed)
        torch.manual_seed(seed)
        self.pipeline.to(
            torch_device=device,
            torch_dtype=torch.float16 if param_dtype == "torch.float16" else torch.float32,
        )

        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
        )

        return result
