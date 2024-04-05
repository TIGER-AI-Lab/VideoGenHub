# Custom pipeline file for VideoGenHub

import os, sys
import torch
import torchvision

from .lavie_src.base.pipelines.pipeline_videogen import VideoGenPipeline
from .lavie_src.base.download import find_model
from .lavie_src.base.models import get_models
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf

class LavieBasePipeline():
    def __init__(self) -> None:
        raise NotImplementedError