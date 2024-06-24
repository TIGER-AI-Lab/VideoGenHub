import os

from PIL.Image import Image
from huggingface_hub import hf_hub_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from videogen_hub.pipelines.opensora.scripts.inference import main
from mmengine import Config as mmengine_config

from videogen_hub.pipelines.opensora.scripts.open_sora_video_generation_pipeline import OpenSoraVideoGenerationPipeline


class OpenSora(BaseI2vInferModel):
    def __init__(self, device="gpu"):
        """
        1. Download the pretrained model and put it inside MODEL_PATH/modelscope
        2. Create Pipeline
        Note: it seems that the model needed from model_dir cannot support cpu
        Args:
            device: 'gpu' or 'cpu' the device to use the model
        """
        self.device = device
        self.resolution = [480, 854]

        self.config = {
            # Basic video frame settings
            "num_frames": 32,  # Total number of frames in a clip
            "frame_interval": 3,  # Interval between frames
            "fps": 24,  # Frames per second
            "image_size": self.resolution,  # Resolution of each frame (height, width)

            # Model configuration for multi-resolution and specific model parameters
            "multi_resolution": "STDiT2",  # Multi-resolution model type
            "model": {
                "type": "STDiT2-XL/2",  # Model type and size
                "from_pretrained": os.path.join(MODEL_PATH, "STDiT2-XL_2"),  # Path to pretrained checkpoint
                "file_name": "model.safetensors",  # Name of the model file
                "input_sq_size": 512,  # Input square size for the model
                "qk_norm": True,  # Whether to normalize query-key in attention
                "enable_flashattn": False,  # Enable flash attention mechanism, require flash_attn package
                "enable_layernorm_kernel": False,  # Enable layer normalization in kernel, requires apex package
            },

            # Variational Autoencoder (VAE) specific settings
            "vae": {
                "type": "VideoAutoencoderKL",  # Type of the autoencoder
                "from_pretrained": "stabilityai/sd-vae-ft-ema",  # Pretrained model from Hugging Face
                "cache_dir": os.path.join(MODEL_PATH, "sd-vae-ft-ema"),  # Local cache directory for model weights
                "micro_batch_size": 4,  # Batch size for processing
            },

            # Text encoder settings for embedding textual information
            "text_encoder": {
                "type": "t5",  # Text encoder model type
                "from_pretrained": "DeepFloyd/t5-v1_1-xxl",  # Pretrained model
                "cache_dir": os.path.join(MODEL_PATH, "t5-v1_1-xxl"),  # Cache directory
                "model_max_length": 200,  # Max length of text inputs
            },

            # Scheduler settings for diffusion models
            "scheduler": {
                "type": "iddpm",  # Type of scheduler for the diffusion process
                "num_sampling_steps": 50,  # Number of sampling steps in diffusion
                "cfg_scale": 7.0,  # Scale for classifier-free guidance
                "cfg_channel": 3,  # Number of channels for guidance
            },

            # Additional settings for processing and output
            "dtype": "bf16",  # Data type for computation (bfloat16)
            "prompt_path": None,  # Path to text prompts
            "prompt": [
                "A beautiful sunset over the city"
            ],  # List of prompts for generation
            "batch_size": 1,  # Batch size for generation
            "seed": 42,  # Seed for random number generators
            "save_dir": "./samples/samples/",  # Directory to save generated samples
            "config": "sample.py",  # Path to this configuration file
            "prompt_as_path": False,  # Treat the prompt as a file path (True/False)
            "reference_path": None,  # Path to reference image/video for conditioning
            "loop": 1,  # Number of times to loop the processing
            "sample_name": None,  # Specific name for the generated sample
            "num_sample": 1,  # Number of samples to generate

            # Additional configurations for completeness
            "resolution": None,
            "aspect_ratio": None,
            "start_index": 0,
            "end_index": None,
            "prompt_generator": "",
            "mask_strategy": [""],
            "save_fps": 24,  # Assuming default save_fps equals fps
            "llm_refine": False,
            "aes": None,
            "flow": None,
            "camera_motion": None,
            "watermark": False,
            "align": None,
            "verbose": 1,
            "condition_frame_length": 5,
            "condition_frame_edit": 0.0,
        }
        self.config = mmengine_config(self.config)

    def download_models(self):
        model_paths = []
        mp = hf_hub_download(
            repo_id="hpcai-tech/OpenSora-STDiT-v2-stage2",
            filename="model.safetensors",
            local_dir=self.config.model.from_pretrained,
        )
        model_paths.append(mp)

        mp = hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-ema",
            filename="diffusion_pytorch_model.safetensors",
            local_dir=self.config.vae.cache_dir,
        )
        model_paths.append(mp)

        mp = hf_hub_download(
            repo_id="DeepFloyd/t5-v1_1-xxl",
            filename="pytorch_model-00001-of-00002.bin",
            local_dir=self.config.text_encoder.cache_dir,
        )
        model_paths.append(mp)
        return model_paths

    def load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = OpenSoraVideoGenerationPipeline(self.config)

    def infer_one_video(
            self,
            input_image: str = None,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
    ):
        f"""
        Generates a single video based on the provided prompt and parameters.
        The generated video always has resolution 256x256

        Args:
            input_image (str, optional): The input image path or tensor to use as the basis for video generation.
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            negative_prompt (str, optional): The negative text prompt to generate the video from. Defaults to None.
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            size (list, optional): The size of the video as [height, width]. Defaults to {self.resolution}.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """

        self.config.num_frames = fps * seconds
        self.config.reference_path = input_image
        self.config.fps = fps
        self.config.seed = seed
        self.config.prompt = [prompt]
        if not size:
            size = self.resolution
        self.config.image_size = size

        all_batch_samples = self.pipeline(self.config)

        sample = all_batch_samples[0][0]
        # sample is torch.Size([1, C, f, H, W])

        output = sample.squeeze(0).permute(1, 2, 3, 0).cpu().float()
        # torch.Size([1, C, f, H, W]) -> torch.Size([f, H, W, C])
        # BFloat16 -> Float

        return output
