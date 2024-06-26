import os

from huggingface_hub import hf_hub_download
from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel
from videogen_hub.pipelines.opensora.scripts.open_sora_video_generation_pipeline import OpenSoraVideoGenerationPipeline
from mmengine import Config as mmConfig


class OpenSora12(BaseI2vInferModel):
    def __init__(self, device="cuda"):
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
            "num_frames": 51,  # Total number of frames in a clip
            "frame_interval": 1,  # Interval between frames
            "fps": 24,  # Frames per second
            "image_size": self.resolution,  # Resolution of each frame (height, width)

            # Model configuration for multi-resolution and specific model parameters
            "multi_resolution": "STDiT2",  # Multi-resolution model type
            "model": {
                "type": "STDiT3-XL/2",  # Model type and size
                "from_pretrained": os.path.join(MODEL_PATH, "STDiT3-XL_2"),  # Path to pretrained checkpoint
                "file_name": "model.safetensors",  # Name of the model file
                "input_sq_size": 512,  # Input square size for the model
                "qk_norm": True,  # Whether to normalize query-key in attention
                "enable_flashattn": False,  # Enable flash attention mechanism, require flash_attn package
                "enable_layernorm_kernel": False,  # Enable layer normalization in kernel, requires apex package
            },

            # Variational Autoencoder (VAE) specific settings
            "vae": {
                "type": "OpenSoraVAE_V1_2",  # Type of the autoencoder
                "from_pretrained": "hpcai-tech/OpenSora-VAE-v1.2",  # Pretrained model from Hugging Face
                "micro_frame_size": 17,
                "micro_batch_size": 4,  # Batch size for processing
            },

            # Text encoder settings for embedding textual information
            "text_encoder": {
                "type": "t5",  # Text encoder model type
                "from_pretrained": "DeepFloyd/t5-v1_1-xxl",  # Pretrained model
                "model_max_length": 300,  # Max length of text inputs
            },

            # Scheduler settings for diffusion models
            "scheduler": {
                "type": "rflow",  # Type of scheduler for the diffusion process
                "num_sampling_steps": 30,  # Number of sampling steps in diffusion
                "cfg_scale": 7.0,  # Scale for classifier-free guidance
                # "cfg_channel": 3,  # Number of channels for guidance
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
        self.config = mmConfig(self.config)
        self.pipeline = None  # Initialize the pipeline to None

    def download_models(self):
        model_paths = []
        mp = hf_hub_download(
            repo_id="hpcai-tech/OpenSora-STDiT-v3",
            filename="model.safetensors",
            local_dir=os.path.join(MODEL_PATH, "STDiT3-XL_2")
        )
        model_paths.append(mp)

        mp = hf_hub_download(
            repo_id="hpcai-tech/OpenSora-VAE-v1.2",
            filename="model.safetensors",
            local_dir=os.path.join(MODEL_PATH, "OpenSora-VAE-v1.2"),
        )
        model_paths.append(mp)

        mp = hf_hub_download(
            repo_id="DeepFloyd/t5-v1_1-xxl",
            filename="pytorch_model-00001-of-00002.bin",
            local_dir=os.path.join(MODEL_PATH, "t5-v1_1-xxl"),
        )
        model_paths.append(mp)
        return model_paths

    def load_pipeline(self):
        if self.pipeline is None:
            self.download_models()
            self.pipeline = OpenSoraVideoGenerationPipeline(self.config)
        self.to(self.device)
        return self.pipeline

    def infer_one_video(
            self,
            input_image: str = None,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
            unload: bool = True
    ):
        """
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
            unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True

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
        self.load_pipeline()
        video = self.pipeline(self.config)
        if unload:
            self.to("cpu")
        return video
