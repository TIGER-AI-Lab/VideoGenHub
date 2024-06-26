import os

from PIL import Image
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_i2v_infer_model import BaseI2vInferModel
from videogen_hub.pipelines.consisti2v.scripts.animate import create_pipeline as create_videogen_pipeline
from videogen_hub.pipelines.consisti2v.scripts.animate import main


class ConsistI2V(BaseI2vInferModel):

    def __init__(self, device="cuda"):
        class Args:
            def __init__(self):
                self.inference_config = "configs/inference/inference.yaml"
                self.prompt = None
                self.n_prompt = ""
                self.seed = "random"
                self.path_to_first_frame = None
                self.prompt_config = "configs/prompts/default.yaml"
                self.format = "mp4"
                self.save_model = False
                self.optional_args = []

        self.args = Args()
        self.resolution = [320, 512]
        self.config = None
        self.device = device
        from videogen_hub.pipelines.consisti2v.scripts.animate import create_pipeline

    def download_models(self) -> str:
        model_path = os.path.join(MODEL_PATH, "ConsistI2V")
        downloaded_path = snapshot_download("TIGER-Lab/ConsistI2V", local_dir=model_path)
        self.model_path = downloaded_path
        return downloaded_path

    def load_pipeline(self):
        if self.pipeline is None:
            self.download_models()
            model_path = self.model_path.replace("\\", "\\\\")
            yaml_config = f"""
                        output_dir: "samples/inference"
                        output_name: "i2v"
                        pretrained_model_path: "{model_path}"
                        unet_path: null
                        unet_ckpt_prefix: "module."
                        pipeline_pretrained_path: null
    
                        sampling_kwargs:
                        height: 256
                        width: 256
                        n_frames: 16
                        steps: 50
                        ddim_eta: 0.0
                        guidance_scale_txt: 7.5
                        guidance_scale_img: 1.0
                        guidance_rescale: 0.0
                        num_videos_per_prompt: 1
                        frame_stride: 3
    
                        unet_additional_kwargs:
                        variant: null
                        n_temp_heads: 8
                        augment_temporal_attention: true
                        temp_pos_embedding: "rotary" # "rotary" or "sinusoidal"
                        first_frame_condition_mode: "concat"
                        use_frame_stride_condition: true
                        noise_sampling_method: "pyoco_mixed" # "vanilla" or "pyoco_mixed" or "pyoco_progressive"
                        noise_alpha: 1.0
    
                        noise_scheduler_kwargs:
                        beta_start: 0.00085
                        beta_end: 0.012
                        beta_schedule: "linear"
                        steps_offset: 1
                        clip_sample: false
                        rescale_betas_zero_snr: false     # true if using zero terminal snr
                        timestep_spacing:       "leading" # "trailing" if using zero terminal snr
                        prediction_type:        "epsilon" # "v_prediction" if using zero terminal snr
    
                        frameinit_kwargs:
                        enable: true
                        camera_motion: null
                        noise_level: 850
                        filter_params:
                            method: 'gaussian'
                            d_s: 0.25
                            d_t: 0.25
                    """
            if self.config and self.pipeline:
                # Ensure that the height/width/n_frames match, and if so, just return the existing pipeline
                if self.config.sampling_kwargs.height == self.resolution[0] and self.config.sampling_kwargs.width == self.resolution[1] and self.config.sampling_kwargs.n_frames == self.seconds * self.fps:
                    return self.pipeline

            self.config = OmegaConf.create(yaml_config)
            self.config.sampling_kwargs.height = self.resolution[0]
            self.config.sampling_kwargs.width = self.resolution[1]
            self.config.sampling_kwargs.n_frames = self.seconds * self.fps
            self.pipeline = create_videogen_pipeline(self.config, self.device)
        self.to(self.device)
        return self.pipeline

    def infer_one_video(
            self,
            input_image: Image.Image,
            prompt: str = None,
            negative_prompt: str = None,
            size: list = None,
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
            unload: bool = True
    ):
        """
        Generates a single video based on a textual prompt and first frame image, using either a provided image or an image path as the starting point. The output is a tensor representing the video.

        Args:
            input_image (PIL.Image.Image): The input image to use as the basis for video generation.
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            negative_prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.
            unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True

        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        if size is not None:
            self.resolution = size

        self.args.prompt = prompt
        self.args.n_prompt = negative_prompt
        self.args.path_to_first_frame = input_image
        self.args.seed = str(seed)
        self.fps = fps
        self.seconds = seconds
        self.load_pipeline()
        result = main(self.args, self.config, self.pipeline)
        if unload:
            self.to("cpu")
        return result
