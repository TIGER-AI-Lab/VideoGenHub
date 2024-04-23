import torch
from typing import Union
from PIL import Image


class ConsistI2V:
    def __init__(self, device="cuda"):

        pass

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
        print("Inference Config:", self.args.inference_config)
        print("Prompt:", self.args.prompt)
        print("Format:", self.args.format)

        yaml_config = """
            output_dir: "samples/inference"
            output_name: "i2v"

            pretrained_model_path: "TIGER-Lab/ConsistI2V"
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

        from omegaconf import OmegaConf

        self.config = OmegaConf.safe_load(yaml_config)

        from videogen_hub.pipelines.consisti2v.scripts.animate import main

        self.pipeline = main

        # raise NotImplementedError

    def infer_one_video(
        self,
        input_image: Image,
        prompt: str = None,
        size: list = [320, 512],
        seconds: int = 2,
        fps: int = 8,
        seed: int = 42,
    ):
        """
        Generates a single video based on the provided prompt and parameters.

        Args:
            prompt (str, optional): The text prompt to generate the video from. Defaults to None.
            size (list, optional): The size of the video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The frames per second of the video. Defaults to 8.
            seed (int, optional): The seed for random number generation. Defaults to 42.

        Returns:
            torch.Tensor: The generated video as a tensor.
        """

        self.args.path_to_first_frame = input_image
        self.args.seed = str(seed)
        self.config.sampling_kwargs.height = size[0]
        self.config.sampling_kwargs.width = size[1]
        self.config.sampling_kwargs.n_frames = seconds * fps

        return self.pipeline(self.args, self.config)
