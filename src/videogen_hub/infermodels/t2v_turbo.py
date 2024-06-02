from huggingface_hub import hf_hub_download, snapshot_download
import torch


class T2VTurbo():
    def __init__(self, base_model="vc2", device="cuda"):
        """
    1. Download the pretrained model and put it inside checkpoints/
    2. Create Pipeline
    Args:
        device: 'cuda' or 'cpu' the device to use the model
    """
        from videogen_hub.pipelines.t2v_turbo.inference_vc2 import T2VTurboVC2Pipeline1
        from videogen_hub.pipelines.t2v_turbo.inference_ms import T2VTurboMSPipeline1

        self.config = {
            "model": {
                "target": "lvdm.models.ddpm3d.LatentDiffusion",
                "params": {
                    "linear_start": 0.00085,
                    "linear_end": 0.012,
                    "num_timesteps_cond": 1,
                    "timesteps": 1000,
                    "first_stage_key": "video",
                    "cond_stage_key": "caption",
                    "cond_stage_trainable": False,
                    "conditioning_key": "crossattn",
                    "image_size": [320, 512],
                    "channels": 4,
                    "scale_by_std": False,
                    "scale_factor": 0.18215,
                    "use_ema": False,
                    "uncond_type": "empty_seq",
                    "use_scale": True,
                    "scale_b": 0.7,
                    "unet_config": {
                        "target": "lvdm.modules.networks.openaimodel3d.UNetModel",
                        "params": {
                            "in_channels": 4,
                            "out_channels": 4,
                            "model_channels": 320,
                            "attention_resolutions": [4, 2, 1],
                            "num_res_blocks": 2,
                            "channel_mult": [1, 2, 4, 4],
                            "num_head_channels": 64,
                            "transformer_depth": 1,
                            "context_dim": 1024,
                            "use_linear": True,
                            "use_checkpoint": True,
                            "temporal_conv": True,
                            "temporal_attention": True,
                            "temporal_selfatt_only": True,
                            "use_relative_position": False,
                            "use_causal_attention": False,
                            "temporal_length": 16,
                            "addition_attention": True,
                            "fps_cond": True
                        }
                    },
                    "first_stage_config": {
                        "target": "lvdm.models.autoencoder.AutoencoderKL",
                        "params": {
                            "embed_dim": 4,
                            "monitor": "val / rec_loss",
                            "ddconfig": {
                                "double_z": True,
                                "z_channels": 4,
                                "resolution": 512,
                                "in_channels": 3,
                                "out_ch": 3,
                                "ch": 128,
                                "ch_mult": [1, 2, 4, 4],
                                "num_res_blocks": 2,
                                "attn_resolutions": [],
                                "dropout": 0.0
                            },
                            "lossconfig": {
                                "target": "torch.nn.Identity"
                            }
                        }
                    },
                    "cond_stage_config": {
                        "target": "lvdm.modules.encoders.condition.FrozenOpenCLIPEmbedder",
                        "params": {
                            "freeze": True,
                            "layer": "penultimate"
                        }
                    }
                }
            }
        }

        if base_model == "vc2":
            base_model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                              filename="model.ckpt",
                                              local_dir="./checkpoints/videocrafter2")

            unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2",
                                             filename="unet_lora.pt",
                                             local_dir="./checkpoints/T2V-Turbo-VC2")
            # It uses the config provided above.
            self.pipeline = T2VTurboVC2Pipeline1(self.config, device, unet_lora_path, base_model_path)
        else:
            base_model_path = snapshot_download(repo_id="ali-vilab/text-to-video-ms-1.7b",
                                                local_dir="./checkpoints/modelscope_1.7b")

            unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-MS",
                                             filename="unet_lora.pt",
                                             local_dir="./checkpoints/T2V-Turbo-MS")
            # It uses the config provided by base_model.
            self.pipeline = T2VTurboMSPipeline1(device, unet_lora_path, base_model_path)

    def infer_one_video(
            self,
            prompt: str = None,
            size: list = [320, 512],
            seconds: int = 2,
            fps: int = 8,
            seed: int = 42,
    ):
        """
    Generates a single video based on the provided prompt and parameters.
    The output is of shape [frames, channels, height, width].
    Args:
        prompt (str, optional): The text prompt to generate the video from. Defaults to None.
        seconds (int, optional): The duration of the video in seconds. Defaults to 2.
        fps (int, optional): The frames per second of the video. Defaults to 8.
        seed (int, optional): The seed for random number generation. Defaults to 42.

    Returns:
        torch.Tensor: The generated video as a tensor.
    """
        output = self.pipeline.inference(prompt=prompt, height=size[0], width=size[1],
                                         seed=seed, num_frames=seconds * fps, fps=fps, randomize_seed=False)

        return output.squeeze().cpu()
