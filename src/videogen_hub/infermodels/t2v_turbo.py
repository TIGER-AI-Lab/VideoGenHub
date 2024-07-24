import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from videogen_hub import MODEL_PATH
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel
from videogen_hub.pipelines.t2v_turbo.inference_vc2 import T2VTurboVC2Pipeline1
from videogen_hub.pipelines.t2v_turbo.inference_ms import T2VTurboMSPipeline1


class T2VTurbo(BaseT2vInferModel):
    def __init__(self, base_model="vc2", merged=True, device="cuda"):
        self.device = device
        self.resolution = [320, 512]
        self.model_path = os.path.join(MODEL_PATH, "T2V-Turbo")
        self.base_model = base_model
        self.merged = merged
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

    def load_pipeline(self):
        if self.pipeline is None:
            unet_lora_path, base_model_path = self.download_models(self.base_model, self.merged)
            if self.base_model == "vc2" and self.merged:
                self.pipeline = T2VTurboVC2Pipeline1(self.config, self.merged, self.device, unet_lora_path, base_model_path)

            elif self.base_model == "vc2":
                self.pipeline = T2VTurboVC2Pipeline1(self.config, self.merged, self.device, unet_lora_path, base_model_path)
            else:
                self.pipeline = T2VTurboMSPipeline1(self.device, unet_lora_path, base_model_path)
        self.to(self.device)
        return self.pipeline

    def download_models(self, base_model=None, merged=None) -> str:
        if base_model is not None and merged is not None:
            if base_model == "vc2" and merged:
                merged_model_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2-Merged",
                                                    filename="t2v_turbo_vc2.pt",
                                                    local_dir=os.path.join(MODEL_PATH, "T2V-Turbo-VC2"))
                return merged_model_path, None
            elif base_model == "vc2":
                base_model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                                  filename="model.ckpt",
                                                  local_dir=os.path.join(MODEL_PATH, "videocrafter2"))

                unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2",
                                                 filename="unet_lora.pt",
                                                 local_dir=os.path.join(MODEL_PATH, "T2V-Turbo-VC2"))
                return base_model_path, unet_lora_path

            else:
                base_model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                                  filename="model.ckpt",
                                                  local_dir=os.path.join(MODEL_PATH, "videocrafter2"))

                unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2",
                                                 filename="unet_lora.pt",
                                                 local_dir=os.path.join(MODEL_PATH, "T2V-Turbo-VC2"))
                return base_model_path, unet_lora_path

        else:
            merged_model_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2-Merged",
                                                filename="t2v_turbo_vc2.pt",
                                                local_dir=os.path.join(MODEL_PATH, "T2V-Turbo-VC2"))

            base_model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                              filename="model.ckpt",
                                              local_dir=os.path.join(MODEL_PATH, "videocrafter2"))

            unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2",
                                             filename="unet_lora.pt",
                                             local_dir=os.path.join(MODEL_PATH, "T2V-Turbo-VC2"))
            base_model_path_2 = snapshot_download(repo_id="ali-vilab/text-to-video-ms-1.7b",
                                                  local_dir=os.path.join(MODEL_PATH, "modelscope_1.7b"))

            unet_lora_path_2 = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-MS",
                                               filename="unet_lora.pt",
                                               local_dir=os.path.join(MODEL_PATH, "T2V-Turbo-MS"))
            model_paths = [merged_model_path, base_model_path, unet_lora_path, base_model_path_2, unet_lora_path_2]
            return model_paths

    def infer_one_video(
            self,
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
    The output is of shape [frames, channels, height, width].
    Args:
        prompt (str, optional): The text prompt to generate the video from. Defaults to None.
        negative_prompt (str, optional): The negative text prompt to generate the video from. Defaults to None.
        size (list, optional): The size of the video as [height, width]. Defaults to [320, 512].
        seconds (int, optional): The duration of the video in seconds. Defaults to 2.
        fps (int, optional): The frames per second of the video. Defaults to 8.
        seed (int, optional): The seed for random number generation. Defaults to 42.
        unload (bool, optional): Whether to unload the model from the device after generating the video. Defaults to True

    Returns:
        torch.Tensor: The generated video as a tensor.
    """
        if size is None:
            size = self.resolution

        self.load_pipeline()

        output = self.pipeline.inference(prompt=prompt, height=size[0], width=size[1],
                                         seed=seed, num_frames=seconds * fps, fps=fps, randomize_seed=False)
        # [channels, frames, height, width] -> [frames, channels, height, width]
        output = output.squeeze().permute(1, 0, 2, 3)
        if unload:
            self.to("cpu")
        return output.cpu()
