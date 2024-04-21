from typing import Union
import torch
from huggingface_hub import hf_hub_download


class DynamiCrafter():
    def __init__(self, version: str = '256'):
        from videogen_hub.pipelines.dynamicrafter.inference import DynamiCrafterPipeline

        if version == '256':
            (self.height, self.width) = 256, 256
            self.fs = 3
            self.model_path = hf_hub_download(repo_id="Doubiiu/DynamiCrafter", filename='model.ckpt',
                                              local_dir="./checkpoints/dynamicrafter_256_v1")

        elif version == '512':
            (self.height, self.width) = 320, 512
            self.fs = 24
            self.model_path = hf_hub_download(repo_id="Doubiiu/DynamiCrafter_512", filename='model.ckpt',
                                              local_dir="./checkpoints/dynamicrafter_512_v1")

        elif version == '1024':
            (self.height, self.width) = 576, 1024
            self.fs = 10
            self.model_path = hf_hub_download(repo_id="Doubiiu/DynamiCrafter_1024", filename='model.ckpt',
                                              local_dir="./checkpoints/dynamicrafter_1024_v1")
        else:
            raise ValueError("Invalid input. Please enter 256, 512, or 1024.")

        self.arg_list = \
            ['--ckpt_path', self.model_path,
             '--config', f'src/videogen_hub/pipelines/dynamicrafter/configs/inference_{version}_v1.0.yaml',
             '--n_samples', '1',
             '--bs', '1',
             '--height', str(self.height),
             '--width', str(self.width),
             '--text_input',
             '--unconditional_guidance_scale', '7.5',
             '--ddim_steps', '50',
             '--ddim_eta', '1.0',
             '--video_length', '16',
             '--frame_stride', str(self.fs)]

        self.pipeline = DynamiCrafterPipeline(self.arg_list)

    def infer_one_video(self, input_image: Union[str, torch.Tensor],
                        prompt: str = None,
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        self.pipeline.args.seed = seed
        self.pipeline.args.text_input = prompt
        self.pipeline.args.video_length = fps * seconds
        video = self.pipeline.run_inference(input_image)

        return video
