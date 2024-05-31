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

        if base_model == "vc2":
            base_model_path = hf_hub_download(repo_id="VideoCrafter/VideoCrafter2",
                                              filename="model.ckpt",
                                              local_dir="./checkpoints/videocrafter2")

            unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-VC2",
                                             filename="unet_lora.pt",
                                             local_dir="./checkpoints/T2V-Turbo-VC2")
            self.pipeline = T2VTurboVC2Pipeline1(device, unet_lora_path, base_model_path)
        else:
            base_model_path = snapshot_download(repo_id="ali-vilab/text-to-video-ms-1.7b",
                                                local_dir="./checkpoints/modelscope_1.7b")

            unet_lora_path = hf_hub_download(repo_id="jiachenli-ucsb/T2V-Turbo-MS",
                                             filename="unet_lora.pt",
                                             local_dir="./checkpoints/T2V-Turbo-MS")
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
