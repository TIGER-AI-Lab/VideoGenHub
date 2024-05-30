from huggingface_hub import snapshot_download, hf_hub_download
import torch


class OpenSoraPlan():
    def __init__(self, device="cuda"):
        """
    1. Download the pretrained model and put it inside checkpoints/
    2. Create Pipeline
    Note: it seems that the model needed from model_dir cannot support cpu
    Args:
        device: 'cuda' or 'cpu' the device to use the model
    """
        from videogen_hub.pipelines.opensora_plan.opensora.sample_t2v import OpenSoraPlanPipeline

        arg_list = ['--model_path', 'LanguageBind/Open-Sora-Plan-v1.1.0',
                    '--version', '65x512x512',
                    '--num_frames', '65',
                    '--height', '512',
                    '--width', '512',
                    '--cache_dir', "./checkpoints",
                    '--text_encoder_name', 'DeepFloyd/t5-v1_1-xxl',
                    '--text_prompt', 'prompt_list_0.txt',
                    '--ae', 'CausalVAEModel_4x8x8',
                    '--ae_path', "/remote-home1/yeyang/CausalVAEModel_4x8x8",
                    '--save_img_path', "./sample_video_65x512x512",
                    '--fps', '24',
                    '--guidance_scale', '7.5',
                    '--num_sampling_steps', '150',
                    '--enable_tiling']
        self.pipeline = OpenSoraPlanPipeline(arg_list, device)

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
    Note that there are only 3 available shapes: (1 or 65 or 221)xHxW
    The output is of shape [frames, channels, height, width].
    Args:
        prompt (str, optional): The text prompt to generate the video from. Defaults to None.
        seconds (int, optional): The duration of the video in seconds. Defaults to 2.
        fps (int, optional): The frames per second of the video. Defaults to 8.
        seed (int, optional): The seed for random number generation. Defaults to 42.

    Returns:
        torch.Tensor: The generated video as a tensor.
    """

        torch.manual_seed(seed)

        self.pipeline.args.text_prompt = prompt
        self.pipeline.args.num_frames = fps * seconds
        self.pipeline.args.fps = fps
        self.pipeline.args.height = size[0]
        self.pipeline.args.width = size[1]

        samples = self.pipeline.inference(save_output=False)
        # samples is torch.Size([B, T, H, W, C])

        output = samples.squeeze(0).permute(0, 3, 1, 2).cpu().float()
        return output
