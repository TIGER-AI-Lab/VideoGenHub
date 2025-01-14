import os

from huggingface_hub import snapshot_download, hf_hub_download

from videogen_hub import MODEL_PATH

class HunyuanVideo:
  def __init__(self, device="gpu"):
    """
    1. Download the pretrained model and put it inside MODEL_PATH/hunyuan_video
    2. Create Pipeline
    Note: it seems that the model needed from model_dir cannot support cpu
    Args:
        device: 'gpu' or 'cpu' the device to use the model
    """
    from videogen_hub.pipelines.hunyuan_video.sample_video import HunyuanVodeoPipeline
    from videogen_hub.pipelines.hunyuan_video.hyvideo.config import parse_args

    hunyuan_model_path = snapshot_download("tencent/HunyuanVideo",
                    local_dir=os.path.join(MODEL_PATH, 'hunyuan_video'))
    
    llava_path = snapshot_download("xtuner/llava-llama-3-8b-v1_1-transformers", 
                    local_dir=os.path.join(hunyuan_model_path, 'llava-llama-3-8b-v1_1-transformers'))
    encoder_path = os.path.join(hunyuan_model_path, 'text_encoder')

    os.system('../pipelines/hunyuan_vodeo/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py', 
    '--input_dir', llava_path, '--output_dir', encoder_path)

    model_path = snapshot_download("openai/clip-vit-large-patch14", 
                    local_dir=os.path.join(hunyuan_model_path, 'text_encoder_2'))

    # Create the pipeline.

    # Use the default argument
    args = parse_args()
    # Set model base path
    args.model_base = hunyuan_model_path
    args.dit_weight = os.path.join(hunyuan_model_path, 'hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt')

    self.pipeline = HunyuanVodeoPipeline(args)


  def infer_one_video(self,
                        prompt: str = None,
                        size: list = [720, 1280],
                        seconds: int = 1,
                        fps: int = 129,
                        seed: int = 42):
    """
    Generates a single video based on a textual prompt.
    Args:
        prompt (str, optional): The text prompt that guides the video generation.
        output_size (list, optional): Specifies the resolution of the output video as [height, width].
        seconds: time of the video
        fps: fps of the video
        seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.

    Returns:
        The generated video as a tensor with shape (num_frames, channels, height, width).
    """
    
    num_frames = fps * seconds
    video = self.pipeline.inference(prompt=prompt,
                      neg_prompt="",
                      output_size=size,
                      video_length=num_frames,
                      seed=seed)
    return video

    


