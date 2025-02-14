import os
import time
from pathlib import Path
# from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


class HunyuanVodeoPipeline():
  def __init__(self, args):
    # args = parse_args()
    # print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    # save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    # if not os.path.exists(args.save_path):
    #     os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    self.args = hunyuan_video_sampler.args

  def inference(self, prompt, neg_prompt, output_size, video_length, seed):
    # output_size = [height, width]
    # Start sampling
    # TODO: batch inference check
    self.args.prompt = prompt
    self.args.video_size = output_size
    self.args.video_length = video_length
    self.args.neg_prompt = neg_prompt
    self.args.seed = seed
    args = self.args

    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    # if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
    #     for i, sample in enumerate(samples):
    #         sample = samples[i].unsqueeze(0)
    #         time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    #         save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
    #         save_videos_grid(sample, save_path, fps=24)
    #         logger.info(f'Sample save to: {save_path}')
    return samples
