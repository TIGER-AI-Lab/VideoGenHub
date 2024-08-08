import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoProcessor

NUM_ASPECT=5
ROUND_DIGIT=3
MAX_LENGTH = 76

MAX_NUM_FRAMES=8

X_CLIP_POINT_LOW=0.15
X_CLIP_POINT_MID=0.225
X_CLIP_POINT_HIGH=0.30


def _read_video_frames(frames, max_frames):
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)

    selected_frames = [np.array(frames[i]) for i in indices]
    return np.stack(selected_frames)


class MetricXCLIPScore():
    def __init__(self, device="cuda") -> None: 
        """
        Initialize a MetricXCLIPScore object with the specified device.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        
        self.model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
        self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

    def evaluate(self, frame_list:List[Image.Image], text:str,):
        """
        Calculate the cosine similarity of between X-CLIP features of text prompt and the given video to test text-to-video alignment, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            text:str, text prompt for generating the video.
            
        Returns:
            xclip_score_avg: float, the computed X-CLIP-Score between video and its text prompt.
            quantized_ans: int, the quantized value of the above avg SSIM scores based on pre-defined thresholds.
        """
        
        input_text = self.tokenizer([text], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
        text_feature = self.model.get_text_features(**input_text).flatten()

        video=_read_video_frames(frame_list,MAX_NUM_FRAMES)
        
        input_video = self.processor(videos=list(video), return_tensors="pt")
        video_feature = self.model.get_video_features(**input_video).flatten()
        cos_sim=F.cosine_similarity(text_feature, video_feature, dim=0).item()
        quantized_ans=0
        if cos_sim < X_CLIP_POINT_LOW:
            quantized_ans=1
        elif cos_sim >= X_CLIP_POINT_LOW and cos_sim < X_CLIP_POINT_MID:
            quantized_ans=2
        elif cos_sim >= X_CLIP_POINT_MID and cos_sim < X_CLIP_POINT_HIGH:
            quantized_ans=3
        else:
            quantized_ans=4
        return cos_sim, quantized_ans


