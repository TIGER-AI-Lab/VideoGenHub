import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import List
from transformers import CLIPProcessor, CLIPModel

NUM_ASPECT=5
ROUND_DIGIT=3
MAX_LENGTH = 76

MAX_NUM_FRAMES=8

CLIP_POINT_LOW=0.27
CLIP_POINT_MID=0.31
CLIP_POINT_HIGH=0.35


class MetricCLIPScore():
    def __init__(self, device="cuda") -> None: 
        """
        Initialize a MetricCLIPScore object with the specified device.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def evaluate(self, frame_list:List[Image.Image], text:str,):
        """
        Calculate the cosine similarity of between CLIP features of text prompt and each frame of a given video to test text-to-video alignment, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            text:str, text prompt for generating the video.
            
        Returns:
            clip_score_avg: float, the computed average CLIP-Score between each frame and the text prompt.
            quantized_ans: int, the quantized value of the above avg SSIM scores based on pre-defined thresholds.
        """
        
        device=self.model.device
        input_t = self.tokenizer(text=text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", padding=True).to(device)
        cos_sim_list=[]
        for image in frame_list:
            input_f = self.tokenizer(images=image, return_tensors="pt", padding=True).to(device)
            output_t = self.model.get_text_features(**input_t).flatten()
            output_f = self.model.get_image_features(**input_f).flatten()
            cos_sim = F.cosine_similarity(output_t, output_f, dim=0).item()
            cos_sim_list.append(cos_sim)
        clip_score_avg=np.mean(cos_sim_list)
        quantized_ans=0
        if clip_score_avg < CLIP_POINT_LOW:
            quantized_ans=1
        elif clip_score_avg >= CLIP_POINT_LOW and clip_score_avg < CLIP_POINT_MID:
            quantized_ans=2
        elif clip_score_avg >= CLIP_POINT_MID and clip_score_avg < CLIP_POINT_HIGH:
            quantized_ans=3
        else:
            quantized_ans=4
        return clip_score_avg, quantized_ans

