import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import List
from transformers import CLIPProcessor, CLIPModel
        
ROUND_DIGIT=3
NUM_ASPECT=5

CLIP_POINT_HIGH=0.97
CLIP_POINT_MID=0.9
CLIP_POINT_LOW=0.8   


class MetricCLIP_sim():
    def __init__(self, device = "cuda") -> None: 
        """
        Initialize a class MetricCLIP_sim with the specified device for testing temporal consistency of a given video.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def evaluate(self,frame_list:List[Image.Image]):
        """
        Calculate the cosine similarity between the CLIP features of adjacent frames of a given video to test temporal consistency, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            
        Returns:
            clip_frame_score: float, the computed CLIP feature cosine similarity between each adjacent pair of frames and then averaged among all the pairs.
            quantized_ans: int, the quantized value of the above avg CLIP-Sim scores based on pre-defined thresholds.
        """
        
        device=self.model.device
        frame_sim_list=[]
        for f_idx in range(len(frame_list)-1):
            frame_1 = frame_list[f_idx]
            frame_2 = frame_list[f_idx+1]
            input_1 = self.tokenizer(images=frame_1, return_tensors="pt", padding=True).to(device)
            input_2 = self.tokenizer(images=frame_2, return_tensors="pt", padding=True).to(device)
            output_1 = self.model.get_image_features(**input_1).flatten()
            output_2 = self.model.get_image_features(**input_2).flatten()
            cos_sim = F.cosine_similarity(output_1, output_2, dim=0).item()
            frame_sim_list.append(cos_sim)
            
        clip_frame_score = np.mean(frame_sim_list)
        quantized_ans=0
        if clip_frame_score >= CLIP_POINT_HIGH:
            quantized_ans=4
        elif clip_frame_score < CLIP_POINT_HIGH and clip_frame_score >= CLIP_POINT_MID:
            quantized_ans=3
        elif clip_frame_score < CLIP_POINT_MID and clip_frame_score >= CLIP_POINT_LOW:
            quantized_ans=2
        else:
            quantized_ans=1
        return clip_frame_score, quantized_ans
