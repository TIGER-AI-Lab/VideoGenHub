import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import List
from torchvision.models import vit_b_16 
import torchvision.transforms as transforms

ROUND_DIGIT=3
NUM_ASPECT=5

DINO_POINT_HIGH=0.97
DINO_POINT_MID=0.9
DINO_POINT_LOW=0.8   


class MetricDINO_sim():
    def __init__(self, device="cuda") -> None: 
        """
        Initialize a class MetricDINO_sim with the specified device for testing temporal consistency of a given video.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.device = device
        self.model = vit_b_16(pretrained=True)
        self.model.to(self.device).eval()  
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def evaluate(self, frame_list:List[Image.Image]):
        """
        Calculate the cosine similarity between the DINO features of adjacent frames of a given video to test temporal consistency, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            
        Returns:
            dino_frame_score: float, the computed DINO feature cosine similarity between each adjacent pair of frames and then averaged among all the pairs.
            quantized_ans: int, the quantized value of the above avg DINO-Sim scores based on pre-defined thresholds.
        """
        
        device = self.device
        frame_sim_list=[]
        for f_idx in range(len(frame_list)-1):
            frame_1=frame_list[f_idx]
            frame_2=frame_list[f_idx+1]
            frame_tensor_1 = self.preprocess(frame_1).unsqueeze(0).to(device)
            frame_tensor_2 = self.preprocess(frame_2).unsqueeze(0).to(device)
            with torch.no_grad():
                feat_1 = self.model(frame_tensor_1).flatten()
                feat_2 = self.model(frame_tensor_2).flatten()
            cos_sim=F.cosine_similarity(feat_1, feat_2, dim=0).item()
            frame_sim_list.append(cos_sim)   
    
        dino_frame_score = np.mean(frame_sim_list)
        quantized_ans=0
        if dino_frame_score >= DINO_POINT_HIGH:
            quantized_ans=4
        elif dino_frame_score < DINO_POINT_HIGH and dino_frame_score >= DINO_POINT_MID:
            quantized_ans=3
        elif dino_frame_score < DINO_POINT_MID and dino_frame_score >= DINO_POINT_LOW:
            quantized_ans=2
        else:
            quantized_ans=1
        return dino_frame_score, quantized_ans
