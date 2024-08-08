from pypiqe import piqe
from PIL import Image
import numpy as np
from typing import List

ROUND_DIGIT=3
NUM_ASPECT=5

PIQE_POINT_LOW=15
PIQE_POINT_MID=30
PIQE_POINT_HIGH=50

class MetricPIQE():
    def __init__(self) -> None:
        """
        Initialize a class MetricPIQE for testing visual quality of a given video.
        
        """
        None

    def evaluate(self,frame_list:List[Image.Image]):
        """
        Calculate PIQE for visual quality for each frame of the given video and take the average value, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            
        Returns:
            piqe_avg: float, the computed average PIQE among the frames.
            quantized_ans: int, the quantized value of the above avg score based on pre-defined thresholds.
        """
        piqe_list=[]
        for frame in frame_list:
            frame=np.array(frame)
            piqe_score, _,_,_ = piqe(frame)
            piqe_list.append(piqe_score)
        piqe_avg=np.mean(piqe_list)
        quantized_ans=0
        if piqe_avg < PIQE_POINT_LOW:
            quantized_ans=4
        elif piqe_avg < PIQE_POINT_MID:
            quantized_ans=3
        elif piqe_avg < PIQE_POINT_HIGH:
            quantized_ans=2
        else:
            quantized_ans=1
        return piqe_avg, quantized_ans
    