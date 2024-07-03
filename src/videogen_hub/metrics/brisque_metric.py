from brisque import BRISQUE
from PIL import Image
import numpy as np
from typing import List

ROUND_DIGIT=3
NUM_ASPECT=5

BRISQUE_POINT_LOW=10
BRISQUE_POINT_MID=30
BRISQUE_POINT_HIGH=50

class MetricBRISQUE():
    def __init__(self) -> None:
        """
        Initialize a class MetricBRISQUE for testing visual quality of a given video.
        
        """
        None

    def evaluate(self,frame_list:List[Image.Image]):
        """
        Calculate BRISQUE for visual quality for each frame of the given video and take the average value, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation
            
        Returns:
            piqe_avg: float, the computed average BRISQUE among the frames
            quantized_ans: int, the quantized value of the above avg score based on pre-defined thresholds.
        """
        brisque_list=[]
        for frame in frame_list:
            brisque_score=BRISQUE().score(frame)
            brisque_list.append(brisque_score)
        brisque_avg=np.mean(brisque_list)
        quantized_ans=0
        if brisque_avg < BRISQUE_POINT_LOW:
            quantized_ans=4
        elif brisque_avg < BRISQUE_POINT_MID:
            quantized_ans=3
        elif brisque_avg < BRISQUE_POINT_HIGH:
            quantized_ans=2
        else:
            quantized_ans=1
        return brisque_avg, quantized_ans
