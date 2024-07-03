import numpy as np
from PIL import Image
from typing import List
from skimage.metrics import structural_similarity as ssim
from skimage import io, color

ROUND_DIGIT=3
DYN_SAMPLE_STEP=4
NUM_ASPECT=5

SSIM_POINT_HIGH=0.9
SSIM_POINT_MID=0.7
SSIM_POINT_LOW=0.5



class MetricSSIM_dyn():
    def __init__(self) -> None: 
        """
        Initialize a class MetricSSIM_dyn for testing dynamic degree of a given video.
        
        """
        None

    def evaluate(self, frame_list:List[Image.Image]):
        """
        Calculate the MSE between frames sampled at regular intervals of a given video to test dynamic_degree, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            
        Returns:
            ssim_avg: float, the computed SSIM between frames sampled at regular intervals and then averaged among all the pairs.
            quantized_ans: int, the quantized value of the above avg SSIM scores based on pre-defined thresholds.
        """
        
        ssim_list=[]
        sampled_list = frame_list[::DYN_SAMPLE_STEP]
        for f_idx in range(len(sampled_list)-1):
            frame_1=sampled_list[f_idx]
            frame_1_gray=color.rgb2gray(frame_1)
            frame_2=sampled_list[f_idx+1]
            frame_2_gray=color.rgb2gray(frame_2)

            ssim_value, _ = ssim(frame_1_gray, frame_2_gray, full=True,\
                                    data_range=frame_2_gray.max() - frame_2_gray.min())
            ssim_list.append(ssim_value)
        ssim_avg=np.mean(ssim_list)
        
        quantized_ans=0
        if ssim_avg >= SSIM_POINT_HIGH:
            quantized_ans=1
        elif ssim_avg <= SSIM_POINT_HIGH and ssim_avg > SSIM_POINT_MID:
            quantized_ans=2
        elif ssim_avg <= SSIM_POINT_MID and ssim_avg > SSIM_POINT_LOW:
            quantized_ans=3
        else:
            quantized_ans=4
        return ssim_avg, quantized_ans
