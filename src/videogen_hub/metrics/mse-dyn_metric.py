import numpy as np
import cv2
from PIL import Image
from typing import List
from skimage.metrics import structural_similarity as ssim
from skimage import io, color

ROUND_DIGIT=3
DYN_SAMPLE_STEP=4
NUM_ASPECT=5

MSE_POINT_HIGH=3000
MSE_POINT_MID=1000
MSE_POINT_LOW=100


class MetricMSE_dyn():
    def __init__(self) -> None: 
        """
        Initialize a class MetricMSE_dyn for testing dynamic degree of a given video.
        
        """
        None

    def evaluate(self, frame_list:List[Image.Image]):
        """
        Calculate the MSE (Mean Squared Error) between frames sampled at regular intervals of a given video to test dynamic_degree, 
        then quantize the orginal output based on some predefined thresholds.
        
        Args:
            frame_list:List[Image.Image], frames of the video used in calculation.
            
        Returns:
            mse_avg: float, the computed MSE between frames sampled at regular intervals and then averaged among all the pairs.
            quantized_ans: int, the quantized value of the above avg MSE scores based on pre-defined thresholds.
        """
        
        mse_list=[]
        sampled_list = frame_list[::DYN_SAMPLE_STEP]
        for f_idx in range(len(sampled_list)-1):        
            imageA = cv2.cvtColor(np.array(sampled_list[f_idx]), cv2.COLOR_RGB2BGR)
            imageB = cv2.cvtColor(np.array(sampled_list[f_idx+1]), cv2.COLOR_RGB2BGR)
            
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])
            mse_value = err
            mse_list.append(mse_value)
        mse_avg=np.mean(mse_list)
        quantized_ans=0
        if mse_avg >= MSE_POINT_HIGH:
            quantized_ans=4
        elif mse_avg < MSE_POINT_HIGH and mse_avg >= MSE_POINT_MID:
            quantized_ans=3
        elif mse_avg < MSE_POINT_MID and mse_avg >= MSE_POINT_LOW:
            quantized_ans=2
        else:
            quantized_ans=1
            
        return mse_avg, quantized_ans
