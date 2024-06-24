import numpy as np
import torch


def tensor_to_video(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 0, 2, 3).float().numpy() # c t h w -> 
    x = (255 * x).astype(np.uint8)
    return x