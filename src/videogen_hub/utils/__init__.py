import torch
from torchvision import transforms
from PIL import Image

def images_to_tensor(image_list):
    """
    Parse a list of PIL images and convert them to a PyTorch tensor in shape (T, C, H, W).
    """
    transform = transforms.ToTensor()
    
    # Convert each PIL image to tensor and store in a list
    tensor_list = [transform(img) for img in image_list]
    
    # Stack the list of tensors along a new dimension to create the final tensor
    tensor = torch.stack(tensor_list, dim=0)
    
    return tensor