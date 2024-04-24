import os, sys
import torch
from diffusers.utils import load_image
# Directly run `python -m pytest` or
# Directly run `python -m pytest -v -s --disable-warnings` for Debugging

# To test single function:
# pytest tests/test_i2v.py::test_function_name

dummy_prompt = "A tiger in a lab coat with a 1980s Miami vibe, turning a well oiled science content machine."
dummy_image = load_image("https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IG/DALLE3/sample_69.jpg")

import sys
sys.path.append("src")

def test_SEINE():
    from videogen_hub.infermodels import SEINE

    model = SEINE()
    assert model is not None
    out_video = model.infer_one_video(dummy_image, dummy_prompt)
    assert out_video is not None
    # check if out_video is a tensor or not
    assert isinstance(out_video, torch.Tensor)
    print(out_video.shape)

def test_ConsistI2V():
    from videogen_hub.infermodels import ConsistI2V

    model = ConsistI2V()
    assert model is not None
    out_video = model.infer_one_video(dummy_image, dummy_prompt)
    assert out_video is not None
    # check if out_video is a tensor or not
    assert isinstance(out_video, torch.Tensor)
    print(out_video.shape)

def test_DynamiCrafter():
    from videogen_hub.infermodels import DynamiCrafter

    model = DynamiCrafter()
    assert model is not None
    out_video = model.infer_one_video(dummy_image, dummy_prompt)
    assert out_video is not None
    # check if out_video is a tensor or not
    assert isinstance(out_video, torch.Tensor)
    print(out_video.shape)
