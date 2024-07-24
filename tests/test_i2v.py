import os, sys
import torch
from diffusers.utils import load_image

from videogen_hub import all_image2video_models

# Directly run `python -m pytest` or
# Directly run `python -m pytest -v -s --disable-warnings` for Debugging

# To test single function:
# pytest tests/test_i2v.py::test_function_name

dummy_prompt = "A tiger in a lab coat with a 1980s Miami vibe, turning a well oiled science content machine."
dummy_image = load_image("https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IG/DALLE3/sample_69.jpg")

sys.path.append("src")

all_i2v_models = all_image2video_models()


def test_models(model_name=None):
    for model in all_i2v_models:
        if model_name is not None and model.__name__ != model_name:
            continue
        print(model.__name__)
        assert model is not None
        out_video = model.infer_one_video(dummy_image, dummy_prompt)
        assert out_video is not None
        # check if out_video is a tensor or not
        assert isinstance(out_video, torch.Tensor)
        print(out_video.shape)


if __name__ == "__main__":
    # See if there's an argument to test a specific model
    if len(sys.argv) > 1:
        test_models(sys.argv[1])
    else:
        test_models()
    print("Everything passed")
    pass
