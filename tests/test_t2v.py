import os, sys
import torch

from videogen_hub import all_text2video_models
import sys

# Directly run `python -m pytest` or
# Directly run `python -m pytest -v -s --disable-warnings` for Debugging

# To test single function:
# pytest tests/test_t2v.py::test_function_name

dummy_prompts = [
    "a teddy bear walking on the street, 2k, high quality",
    "a panda taking a selfie, 2k, high quality",
    "a polar bear playing drum kit in NYC Times Square, 4k, high resolution",
    "jungle river at sunset, ultra quality",
    "a shark swimming in clear Carribean ocean, 2k, high quality",
    "a Corgi walking in the park at sunrise, oil painting style",
    "a cat playing with a ball of yarn, 2k, high quality",
    "a dog playing fetch in the park, 2k, high quality",
    "a cat sleeping on the couch, 2k, high quality",
    "a dog chasing its tail, 2k, high quality",
    "a cat chasing a mouse, 2k, high quality",
]

sys.path.append("src")

all_t2v_models = all_text2video_models()


def test_models(model_name=None):
    for i, model in enumerate(all_t2v_models):
        if model_name is not None and model.__name__ != model_name:
            continue
        print(f"Testing {i + 1}. {model.__name__}")
        out_video = model().infer_one_video(dummy_prompts[i])
        assert isinstance(out_video, torch.Tensor)
        print(out_video.shape)


if __name__ == "__main__":
    # See if there's an argument to test a specific model
    if len(sys.argv) > 1:
        test_models(sys.argv[1])
    else:
        test_models()
