import os
import sys

sys.path.append("src")

from videogen_hub import all_models


def load_all_models():
    """
    Download *every model* into the directory defined by MODEL_PATH.
    Returns: None
    """
    all_model_classes = all_models()
    for model_class in all_model_classes:
        print(f"Checking models for {model_class.__name__}...")
        model_paths = model_class().download_models()
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        for model_path in model_paths:
            assert os.path.exists(model_path), f"Model path {model_path} does not exist."
        print(f"All models for {model_class.__name__} exist.")


if __name__ == '__main__':
    load_all_models()
