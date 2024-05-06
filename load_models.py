import os, sys
import torch


def load_all_models():
    """
    Download models of Lavie, VideoCrafter2, SEINE, ModelScope, and DynamiCrafter,
    into the directory of './checkpoints/',
    with cuda cache emptied.
    Returns: None
    """
    sys.path.insert(0, './src/')
    from src.videogen_hub.infermodels import LaVie
    from src.videogen_hub.infermodels import VideoCrafter2
    from src.videogen_hub.infermodels import SEINE
    from src.videogen_hub.infermodels import ModelScope
    from src.videogen_hub.infermodels import DynamiCrafter

    LaVie()
    torch.cuda.empty_cache()
    assert os.path.exists('./checkpoints/lavie')
    print("Lavie Model has already been downloaded!")

    VideoCrafter2()
    torch.cuda.empty_cache()
    assert os.path.exists('./checkpoints/videocrafter')
    print("VideoCrafter has already been downloaded!")

    ModelScope()
    torch.cuda.empty_cache()
    assert os.path.exists('./checkpoints/modelscope')
    print("ModelScope has already been downloaded!")

    SEINE()
    torch.cuda.empty_cache()
    assert os.path.exists('./checkpoints/SEINE')
    print("SEINE has already been downloaded!")

    DynamiCrafter()
    torch.cuda.empty_cache()
    assert os.path.exists('./checkpoints/dynamicrafter_256_v1')
    print("DynamiCrafter has already been downloaded!")


if __name__ == '__main__':
    load_all_models()
