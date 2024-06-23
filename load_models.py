import os
import sys
import traceback

import torch


def load_all_models():
    """
    Download models of Lavie, VideoCrafter2, SEINE, ModelScope, and DynamiCrafter,
    into the directory defined by MODEL_PATH,
    with cuda cache emptied.
    Returns: None
    """
    sys.path.insert(0, './src/')
    #from src.videogen_hub.infermodels import CogVideo
    from src.videogen_hub.infermodels import ConsistI2V
    from src.videogen_hub.infermodels import DynamiCrafter
    from src.videogen_hub.infermodels import I2VGenXL
    from src.videogen_hub.infermodels import LaVie
    from src.videogen_hub.infermodels import ModelScope
    from src.videogen_hub.infermodels import OpenSora
    from src.videogen_hub.infermodels import OpenSoraPlan
    from src.videogen_hub.infermodels import SEINE
    from src.videogen_hub.infermodels import ShowOne
    from src.videogen_hub.infermodels import StreamingT2V
    from src.videogen_hub.infermodels import T2VTurbo
    from src.videogen_hub.infermodels import VideoCrafter2

    from src.videogen_hub import MODEL_PATH

    try:
        ConsistI2V()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'ConsistI2V'))
    print("ConsistI2V has already been downloaded!")

    try:
        DynamiCrafter()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'dynamicrafter_256_v1'))
    print("DynamiCrafter has already been downloaded!")

    try:
        I2VGenXL()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'i2vgen-xl'))
    print("I2VGenXL has already been downloaded!")

    try:
        LaVie()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'lavie'))
    print("Lavie Model has already been downloaded!")

    try:
        ModelScope()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'modelscope'))
    print("ModelScope has already been downloaded!")

    try:
        SEINE()
    except Exception as e:
        print(f"Seine exception: {e}")
        traceback.print_exc()
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'SEINE'))
    print("SEINE has already been downloaded!")

    try:
        ShowOne()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'showlab'))
    print("ShowOne has already been downloaded!")

    try:
        StreamingT2V()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'streamingtv2'))
    print("StreamingTV has already been downloaded!")

    try:
        T2VTurbo()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'T2V-Turbo-VC2'))
    print("T2VTurbo has already been downloaded!")

    try:
        VideoCrafter2()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'videocrafter2'))
    print("VideoCrafter has already been downloaded!")

    # Do these last, as they're linux-only...
    try:
        OpenSora()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'STDiT2-XL_2'))
    print("OpenSora has already been downloaded!")

    try:
        OpenSoraPlan()
    except:
        pass
    torch.cuda.empty_cache()
    assert os.path.exists(os.path.join(MODEL_PATH, 'Open-Sora-Plan-v1.1.0'))
    print("OpenSoraPlan has already been downloaded!")



if __name__ == '__main__':
    load_all_models()
