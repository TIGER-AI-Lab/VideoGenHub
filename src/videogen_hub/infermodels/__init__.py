# ==========================================================
# Text-to-Video Generation
from videogen_hub.infermodels.lavie import LaVie
from videogen_hub.infermodels.videocrafter import VideoCrafter2
from videogen_hub.infermodels.modelscope import ModelScope
from videogen_hub.infermodels.streamingt2v import StreamingT2V
from videogen_hub.infermodels.show_one import ShowOne
from videogen_hub.infermodels.opensora import OpenSora
from videogen_hub.infermodels.opensora_plan import OpenSoraPlan
from videogen_hub.infermodels.t2v_turbo import T2VTurbo
from videogen_hub.infermodels.opensora_12 import OpenSora12

# from .cogvideo import CogVideo # Not supporting CogVideo ATM

# ==========================================================
# Image-to-Video Generation
from videogen_hub.infermodels.seine import SEINE
from videogen_hub.infermodels.consisti2v import ConsistI2V
from videogen_hub.infermodels.dynamicrafter import DynamiCrafter
from videogen_hub.infermodels.i2vgen_xl import I2VGenXL

# ==========================================================

import sys
from functools import partial


def get_model(model_name: str = None, init_with_default_params: bool = True):
    """
    Retrieves a model class or instance by its name.

    Args:
        model_name (str): Name of the model class. Triggers an error if the module name does not exist.
        init_with_default_params (bool, optional): If True, returns an initialized model instance; otherwise, returns
            the model class. Default is True. If set to True, be cautious of potential ``OutOfMemoryError`` with insufficient CUDA memory.

    Returns:
        model_class or model_instance: Depending on ``init_with_default_params``, either the model class or an instance of the model.

    Examples::
        initialized_model = infermodels.get_model(model_name='<Model>', init_with_default_params=True)

        uninitialized_model = infermodels.get_model(model_name='<Model>', init_with_default_params=False)
        initialized_model = uninitialized_model(device="cuda", <...>)
    """

    if not hasattr(sys.modules[__name__], model_name):
        raise ValueError(f"No model named {model_name} found in infermodels.")

    model_class = getattr(sys.modules[__name__], model_name)
    if init_with_default_params:
        model_instance = model_class()
        return model_instance
    return model_class


load_model = partial(get_model, init_with_default_params=True)
load = partial(get_model)
