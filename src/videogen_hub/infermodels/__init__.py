# ==========================================================
# Text-to-Video Generation
from .lavie import LaVie
from .videocrafter import VideoCrafter2
from .modelscope import ModelScope
from .streamingt2v import StreamingT2V
#from .cogvideo import CogVideo # Not supporting CogVideo ATM

# ==========================================================
# Image-to-Video Generation
from .seine import SEINE
from .consisti2v import ConsistI2V
from .dynamicrafter import DynamiCrafter

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
