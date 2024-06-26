import inspect
import os

from videogen_hub._version import __version__
from videogen_hub.base.base_infer_model import BaseInferModel
from videogen_hub.base.base_t2v_infer_model import BaseT2vInferModel

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
if os.environ.get("VIDEO_MODEL_PATH"):
    MODEL_PATH = os.environ.get("VIDEO_MODEL_PATH")

# (cogVideo) Set the SAT_HOME env variable to MODEL_PATH if not set
if not os.environ.get("SAT_HOME"):
    os.environ["SAT_HOME"] = MODEL_PATH

from videogen_hub.infermodels import load, get_model, load_model


def all_text2video_models():
    # Find all models which inherit from BaseT2vInferModel in the infermodels directory
    from videogen_hub import infermodels
    all_infer_models = [cls for name, cls in inspect.getmembers(infermodels, inspect.isclass) if issubclass(cls, BaseT2vInferModel)]
    return all_infer_models

def all_image2video_models():
    # Find all models which inherit from BaseT2vInferModel in the infermodels directory
    from videogen_hub import infermodels
    all_infer_models = [cls for name, cls in inspect.getmembers(infermodels, inspect.isclass) if issubclass(cls, BaseT2vInferModel)]
    return all_infer_models

def all_models():
    # Find all models which inherit from BaseInferModel in the infermodels directory
    from videogen_hub import infermodels
    all_infer_models = [cls for name, cls in inspect.getmembers(infermodels, inspect.isclass) if issubclass(cls, BaseInferModel)]
    return all_infer_models