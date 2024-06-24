import os

from videogen_hub._version import __version__
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
if os.environ.get("VIDEO_MODEL_PATH"):
    MODEL_PATH = os.environ.get("VIDEO_MODEL_PATH")

# (cogVideo) Set the SAT_HOME env variable to MODEL_PATH if not set
if not os.environ.get("SAT_HOME"):
    os.environ["SAT_HOME"] = MODEL_PATH

from videogen_hub.infermodels import load, get_model, load_model
