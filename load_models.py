import os, sys
import torch

def load_all_models():
  sys.path.insert(0, './src/')
  from src.videogen_hub.infermodels import LaVie
  from src.videogen_hub.infermodels import VideoCrafter2
  from src.videogen_hub.infermodels import SEINE
  from src.videogen_hub.infermodels import ModelScope
  from src.videogen_hub.infermodels import DynamiCrafter

  lavie = LaVie()
  print("Lavie Model is already loaded!")
  videocrafter2 = VideoCrafter2()
  print("VideoCrafter is already loaded!")
  modelscope = ModelScope()
  print("ModelScope is already loaded!")
  seine = SEINE()
  print("SEINE is already loaded!")
  dynamicrafter = DynamiCrafter()
  print("DynamiCrafter is already loaded!")

  return [lavie, videocrafter2, modelscope, seine, dynamicrafter]
