# Quickstart

## Running single model
```python
import videogen_hub

model = videogen_hub.load('VideoCrafter2')
video = model.infer_one_video(prompt="A child excitedly swings on a rusty swing set, laughter filling the air.")

# Here video is a torch tensor of shape torch.Size([16, 3, 320, 512])
```
