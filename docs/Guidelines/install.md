# Installation

## Installing the project

To install from pypi:
```shall
pip install videogen-hub
```

To install from github:
```shall
git clone https://github.com/TIGER-AI-Lab/VideoGenHub.git
cd VideoGenHub
cd env_cfg
pip install -r requirements.txt
cd ..
pip install -e .
```
The requirement of opensora is in `env_cfg/opensora.txt`

For some models like show one, you need to login through `huggingface-cli`.

## Verify the installation
```python
import videogen_hub

print(videogen_hub.__version__) # should print a string
```

## **Downloading weights** into `checkpoints` folder
```shell
./download_models.sh
```