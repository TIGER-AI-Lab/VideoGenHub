import functools
import logging
import operator
import os
from typing import Tuple

import torch
import torch.distributed as dist
from torchvision.datasets.utils import download_url

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": hf_endpoint
    + "/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": hf_endpoint
    + "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
    "OpenSora-v1-16x256x256.pth": hf_endpoint
    + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth",
    "OpenSora-v1-HQ-16x256x256.pth": hf_endpoint
    + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-HQ-16x512x512.pth": hf_endpoint
    + "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth",
}


def reparameter(ckpt, name=None, model=None):
    if name in ["DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"]:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    if name in ["Latte-XL-2-256x256-ucf101.pt"]:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    if name in [
        "PixArt-XL-2-256x256.pth",
        "PixArt-XL-2-SAM-256x256.pth",
        "PixArt-XL-2-512x512.pth",
    ]:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]

    # no need pos_embed
    if "pos_embed_temporal" in ckpt:
        del ckpt["pos_embed_temporal"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    # different text length
    if "y_embedder.y_embedding" in ckpt:
        if (
            ckpt["y_embedder.y_embedding"].shape[0]
            < model.y_embedder.y_embedding.shape[0]
        ):
            print(
                f"Extend y_embedding from {ckpt['y_embedder.y_embedding'].shape[0]} to {model.y_embedder.y_embedding.shape[0]}"
            )
            additional_length = (
                model.y_embedder.y_embedding.shape[0]
                - ckpt["y_embedder.y_embedding"].shape[0]
            )
            new_y_embedding = torch.zeros(
                additional_length, model.y_embedder.y_embedding.shape[1]
            )
            new_y_embedding[:] = ckpt["y_embedder.y_embedding"][-1]
            ckpt["y_embedder.y_embedding"] = torch.cat(
                [ckpt["y_embedder.y_embedding"], new_y_embedding], dim=0
            )
        elif (
            ckpt["y_embedder.y_embedding"].shape[0]
            > model.y_embedder.y_embedding.shape[0]
        ):
            print(
                f"Shrink y_embedding from {ckpt['y_embedder.y_embedding'].shape[0]} to {model.y_embedder.y_embedding.shape[0]}"
            )
            ckpt["y_embedder.y_embedding"] = ckpt["y_embedder.y_embedding"][
                : model.y_embedder.y_embedding.shape[0]
            ]

    return ckpt


def find_model(model_name, model=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model_ckpt = download_model(model_name)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(
            model_name
        ), f"Could not find DiT checkpoint at {model_name}"
        model_ckpt = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    return model_ckpt


def download_model(model_name=None, local_path=None, url=None):
    """
    Downloads a pre-trained DiT model from the web.
    """
    if model_name is not None:
        assert model_name in pretrained_models
        local_path = f"pretrained_models/{model_name}"
        web_path = pretrained_models[model_name]
    else:
        assert local_path is not None
        assert url is not None
        web_path = url
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        dir_name = os.path.dirname(local_path)
        file_name = os.path.basename(local_path)
        download_url(web_path, dir_name, file_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def load_from_sharded_state_dict(model, ckpt_path):
    import os
    from contextlib import suppress

    # Attempt to import colossalAI first
    colossal_imported = False
    with suppress(ImportError):
        from colossalai.checkpoint import GeneralCheckpointIO
        colossal_imported = True

    if not colossal_imported:
        # Fall back to torch if colossalAI import fails
        import torch
        from torch import load as torch_load

    def load_model_with_fallback(model, ckpt_path):
        if colossal_imported:
            ckpt_io = GeneralCheckpointIO()
            ckpt_io.load_model(model, ckpt_path)
        else:
            model.load_state_dict(torch_load(os.path.join(ckpt_path, 'model')))

    print(os.getcwd())
    print(f"path={os.path.join(ckpt_path, 'model')}")
    relative_path = os.path.join(ckpt_path, "model")
    global_path = os.path.join(os.getcwd(), relative_path)

    # Load the model using the appropriate method
    load_model_with_fallback(model, ckpt_path)


def model_sharding(model: torch.nn.Module):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(
                param.data.view(-1), [0, padding_size]
            )
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def model_gathering(model: torch.nn.Module, model_shape_dict: dict):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(
                model_shape_dict[name]
            )
    dist.barrier()


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_checkpoint(model, ckpt_path, save_as_pt=False):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path, model=model)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, "model_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
