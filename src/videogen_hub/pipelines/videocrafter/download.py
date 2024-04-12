import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch
from filelock import FileLock
from tqdm.auto import tqdm

CACHE = "~/checkpoints"
model_name = "videocrafter2"
MODEL_URLS = {
    "videocrafter2": "https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt"
}

@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(CACHE, model_name)  # use diffuser path.


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.

    If cache_dir is specified, it will be used to models the files.
    Otherwise, default_cache_dir() is used.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(local_path):
        return local_path
    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


def fetch_checkpoint(
    checkpoint_name: str,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> str:
    if checkpoint_name not in MODEL_URLS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_URLS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_URLS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    return path