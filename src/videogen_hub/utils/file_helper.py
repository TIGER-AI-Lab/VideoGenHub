import os
from typing import Union, List, Optional
from urllib.parse import urlparse
import requests

def get_file_path(filename: Union[str, os.PathLike], search_from: Union[str, os.PathLike] = "."):
    """
    Search for a file across a directory and return its absolute path.

    Args:
        filename (Union[str, os.PathLike]): The name of the file to search for.
        search_from (Union[str, os.PathLike], optional): The directory from which to start the search. Defaults to ".".

    Returns:
        str: Absolute path to the found file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    for root, dirs, files in os.walk(search_from):
        for name in files:
            if name == filename:
                return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError(filename, "not found.")
