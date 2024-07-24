from typing import Dict

import decord
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

decord.bridge.set_bridge("torch")


class Annotations():

    def __init__(self,
                 annotation_cfg: Dict) -> None:
        self.annotation_cfg = annotation_cfg

    # TODO find all special characters

    @staticmethod
    def process_string(string):
        for special_char in [".", ",", ":"]:
            result = ""
            i = 0
            while i < len(string):
                if string[i] == special_char:
                    if i > 0 and i < len(string) - 1 and string[i - 1].isalpha() and string[i + 1].isalpha():
                        result += special_char + " "
                    else:
                        result += special_char
                else:
                    result += string[i]
                i += 1
            string = result
        string = result
        return result

    @staticmethod
    def clean_prompt(prompt):
        prompt = " ".join(prompt.split())
        prompt = prompt.replace(" , ", ", ")
        prompt = prompt.replace(" . ", ". ")
        prompt = prompt.replace(" : ", ": ")
        prompt = Annotations.process_string(prompt)
        return prompt
        # return " ".join(prompt.split())
