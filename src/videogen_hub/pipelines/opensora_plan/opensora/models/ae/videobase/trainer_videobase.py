import json
import os
from typing import Optional

import torch
from transformers import Trainer
from transformers.utils import WEIGHTS_NAME


class VideoBaseTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()

        # get model config
        model_config = self.model.config.to_dict()

        # add more information
        model_config['model'] = self.model.__class__.__name__

        with open(os.path.join(output_dir, "config.json"), "w") as file:
            json.dump(self.model.config.to_dict(), file)
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
