# Adding new models

* You developed a new model / framework that perform very good results. Now you want to benchmark it with other models. How you can do it?

In this guide we will be adding new model to the codebase and extend the code.

## Integrating your model into VideoGenHub


To add your model codebase into VideoGenHub codebase, you must modify the following folders:

* `src/videogen_hub/infermodels` : where you create a class interface for the model inference.
* `src/videogen_hub/pipelines` : where you move your codebase into it without much tidy up work.

### How to write the infermodel class
The infermodel class is designed to have minimal methods. However, it must contain the following methods:

* `__init__(args)` for class initialization.
* `infer_one_video(args)` to produce 1 video output. Please try to set the seed as 42.

In that case, you will add a new file in `infermodels` folder.
`infermodels/awesome_model.py`
```python
import torch
from videogen_hub.pipelines.awesome_model import AwesomeModelPipeline
class AwesomeModelClass():
    """
    A wrapper ...
    """
    def __init__(self, device="cuda"):
        """
        docstring
        """
        self.pipe = AwesomeModelPipeline(device=device)

    def infer_one_video(self, prompt, seed=42):
        """
        docstring
        """
        self.pipe.set_seed(seed)
        video = self.pipe(prompt=prompt)
        return video
```
Then you can add a line in `infermodels/__init__.py`:
```shell
from .awesome_model import AwesomeModelClass
```

### Writing your pipeline
About `AwesomeModelPipeline`, it means you need to write a Pipeline file that wraps the function of your codebase, such that the infermodel class can call it with ease.

We recommend structuring code in the `pipelines` folder in this way:

```shell
└── awesome_model
    ├── pipeline_awesome_model.py
    ├── awesome_model_src
    │   └── ...
    └── __init__.py
```

## Running experiment with new model
After finishing and reinstalling the package through 
```shell
pip install -e .
```
You should be able to use the new model.


### Matching environment
Make sure the code can be run with the VideoGenHub environment. If new dependency is added, please add them to the env_cfg file.

## Submitting your model as through a PR

Finally, you can submit this new model through submiting a Pull Request! Make sure it match the code style in our contribution guide.