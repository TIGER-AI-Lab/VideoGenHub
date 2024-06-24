from abc import ABC


class BaseInferModel(ABC):
    resolution = [320, 512]
    seconds = 2
    fps = 8
    seed = 42
    model = None
    pipeline = None

    def download_models(self) -> str:
        pass

    def unload_pipeline(self):
        pass

    def load_pipeline(self):
        pass

