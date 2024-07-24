from abc import ABC
from typing import List, Any


class BaseInferModel(ABC):
    resolution = [320, 512]
    seconds = 2
    fps = 8
    seed = 42
    model = None
    pipeline = None
    components: List[Any] = []

    def download_models(self) -> str:
        pass

    def to(self, device="cpu"):
        # If the pipeline is not None and has a ".to" method, unload it from the device
        all_components = [self.pipeline] + self.components
        for component in all_components:
            try:
                if component is not None and hasattr(component, "to"):
                    component.to(device)
            except Exception as e:
                print(f"Error moving component to {device}: {e}")

    def load_pipeline(self):
        pass
