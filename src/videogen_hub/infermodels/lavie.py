
class LaVie():
    def __init__(self, device="cuda"):
        
        raise NotImplementedError

    def infer_one_video(self,
                        prompt: str = None, 
                        size: list = [320, 512],
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        raise NotImplementedError

    