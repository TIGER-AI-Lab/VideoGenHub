import torch


class ShowOne():
    def __init__(self):
        """
        Initialize the Pipeline, which download all necessary models.
        """
        from videogen_hub.pipelines.show_1.run_inference import ShowOnePipeline

        self.pipeline = ShowOnePipeline()

    def infer_one_video(self,
                        prompt: str = None,
                        size: list = [320, 512],
                        seconds: int = 2,
                        fps: int = 8,
                        seed: int = 42):
        """
        Generates a single video based on a textual prompt. The output is a tensor representing the video.
        Since the initial_num_frames is set to be 8 as shown in paper in the pipeline,
        we need the (number of frames - 1) divisible by 7 to manage interpolation.
    
        Args:
            prompt (str, optional): The text prompt that guides the video generation. If not specified, the video generation will rely solely on the input image. Defaults to None.
            size (list, optional): Specifies the resolution of the output video as [height, width]. Defaults to [320, 512].
            seconds (int, optional): The duration of the video in seconds. Defaults to 2.
            fps (int, optional): The number of frames per second in the generated video. This determines how smooth the video appears. Defaults to 8.
            seed (int, optional): A seed value for random number generation, ensuring reproducibility of the video generation process. Defaults to 42.
    
        Returns:
            torch.Tensor: A tensor representing the generated video, structured as (time, channel, height, width).
        """
        num_frames = fps * seconds

        assert (num_frames - 1) % 7 == 0
        scaling_factor = (num_frames - 1) // 7
        video = self.pipeline.inference(prompt=prompt,
                                        negative_prompt="",
                                        output_size=size,
                                        initial_num_frames=8,
                                        scaling_factor=scaling_factor,
                                        seed=seed)

        return video
