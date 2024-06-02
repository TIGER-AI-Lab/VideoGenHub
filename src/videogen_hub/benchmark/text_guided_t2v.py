from typing import Optional
import os
from tqdm import tqdm
from videogen_hub.infermodels import load_model
import cv2, json 
import numpy as np
import argparse
from videogen_hub.utils.file_helper import get_file_path
from moviepy.editor import ImageSequenceClip


def infer_text_guided_vg_bench(
    model,
    result_folder: str = "results",
    experiment_name: str = "Exp_Text-Guided_VG",
    overwrite_model_outputs: bool = False,
    overwrite_inputs: bool = False,
    limit_videos_amount: Optional[int] = None,
):
    """
    Performs inference on the VideogenHub dataset using the provided text-guided video generation model.

    Args:
        model: Instance of a model that supports text-guided video generation. Expected to have
               a method 'infer_one_video' for inferencing.
        result_folder (str, optional): Path to the root directory where the results should be saved.
               Defaults to 'results'.
        experiment_name (str, optional): Name of the folder inside 'result_folder' where results
               for this particular experiment will be stored. Defaults to "Exp_Text-Guided_IG".
        overwrite_model_outputs (bool, optional): If set to True, will overwrite any pre-existing
               model outputs. Useful for resuming runs. Defaults to False.
        overwrite_inputs (bool, optional): If set to True, will overwrite any pre-existing input
               samples. Typically, should be set to False unless there's a need to update the inputs.
               Defaults to False.
        limit_videos_amount (int, optional): Limits the number of videos to be processed. If set to
               None, all videos in the dataset will be processed.

    Returns:
        None. Results are saved in the specified directory.

    Notes:
        The function processes each sample from the dataset, uses the model to infer an video based
        on text prompts, and then saves the resulting videos in the specified directories.
    """
    benchmark_prompt_path = "t2v_vbench_1000.json"
    prompts = json.load(open(get_file_path(benchmark_prompt_path), "r"))
    save_path = os.path.join(result_folder, experiment_name, "dataset_lookup.json")
    if overwrite_inputs or not os.path.exists(save_path):
        if not os.path.exists(os.path.join(result_folder, experiment_name)):
            os.makedirs(os.path.join(result_folder, experiment_name))
        with open(save_path, "w") as f:
            json.dump(prompts, f, indent=4)

    print(
        "========> Running Benchmark Dataset:",
        experiment_name,
        "| Model:",
        model.__class__.__name__,
    )

    for file_basename, prompt in tqdm(prompts.items()):
        idx = int(file_basename.split("_")[0])
        dest_folder = os.path.join(
            result_folder, experiment_name, model.__class__.__name__
        )
        # file_basename = f"{idx}_{prompt['prompt_en'].replace(' ', '_')}.mp4"
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        dest_file = os.path.join(dest_folder, file_basename)
        if overwrite_model_outputs or not os.path.exists(dest_file):
            print("========> Inferencing", dest_file)
            frames = model.infer_one_video(prompt=prompt["prompt_en"])
            print("======> frames.shape", frames.shape)

            #special_treated_list = ["LaVie", "ModelScope", "T2VTurbo"]
            special_treated_list = []
            if model.__class__.__name__ in special_treated_list:
                print("======> Saved through cv2.VideoWriter_fourcc")
                # save the video
                fps = 8
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
                out = cv2.VideoWriter(
                    dest_file, fourcc, fps, (frames.shape[2], frames.shape[1])
                )

                # Convert each tensor frame to numpy and write it to the video
                for i in range(frames.shape[0]):
                    frame = frames[i].numpy().astype(np.uint8)
                    out.write(frame)

                out.release()
            else:
                def tensor_to_video(tensor, output_path, fps=8):
                    """
                    Converts a PyTorch tensor to a video file.
                    
                    Args:
                        tensor (torch.Tensor): The input tensor of shape (T, C, H, W).
                        output_path (str): The path to save the output video.
                        fps (int): Frames per second for the output video.
                    """
                    # Ensure the tensor is on the CPU and convert to NumPy array
                    tensor = tensor.cpu().numpy()
                    
                    # Normalize the tensor values to [0, 1]
                    tensor_min = tensor.min()
                    tensor_max = tensor.max()
                    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
                    
                    # Permute dimensions from (T, C, H, W) to (T, H, W, C) and scale to [0, 255]
                    video_frames = (tensor.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                    
                    # Create a video clip from the frames
                    clip = ImageSequenceClip(list(video_frames), fps=fps)
                    
                    # Write the video file
                    clip.write_videofile(output_path, codec='libx264')

                if frames.shape[-1] == 3:
                    frames = frames.permute(0, 3, 1, 2)
                    print("======> corrected frames.shape", frames.shape)

                tensor_to_video(frames, dest_file)
        else:
            print("========> Skipping", dest_file, ", it already exists")

        if limit_videos_amount is not None and (idx >= limit_videos_amount):
            break


# for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model by name")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
    args = parser.parse_args()
    
    model = load_model(args.model_name)
    infer_text_guided_vg_bench(model)
