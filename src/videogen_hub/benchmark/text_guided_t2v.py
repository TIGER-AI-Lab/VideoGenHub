from typing import Union, Optional, Callable
import os
from tqdm import tqdm
from videogen_hub.infermodels import load_model
import cv2, json, random
import numpy as np


def infer_text_guided_ig_bench(
    model,
    result_folder: str = "results",
    experiment_name: str = "Exp_Text-Guided_IG",
    overwrite_model_outputs: bool = False,
    overwrite_inputs: bool = False,
    limit_videos_amount: Optional[int] = None,
):
    """
    Performs inference on the VideonHub dataset using the provided text-guided video generation model.

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
    prompts = json.load(open("VBench_full_info.json", "r"))

    # construct dimension_count map
    dimension_count_map = {}
    dimension_prompt_idx_map = {}
    dimensions_count = 0
    for i in range(len(prompts)):
        prompt = prompts[i]
        dimensions = prompt["dimension"]
        for dimension in dimensions:
            if dimension not in dimension_prompt_idx_map:
                dimension_prompt_idx_map[dimension] = []
            dimension_prompt_idx_map[dimension].append(i)

            if dimension not in dimension_count_map:
                dimension_count_map[dimension] = 0

            dimension_count_map[dimension] += 1

            dimensions_count += 1

    print(
        "Dimensions count (each prompt can contribute to more than one dimension count):",
        dimensions_count,
    )
    print(dimension_count_map)

    target_prompts_count = 200
    # sample prompts based on the distribution of dimensions
    sampled_prompts = list()
    dimension_probs = np.array(list(dimension_count_map.values())) / dimensions_count
    dimensions = list(dimension_count_map.keys())
    sample_counts = np.random.multinomial(target_prompts_count, dimension_probs)
    print(sample_counts)
    for dimension, count in zip(dimensions, sample_counts):

        sampled_prompts_idx = random.sample(dimension_prompt_idx_map[dimension], count)
        for idx in sampled_prompts_idx:
            sampled_prompts.append(prompts[idx])

    save_path = os.path.join(result_folder, experiment_name, "dataset_lookup.json")
    if overwrite_inputs or not os.path.exists(save_path):
        if not os.path.exists(os.path.join(result_folder, experiment_name)):
            os.makedirs(os.path.join(result_folder, experiment_name))
        with open(save_path, "w") as f:
            json.dump(sampled_prompts, f, indent=4)

    print(
        "========> Running Benchmark Dataset:",
        experiment_name,
        "| Model:",
        model.__class__.__name__,
    )

    for idx, prompt in enumerate(tqdm(sampled_prompts)):
        dest_folder = os.path.join(
            result_folder, experiment_name, model.__class__.__name__
        )
        file_basename = f"{idx}_{prompt['prompt_en'].replace(' ', '_')}.mp4"
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        dest_file = os.path.join(dest_folder, file_basename)
        if overwrite_model_outputs or not os.path.exists(dest_file):
            print("========> Inferencing", dest_file)
            frames = model.infer_one_video(prompt=prompt["prompt_en"])

            # save the video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
            out = cv2.VideoWriter(
                dest_file, fourcc, 20.0, (frames.shape[2], frames.shape[1])
            )

            # Convert each tensor frame to numpy and write it to the video
            for i in range(frames.shape[0]):
                frame = frames[i].numpy()
                out.write(frame)

            out.release()
        else:
            print("========> Skipping", dest_file, ", it already exists")

        if limit_videos_amount is not None and (idx >= limit_videos_amount):
            break


# for testing
if __name__ == "__main__":
    model = load_model("ModelScope")
    # model = ""
    infer_text_guided_ig_bench(model, limit_videos_amount=10)
    pass
