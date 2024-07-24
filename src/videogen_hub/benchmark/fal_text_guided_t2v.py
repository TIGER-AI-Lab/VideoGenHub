import json
import os
from typing import Optional

import fal_client
import requests
from tqdm import tqdm


# import json

def infer_text_guided_vg_bench(
        model_name,
        result_folder: str = "results",
        experiment_name: str = "Exp_Text-Guided_VG",
        overwrite_model_outputs: bool = False,
        overwrite_inputs: bool = False,
        limit_videos_amount: Optional[int] = None,
):
    """
    Performs inference on the VideogenHub dataset using the provided text-guided video generation model.

    Args:
        model_name: name of the model we want to run inference on
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
    prompts = json.load(open(benchmark_prompt_path, "r"))
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
        model_name,
    )

    if model_name == 'AnimateDiff':
        fal_model_name = 'fast-animatediff/text-to-video'
    elif model_name == 'AnimateDiffTurbo':
        fal_model_name = 'fast-animatediff/turbo/text-to-video'
    elif model_name == 'FastSVD':
        fal_model_name = 'fast-svd/text-to-video'
    else:
        raise ValueError("Invalid model_name")

    for file_basename, prompt in tqdm(prompts.items()):
        idx = int(file_basename.split('_')[0])
        dest_folder = os.path.join(
            result_folder, experiment_name, model_name
        )
        # file_basename = f"{idx}_{prompt['prompt_en'].replace(' ', '_')}.mp4"
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        dest_file = os.path.join(dest_folder, file_basename)
        if overwrite_model_outputs or not os.path.exists(dest_file):
            print("========> Inferencing", dest_file)

            handler = fal_client.submit(
                f"fal-ai/{fal_model_name}",
                arguments={
                    "prompt": prompt["prompt_en"]
                },
            )

            # for event in handler.iter_events(with_logs=True):
            #     if isinstance(event, fal_client.InProgress):
            #         print('Request in progress')
            #         print(event.logs)

            result = handler.get()
            result_url = result['video']['url']
            download_mp4(result_url, dest_file)
        else:
            print("========> Skipping", dest_file, ", it already exists")

        if limit_videos_amount is not None and (idx >= limit_videos_amount):
            break


def download_mp4(url, filename):
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Open a local file with write-binary mode
        with open(filename, 'wb') as file:
            # Write the response content to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # print(f"Download complete: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


if __name__ == "__main__":
    pass
    # infer_text_guided_vg_bench(model_name="AnimateDiff")
    infer_text_guided_vg_bench(result_folder="/mnt/tjena/maxku/max_projects/VideoGenHub/results", model_name="FastSVD")
    # infer_text_guided_vg_bench(result_folder="/mnt/tjena/maxku/max_projects/VideoGenHub/results", model_name="AnimateDiff")
    # infer_text_guided_vg_bench(result_folder="/mnt/tjena/maxku/max_projects/VideoGenHub/results", model_name="AnimateDiffTurbo")
