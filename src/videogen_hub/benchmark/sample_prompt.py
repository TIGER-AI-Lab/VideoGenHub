import json
import os
import random

import numpy as np


# Randomly sample a subset of prompts for benchmarking
def main(prompt_path, overwrite_inputs=False):
    prompts = json.load(open(prompt_path, "r"))

    # construct dimension_count map
    dimension_count_map = {}
    dimension_prompt_idx_map = {}
    dimensions_count = 0
    for key, prompt in prompts.items():
        dimensions = prompt["dimension"]
        for dimension in dimensions:
            if dimension not in dimension_prompt_idx_map:
                dimension_prompt_idx_map[dimension] = []
            dimension_prompt_idx_map[dimension].append(key)

            if dimension not in dimension_count_map:
                dimension_count_map[dimension] = 0

            dimension_count_map[dimension] += 1

            dimensions_count += 1

    print(
        "Dimensions count (each prompt can contribute to more than one dimension count):",
        dimensions_count,
    )
    print(dimension_count_map)

    target_prompts_count = 800
    # sample prompts based on the distribution of dimensions
    sampled_prompts = {}
    remaining_prompts = {}
    dimension_probs = np.array(list(dimension_count_map.values())) / dimensions_count
    dimensions = list(dimension_count_map.keys())
    sample_counts = np.random.multinomial(target_prompts_count, dimension_probs)
    print(np.sum(sample_counts))
    print(sample_counts)
    for dimension, count in zip(dimensions, sample_counts):

        sampled_prompts_keys = random.sample(dimension_prompt_idx_map[dimension], count)
        for key in prompts.keys():
            if key in sampled_prompts_keys:
                while key in sampled_prompts:
                    key = random.sample(dimension_prompt_idx_map[dimension], 1)[0]
                sampled_prompts[key] = prompts[key]
            else:
                remaining_prompts[key] = prompts[key]

    save_path = "./t2v_vbench_800.json"
    remaing_data_save_path = "./t2v_vbench_remain_1000.json"
    print(len(sampled_prompts.keys()))
    if overwrite_inputs or not os.path.exists(save_path):
        # if not os.path.exists(os.path.join(result_folder, experiment_name)):
        #     os.makedirs(os.path.join(result_folder, experiment_name))
        with open(save_path, "w") as f:
            json.dump(sampled_prompts, f, indent=4)

        with open(remaing_data_save_path, "w") as f:
            json.dump(remaining_prompts, f, indent=4)
    else:
        print("Dataset already exists, skipping generation")


if __name__ == "__main__":
    # main(prompt_path="VBench_full_info.json")
    main(prompt_path="t2v_vbench_remain_200.json")
