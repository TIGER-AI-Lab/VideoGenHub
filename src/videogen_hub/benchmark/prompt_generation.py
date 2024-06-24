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

    target_prompts_count = 800
    # sample prompts based on the distribution of dimensions
    sampled_prompts = list()
    remaining_prompts = list()
    dimension_probs = np.array(list(dimension_count_map.values())) / dimensions_count
    dimensions = list(dimension_count_map.keys())
    sample_counts = np.random.multinomial(target_prompts_count, dimension_probs)
    print(sample_counts)
    for dimension, count in zip(dimensions, sample_counts):

        sampled_prompts_idx = random.sample(dimension_prompt_idx_map[dimension], count)
        for idx in range(len(prompts)):
            if idx in sampled_prompts_idx:
                sampled_prompts.append(prompts[idx])
            else:
                remaining_prompts.append(prompts[idx])

    save_path = "./t2v_vbench_1000.json"
    remaing_data_save_path = "./t2v_vbench_remain_1000.json"
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
    main(prompt_path="VBench_full_info.json")
    # main(prompt_path="t2v_vbench_remain_200.json")