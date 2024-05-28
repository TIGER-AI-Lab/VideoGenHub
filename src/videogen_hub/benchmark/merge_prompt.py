import json

def main(prmopts_path_1, prompts_path_2):
    prompts_1 = json.load(open(prmopts_path_1, "r"))
    prompts_2 = json.load(open(prompts_path_2, "r"))
    
    new_prompts = {}
    new_idx = 0
    for prompt_key in prompts_1:
        prompt_key_lst = prompt_key.split("_")
        prompt_key_lst[0] = str(new_idx)
        new_prompts['_'.join(prompt_key_lst)] = prompts_1[prompt_key]
        new_idx += 1
    
    for prompt_key in prompts_2:
        prompt_key_lst = prompt_key.split("_")
        prompt_key_lst[0] = str(new_idx)
        new_prompts['_'.join(prompt_key_lst)] = prompts_2[prompt_key]
        new_idx += 1
    
    with open(f"t2v_vbench_1000.json", "w") as f:
        json.dump(new_prompts, f, indent=4)

if __name__ == "__main__":
    main("t2v_vbench_200.json", "t2v_vbench_800.json")