import json


def main(prompt_path):
    new_prompt = {}
    prompts = json.load(open(prompt_path, "r"))

    for idx, prompt in enumerate(prompts):
        new_prompt[f"{idx}_{prompt['prompt_en'].replace(' ', '_')}.mp4"] = prompt

    with open(f"new_{prompt_path}", "w") as f:
        json.dump(new_prompt, f, indent=4)


if __name__ == "__main__":
    # main("t2v_vbench_200.json")
    main("t2v_vbench_remain.json")
