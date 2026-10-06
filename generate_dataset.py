
import os
import pandas as pd
import json
import random

n_test_samples = 100

def get_pairs():
    prompts = []
    df = pd.read_excel("./datasets/pairs.xlsx", sheet_name = "Drug Repurposing Dataset")
    for idx, row in df.iterrows():
        side_effect = row["Condition / Side-Effect"]
        mols = (row["Drug 1"], row["Drug 2"], row["Drug 3"], row["Drug 4"], row["Drug 5"])
        pair = (side_effect, mols)
        prompts.append(pair)
    random.shuffle(prompts)
    train_prompts = prompts[n_test_samples:]
    test_prompts = prompts[:n_test_samples]
    return train_prompts, test_prompts

def save_pairs():
    save_dir = "./datasets"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    data_dict = {
        "prompt": [],
        "solution": []
    }
    train_prompts, test_prompts = get_pairs()
    print(len(train_prompts), len(test_prompts))

    for prompt, solution in train_prompts:
        ip_prompt = f"List 10 common drugs that can be repurposed to treat {prompt}."
        data_dict["prompt"].append(ip_prompt)
        data_dict["solution"].append(list(solution))
    
    save_name = os.path.join(save_dir, "train_set.json")
    with open(save_name, "w", newline = "") as f:
        json.dump(data_dict, f, indent = 2)

    test_dict = {
        "prompt": [],
        "solution": []
    }

    for prompt, solution in test_prompts:
        ip_prompt = f"List 10 common drugs that can be repurposed to treat {prompt}."
        test_dict["prompt"].append(ip_prompt)
        test_dict["solution"].append(list(solution))
        
    save_name = os.path.join(save_dir, "test_set.json")
    with open(save_name, "w", newline = "") as f:
        json.dump(test_dict, f, indent = 2)

save_pairs()
