import os
import re
import torch
from datasets import load_dataset, Dataset


def preprocess_dataset(train_name, test_name, system_prompt):
    train = load_dataset("json", data_files = train_name)["train"]
    test = Dataset.from_dict(load_dataset("json", data_files = test_name)["train"][0]).select([i for i in range(0, 10)])
    
    def format_train_example(x):
        ans = " ".join(x["solution"])

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": ans
        }
    
    def format_test_example(x):
        ans = " ".join(x["solution"])

        return {
            "prompt": [
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": ans
        }

    train = train.map(format_train_example, remove_columns=train.column_names)
    test = test.map(format_test_example, remove_columns=test.column_names)
    return train, test

def get_reward_fns(match_format, think_format, reasoning_start, reasoning_end):

    def format_reward_func(completions, **kwargs):
        scores = []

        for completion_item in completions:
            if not completion_item or not isinstance(completion_item[0], dict) or "content" not in completion_item[0]:
                scores.append(-2.0)
            else:
                scores.append(0.0)
        return scores

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if match_format.search(response) is not None: score += 2.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score += 0.5 if response.count(reasoning_start) == 1 else -0.5
            score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            # 3 points for each correct answer!
            for mol in true_answer.split():
                if mol in guess:
                    score += 3.0
            scores.append(score)
        return scores
        
    def think_bonus(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := think_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-1.0)
                continue
            # 2 point for each correct answer!
            for mol in true_answer.split():
                if mol in guess:
                    score += 2.0
            scores.append(score)
        return scores
    
    return format_reward_func, match_format_exactly, match_format_approximately, check_answer, think_bonus
    

def log_inference_metrics(test, outputs):
    for sample, output in zip(test, outputs):
        answer = sample["answer"].split()

def load_crosscoder_data(input_dim, dims, crosscoder_dir):
    small = []
    medium = []
    large = []
    pos_acts = []
    neg_acts = []
    for f in os.listdir(crosscoder_dir):
        acts = torch.load(os.path.join(crosscoder_dir, f), weights_only = False)
        if "activations_correct" in acts:
            if acts["activations_correct"][0].shape[-1] == dims[0]:
                if input_dim == dims[0]:
                    test_idx = len(acts["activations_correct"]) // 10
                    pos_acts = pos_acts + acts["activations_correct"][:test_idx]
                    neg_acts = neg_acts + acts["activations_incorrect"][:test_idx]
                    small = small + acts["activations_correct"][test_idx:]
                    small = small + acts["activations_incorrect"][test_idx:]
                else:
                    small = small + acts["activations_correct"]
                    small = small + acts["activations_incorrect"]
            elif acts["activations_correct"][0].shape[-1] == dims[1]:
                if input_dim == dims[1]:
                    test_idx = len(acts["activations_correct"]) // 10
                    pos_acts = pos_acts + acts["activations_correct"][:test_idx]
                    neg_acts = neg_acts + acts["activations_incorrect"][:test_idx]
                    medium = medium + acts["activations_correct"][test_idx:]
                    medium = medium + acts["activations_incorrect"][test_idx:]
                else:
                    medium = medium + acts["activations_correct"]
                    medium = medium + acts["activations_incorrect"]
            else:
                if input_dim == dims[2]:
                    test_idx = len(acts["activations_correct"]) // 10
                    pos_acts = pos_acts + acts["activations_correct"][:test_idx]
                    neg_acts = neg_acts + acts["activations_incorrect"][:test_idx]
                    large = large + acts["activations_correct"][test_idx:]
                    large = large + acts["activations_incorrect"][test_idx:]
                else:
                    large = large + acts["activations_correct"]
                    large = large + acts["activations_incorrect"]
        elif "activations_think" in acts:
            if acts["activations_think"][0].shape[-1] == dims[0]:
                small = small + acts["activations_think"]
                small = small + acts["activations_guess"]
            elif acts["activations_think"][0].shape[-1] == dims[1]:
                medium = medium + acts["activations_think"]
                medium = medium + acts["activations_guess"]
            else:
                large = large + acts["activations_think"]
                large = large + acts["activations_guess"]            
        else:
            if acts["embedding"][0].shape[-1] == dims[0]:
                small = small + acts["embedding"]
            elif acts["embedding"][0].shape[-1] == dims[1]:
                medium = medium + acts["embedding"]
            else:
                large = large + acts["embedding"]
    return small, medium, large, pos_acts, neg_acts
