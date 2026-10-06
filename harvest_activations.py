import os
import re
import torch
import wandb
import csv
import json
import numpy as np
from unsloth import FastLanguageModel
from vllm import SamplingParams

from openai import OpenAI
from data_utils import *
from utils import *
from metrics import *
from steering_wrapper import *

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_USE_ASCEND"] = "0"

judge_prompt = \
f"""You are given a problem and a solution.
Provide the number of correct answers in the solution. Your output must be a single number.\n"""

def get_response(prompt, model, client) -> tuple[str, str]:
    ip_prompt = f"List 10 common drugs that can be repurposed to treat {prompt}."
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": judge_prompt + ip_prompt},
        ],
    )
    response = completion.choices[0].message.content
    return response

def filter_result(response: str) -> float:
    response = response.replace(".", "")
    numbers = re.findall(r'\d+\.?\d*', response)
    if len(numbers) >= 1:
        return numbers[0]
    return "0"

client = OpenAI(base_url = "https://openrouter.ai/api/v1")
model_name = "~openai/gpt-mini-latest"
experiment_name = "r1_activations_base"
save_dir = "./logs"
layers = 36
precision_data = torch.bfloat16
max_seq_length = 4096
gen_length = 4096
n = 10 # dry run
lora_rank = 8
user_token = None
assistant_token = None

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-0528-Qwen3-8B",
    # model_name = "unsloth/Qwen3-8B",
    # model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    # model_name = "unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit",
    # model_name = "unsloth/gemma-4-12b-it-GGUF",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)


reasoning_start = "<think>"
reasoning_end = "</think>"
for token in tokenizer.get_added_vocab().keys():
    if "think" in token and "/" in token:
        reasoning_end = token
    elif "think" in token:
        reasoning_start = token
    elif "user" in token:
        user_token = token
    elif "assistant" in token:
        assistant_token = token

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
You must think in English."""

solution_end_regex = rf"{reasoning_end}(.*)"
match_format = re.compile(solution_end_regex, re.DOTALL)
think_regex = rf"{reasoning_start}(.*){re.escape(reasoning_end)}"
think_format = re.compile(think_regex, re.DOTALL)

train, test = preprocess_dataset("datasets/train_set.json", "datasets/test_set.json", system_prompt)


# inference
FastLanguageModel.for_inference(model)

torch.cuda.empty_cache()
del train

metrics = {
    "answer/correct": [],
    "answer/hit": [],
    "answer/precision": [],
    "answer/recall": [],
    "answer/f1": [],
    "answer/pass@1": [],
    "answer/pass@2": [],
    "answer/pass@5": [],
    "answer/pass@10": [],
    "think/correct": [],
    "think/hit": [],
    "think/precision": [],
    "think/recall": [],
    "think/f1": [],
    "think/pass@1": [],
    "think/pass@2": [],
    "think/pass@5": [],
    "think/pass@10": [],
}
logs = {
    "prompt": [],
    "answer": [],
    "trace": [],
    "response": [],
    "activations_correct": [], # [layers, 1, seq, dim]
    "activations_incorrect": [] # [layers, 1, seq, dim]
}


for i, sample in enumerate(test):
    torch.cuda.empty_cache()
    sample_text = tokenizer.apply_chat_template(
        sample["prompt"], tokenize=False,
        add_generation_prompt = True
    )
    sampling_params = SamplingParams(
            temperature=0.7,
            top_k=50,
            max_tokens=gen_length,
            n=1
    )
    outputs = model.fast_generate(
        [sample_text]*n,
        sampling_params = sampling_params,
    )

    solution = sample["answer"].split(" ")
    corrects = []
    hits = []
    precs = []
    recs = []
    f1s = []
    corrects_think = []
    hits_think = []
    precs_think = []
    recs_think = []
    f1s_think = []
    texts = []
    for k in range(n):
        output_text = outputs[k].outputs[0].text
        response = get_response(output_text, model_name, client)
        correct = int(filter_result(response))
        hit = correct / 10

        if reasoning_end in output_text:
            think, guess = output_text.split(reasoning_end, 1)
        elif "Assistant" in output_text:
            think, guess = output_text.split("Assistant", 1)
        else:
            think, guess = "None", output_text

        texts.append((think, guess))
        precs_local = []
        recs_local = []
        f1s_local = []
        corrects_local = []
        hits_local = []
        precs_think_local = []
        recs_think_local = []
        f1s_think_local = []
        corrects_think_local = []
        hits_think_local = []

        for answer in solution:
            precision, recall, f1 = token_f1(guess, answer)
            precs_local.append(precision)
            recs_local.append(recall)
            f1s_local.append(f1)
            corrects_local.append(correct_samples(guess, answer))
            hits_local.append(hit_rate(guess, answer))

            precision, recall, f1 = token_f1(think, answer)
            precs_think_local.append(precision)
            recs_think_local.append(recall)
            f1s_think_local.append(f1)
            corrects_think_local.append(correct_samples(think, answer))
            hits_think_local.append(hit_rate(think, answer))
        
        precs.append(np.mean(precs_local))
        recs.append(np.mean(recs_local))
        f1s.append(np.mean(f1s_local))

        precs_think.append(np.mean(precs_think_local))
        recs_think.append(np.mean(recs_think_local))
        f1s_think.append(np.mean(f1s_think_local))
        corrects_think.append(np.mean(corrects_think_local))
        hits_think.append(np.mean(hits_think_local))

        corrects.append(correct)
        hits.append(hit)

    del outputs

    # collect best sample for global logging
    max_ele = max(corrects)
    best_idx = corrects.index(max_ele)
    think, guess = texts[best_idx]

    metrics["answer/correct"].append(np.mean(corrects))
    metrics["answer/hit"].append(np.mean(hits))
    metrics["answer/pass@1"].append(pass_at_k(sum([1 for x in corrects if x > 0]), n, k = 1))
    metrics["answer/pass@2"].append(pass_at_k(sum([1 for x in corrects if x > 0]), n, k = 2))
    metrics["answer/pass@5"].append(pass_at_k(sum([1 for x in corrects if x > 0]), n, k = 5))
    metrics["answer/pass@10"].append(pass_at_k(sum([1 for x in corrects if x > 0]), n, k = 10))
    metrics["answer/precision"].append(np.mean(precs))
    metrics["answer/recall"].append(np.mean(recs))
    metrics["answer/f1"].append(np.mean(f1s))

    metrics["think/correct"].append(np.mean(corrects_think))
    metrics["think/hit"].append(np.mean(hits_think))
    metrics["think/pass@1"].append(pass_at_k(sum([1 for x in corrects_think if x > 0]), n, k = 1))
    metrics["think/pass@2"].append(pass_at_k(sum([1 for x in corrects_think if x > 0]), n, k = 2))
    metrics["think/pass@5"].append(pass_at_k(sum([1 for x in corrects_think if x > 0]), n, k = 5))
    metrics["think/pass@10"].append(pass_at_k(sum([1 for x in corrects_think if x > 0]), n, k = 10))
    metrics["think/precision"].append(np.mean(precs_think))
    metrics["think/recall"].append(np.mean(recs_think))
    metrics["think/f1"].append(np.mean(f1s_think))
    logs["prompt"].append(sample["prompt"])
    logs["answer"].append(sample["answer"])
    logs["trace"].append(think)
    logs["response"].append(guess)

    # second forward pass for activations - best
    torch.cuda.empty_cache()
    inputs_guess = tokenizer(think  + "\n" + guess, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(
            **inputs_guess,
            output_hidden_states=True,
        )
    activations = torch.stack(outputs.hidden_states, dim = 0).squeeze(1).to("cpu")
    logs["activations_correct"].append(activations.to(precision_data)[:, -1, :].clone())
    del inputs_guess
    del outputs

    # second forward pass for activations - worst
    torch.cuda.empty_cache()
    min_ele = min(corrects)
    worst_idx = corrects.index(min_ele)
    think, guess = texts[worst_idx]

    inputs_guess = tokenizer(think  + "\n" + guess, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(
            **inputs_guess,
            output_hidden_states=True,
        )
    activations = torch.stack(outputs.hidden_states, dim = 0).squeeze(1).to("cpu")
    logs["activations_incorrect"].append(activations.to(precision_data)[:, -1, :].clone())
    del inputs_guess
    del outputs

    # break # dry run

logs = logs | metrics
torch.save(logs, f'logs/{experiment_name}.pt')
get_remaining_credits(client)

