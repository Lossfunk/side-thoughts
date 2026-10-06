import os
import json
import torch

from openai import OpenAI
from utils import *

precision_data = torch.bfloat16

def get_response(client, prompt) -> tuple[str, str]:
    response = client.embeddings.create(
        input=[prompt],
        # model="qwen/qwen3-embedding-4b",
        model="google/gemini-embedding-2",
        # model="openai/text-embedding-3-large",
        # model="perplexity/pplx-embed-v1-4b",
        encoding_format="float"
    )
    return response.data[0].embedding    


def simulate_responses(save_dir: str = "logs") -> None:
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    client = OpenAI(base_url = "https://openrouter.ai/api/v1")
    data_dict = {
        "solution": [],
        "embedding": []
    }
    with open("./datasets/test_set.json", "r") as f:
        data = json.load(f)
    for prompts in data["solution"]:
        print(prompts)
        for solution in prompts:
            embedding = get_response(client, solution)
            data_dict["solution"].append(solution)
            data_dict["embedding"].append(torch.Tensor(embedding).to(precision_data).clone())
    
    print(len(data_dict["embedding"]))
    
    save_name = os.path.join(save_dir, "embeddings_teacher_qwen.pt")
    torch.save(data_dict, save_name)
    get_remaining_credits(client)


if __name__ == "__main__":
    simulate_responses()