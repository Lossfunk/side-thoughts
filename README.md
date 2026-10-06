# Side Thoughts: Understanind Drug Repositioning via Chemical Reasoning

This repository contains the official codebase for _Crosscoder Steering Structures and Stabilizes Multi-Naswer Biological Reasoning_. Crosscoder-guided steering and teacher-guided steering are applied to structure and stabilize token activations during multi-answer biological reasoning.

## Instructions

1. Setup the venv and install packages.

```
bash setup.sh
source check/bin/activate
cd side-thoughts
```

An OpenRouter API Key is required for grading. Enter your API key in the environment.

```
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

2. (Optional) Create the repurposing dataset.

Preprocessed datasets are stored in the `datasets` directory.

```
python3 generate_dataset.py
```

3. Harvest activations for steering.

```
python3 harvest_activations.py
```
Last-token harvested activations will be stored in `logs` directory.

4. Harvest teacher embeddings for teacher steering.

```
python3 generate_embeddings.py
```
Harvested text embeddings will be stored in `logs` directory.

5. Evaluate with distributional steering.

```
python3 evaluate.py
```

Metrics, trace and response activations will be stored in the `logs` directory.
