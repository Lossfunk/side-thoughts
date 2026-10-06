import re
from collections import Counter
from math import comb
import unicodedata

subscript_to_normal = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎",
    "0123456789+-=()"
)

def un_subscript(s):
    return s.translate(subscript_to_normal)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return un_subscript(normalize(text)).split()

def token_f1(pred, truth):
    pred_tokens = tokenize(pred)
    truth_tokens = tokenize(truth)

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    common = pred_counter & truth_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0, 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def correct_samples(guess, answer):
    guess = tokenize(guess)
    answer = tokenize(answer)
    correct = 0
    for mol in answer:
        if mol in guess:
            correct += 1
    return correct

def hit_rate(guess, answer):
    guess = tokenize(guess)
    answer = tokenize(answer)
    if not answer:
        return 0
    hit = 0
    for mol in answer:
        if mol in guess:
            hit += 1
    return hit / len(answer)

def pass_at_k(c, n, k = 1):
    if c == 0:
        return 0.0
    if c >= n or k >= n:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


