import re
import string
from collections import Counter

import httpx
import pandas as pd
import torch


def prepare_content(content: str) -> str:
    sentences = content.split("\n")
    sentences = [s for s in sentences if len(s) > 15]
    return sentences

def calculate_bert_score(pred: list[str], true: list[str] | list[list[str]]) -> dict:
    "TODO rewrite as local function not as a microservice"
    if pred is None:
        pred = []
    if true is None:
        true = []
    url = "http://34.147.16.246:57546/predict"
    payload = {"pred": pred, "true": true}
    try:
        headers = {"Content-Type": "application/json"}
        response = httpx.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: Status code {response.status_code}, {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"


def calculate_surprisal_score(
    summary: str,
    model: object,
    tokenizer: object,
    verbose: bool = False,
    device: str = "cpu",
) -> dict:
    """
    Calculate surprisal scores for a given text summary.

    Args:
        summary: Text to calculate surprisal for
        model: Language model to use
        tokenizer: Tokenizer for the model
        verbose: Whether to print debug information and return detailed outputs
        device: Device to run computation on ('cpu' or 'cuda')

    Returns:
        Dictionary containing surprisal statistics
    """
    # Prepare input text
    bos_token = tokenizer.bos_token or ""
    text = bos_token + summary

    # Tokenize and prepare inputs/targets
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    targets = input_ids.clone()

    # Shift targets to predict next token
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = tokenizer.eos_token_id

    if verbose:
        print("input_ids:", input_ids)
        print("targets:", targets)

    # Calculate token surprisals
    with torch.no_grad():
        outputs = model(input_ids, labels=targets)
        logits = outputs.logits
        log_probs = -1 * torch.log_softmax(logits, dim=-1)

        token_surprisals = []
        batch_size, seq_length = targets.shape

        for batch_idx in range(batch_size):
            for pos in range(seq_length):
                target_token = targets[batch_idx, pos]
                if verbose:
                    print(
                        f"input_ids: {input_ids[batch_idx, pos]}, target_token: {target_token}"
                    )
                token_surprisal = log_probs[batch_idx, pos, target_token].item()
                token_surprisals.append(token_surprisal)

    # Convert list to tensor and calculate statistics
    token_surprisals = torch.tensor(token_surprisals)
    overall_surprisal_sum = token_surprisals.sum().item()
    overall_surprisal_mean = token_surprisals.mean().item()

    return {
        "surprisal_sum": overall_surprisal_sum,
        "surprisal_mean": overall_surprisal_mean,
    }


# TRIVIA QA EVALUATION
# ------------------------------------
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
# https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def f1_score(prediction: str, ground_truth: list[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: list[str]) -> int:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(
    metric_fn: callable, prediction: str, ground_truths: list[str]
) -> float:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_triviaqa(ground_truth: list[str], prediction: str) -> dict[str, float]:
    f1 = exact_match = 0

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    em_for_this_question = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truth
    )
    exact_match += em_for_this_question
    f1_for_this_question = metric_max_over_ground_truths(
        f1_score, prediction, ground_truth
    )
    f1 += f1_for_this_question

    return {"exact_match": exact_match, "f1": f1}


def check_answer_em(row: pd.Series) -> bool:
    try:
        e = evaluate_triviaqa(row["true"], row["pred"])
        return e["exact_match"]
    except Exception as e:
        return False



def check_answer_f1(row: pd.Series) -> float:
    try:
        e = evaluate_triviaqa(row["true"], row["pred"])
        return e["f1"]
    except Exception as e:
        return 0
