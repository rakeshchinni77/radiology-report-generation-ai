"""Core text normalization and report-level evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


SPECIAL_TOKENS = ("[start]", "[end]", "[pad]")


def clean_text(text: str) -> str:
    """Normalize report text for metric computation."""
    normalized = text.lower()
    for token in SPECIAL_TOKENS:
        normalized = normalized.replace(token, " ")
    normalized = " ".join(normalized.split())
    return normalized.strip()


def tokenize_text(text: str) -> list[str]:
    """Split cleaned text into whitespace tokens."""
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def prepare_references_and_predictions(
    references: Sequence[str],
    predictions: Sequence[str],
) -> tuple[list[list[list[str]]], list[list[str]]]:
    """Convert raw strings into corpus-bleu compatible token lists."""
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction length mismatch: {len(references)} references vs {len(predictions)} predictions"
        )

    references_tokens: list[list[list[str]]] = []
    predictions_tokens: list[list[str]] = []

    for reference, prediction in zip(references, predictions, strict=True):
        references_tokens.append([tokenize_text(reference)])
        predictions_tokens.append(tokenize_text(prediction))

    return references_tokens, predictions_tokens


def compute_bleu_scores(
    references: Sequence[str],
    predictions: Sequence[str],
    smoothing: bool = True,
) -> dict[str, float]:
    """Compute BLEU-1 through BLEU-4 using NLTK corpus BLEU."""
    references_tokens, predictions_tokens = prepare_references_and_predictions(references, predictions)
    smoothing_function = SmoothingFunction().method1 if smoothing else None

    bleu_weights = {
        "bleu_1": (1.0, 0.0, 0.0, 0.0),
        "bleu_2": (0.5, 0.5, 0.0, 0.0),
        "bleu_3": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    scores: dict[str, float] = {}
    for metric_name, weights in bleu_weights.items():
        scores[metric_name] = float(
            corpus_bleu(
                references_tokens,
                predictions_tokens,
                weights=weights,
                smoothing_function=smoothing_function,
            )
        )
    return scores


def compute_rouge_l_scores(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    """Compute corpus-average ROUGE-L F1."""
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction length mismatch: {len(references)} references vs {len(predictions)} predictions"
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_f1_scores: list[float] = []

    for reference, prediction in zip(references, predictions, strict=True):
        reference_clean = clean_text(reference)
        prediction_clean = clean_text(prediction)
        score = scorer.score(reference_clean, prediction_clean)["rougeL"].fmeasure
        rouge_l_f1_scores.append(float(score))

    rouge_l = sum(rouge_l_f1_scores) / len(rouge_l_f1_scores) if rouge_l_f1_scores else 0.0
    return {"rouge_l": rouge_l}


def build_metrics_dictionary(
    references: Sequence[str],
    predictions: Sequence[str],
    cider_score: float,
    smoothing: bool = True,
) -> dict[str, float]:
    """Combine all metric outputs into a single evaluation dictionary."""
    metrics = compute_bleu_scores(references, predictions, smoothing=smoothing)
    metrics.update(compute_rouge_l_scores(references, predictions))
    metrics["cider"] = float(cider_score)
    return metrics
