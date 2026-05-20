"""ROUGE-L metric helpers for radiology report evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.metrics import compute_rouge_l_scores


def evaluate_rouge_l(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    """Return the corpus-average ROUGE-L F1 score."""
    return compute_rouge_l_scores(references=references, predictions=predictions)
