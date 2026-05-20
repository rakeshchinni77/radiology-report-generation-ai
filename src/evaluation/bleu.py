"""BLEU metric helpers for radiology report evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.metrics import compute_bleu_scores


def evaluate_bleu(references: Sequence[str], predictions: Sequence[str], smoothing: bool = True) -> dict[str, float]:
    """Return BLEU-1 through BLEU-4 scores."""
    return compute_bleu_scores(references=references, predictions=predictions, smoothing=smoothing)
