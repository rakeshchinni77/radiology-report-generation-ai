"""CIDEr metric helpers for radiology report evaluation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from importlib.util import find_spec

from src.evaluation.metrics import clean_text


def _prepare_coco_inputs(references: Sequence[str], predictions: Sequence[str]) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Prepare COCO-style dictionaries for CIDEr computation."""
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction length mismatch: {len(references)} references vs {len(predictions)} predictions"
        )

    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for index, (reference, prediction) in enumerate(zip(references, predictions, strict=True)):
        gts[index] = [clean_text(reference)]
        res[index] = [clean_text(prediction)]
    return gts, res


def evaluate_cider(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    """Compute CIDEr if pycocoevalcap is available, otherwise fall back gracefully."""
    if find_spec("pycocoevalcap") is None:
        warnings.warn(
            "pycocoevalcap is unavailable; CIDEr will fall back to 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"cider": 0.0}

    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception as exc:  # pragma: no cover - defensive import guard
        warnings.warn(
            f"Failed to import pycocoevalcap CIDEr implementation ({exc}); falling back to 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"cider": 0.0}

    gts, res = _prepare_coco_inputs(references, predictions)
    scorer = Cider()
    cider_score, _ = scorer.compute_score(gts, res)
    return {"cider": float(cider_score)}
