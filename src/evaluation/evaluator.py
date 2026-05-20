"""Research-grade evaluator for radiology report generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.bleu import evaluate_bleu
from src.evaluation.cider import evaluate_cider
from src.evaluation.metrics import build_metrics_dictionary, clean_text
from src.evaluation.rouge import evaluate_rouge_l


DEFAULT_METRICS_PATH = Path("results/metrics.json")

METRIC_OUTPUT_KEY_MAP = {
    "bleu_1": "BLEU_1",
    "bleu_2": "BLEU_2",
    "bleu_3": "BLEU_3",
    "bleu_4": "BLEU_4",
    "cider": "CIDEr",
    "rouge_l": "ROUGE_L",
}


def log(message: str) -> None:
    print(f"[evaluator] {message}")


def format_metrics_for_output(metrics: dict[str, float]) -> dict[str, float]:
    """Convert internal lowercase metric names to rubric-ready output keys."""
    formatted: dict[str, float] = {}
    for key, value in metrics.items():
        formatted_key = METRIC_OUTPUT_KEY_MAP.get(key, key)
        formatted[formatted_key] = value
    return formatted


def _load_text_file(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
        if payload is not None:
            if isinstance(payload, list):
                return [str(item) for item in payload]
            if isinstance(payload, dict):
                for key in ("reports", "predictions", "references", "data"):
                    if key in payload and isinstance(payload[key], list):
                        return [str(item) for item in payload[key]]
            raise ValueError(f"Unsupported JSON structure in {path}")

        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, str):
                lines.append(record)
            elif isinstance(record, dict):
                for key in ("report", "prediction", "reference", "text"):
                    if key in record:
                        lines.append(str(record[key]))
                        break
                else:
                    raise ValueError(f"Could not infer text field from JSONL record in {path}")
            else:
                lines.append(str(record))
        return lines

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class EvaluationResult:
    """Container for computed metrics and sample count."""

    metrics: dict[str, float]
    num_samples: int


class Evaluator:
    """Evaluate generated radiology reports against references."""

    def __init__(self, references: list[str] | None = None, predictions: list[str] | None = None) -> None:
        self.references = references or []
        self.predictions = predictions or []

    @classmethod
    def from_files(cls, references_path: str | Path, predictions_path: str | Path) -> "Evaluator":
        """Load references and predictions from disk."""
        references = _load_text_file(references_path)
        predictions = _load_text_file(predictions_path)
        return cls(references=references, predictions=predictions)

    @classmethod
    def from_texts(cls, references: list[str], predictions: list[str]) -> "Evaluator":
        """Build an evaluator from in-memory report lists."""
        return cls(references=references, predictions=predictions)

    def validate(self) -> None:
        """Validate that the report collections are aligned."""
        if len(self.references) != len(self.predictions):
            raise ValueError(
                f"Reference/prediction length mismatch: {len(self.references)} references vs {len(self.predictions)} predictions"
            )

        if not self.references:
            raise ValueError("No reports provided for evaluation")

    def compute(self, smoothing: bool = True) -> EvaluationResult:
        """Compute all metrics for the loaded reports."""
        self.validate()

        bleu_metrics = evaluate_bleu(self.references, self.predictions, smoothing=smoothing)
        rouge_metrics = evaluate_rouge_l(self.references, self.predictions)
        cider_metrics = evaluate_cider(self.references, self.predictions)

        metrics = build_metrics_dictionary(
            references=self.references,
            predictions=self.predictions,
            cider_score=cider_metrics["cider"],
            smoothing=smoothing,
        )

        # Preserve the standalone metric helper behavior and make the source of truth explicit.
        metrics.update(bleu_metrics)
        metrics.update(rouge_metrics)
        metrics.update(cider_metrics)
        return EvaluationResult(metrics=metrics, num_samples=len(self.references))

    def save_metrics(self, output_path: str | Path = DEFAULT_METRICS_PATH, smoothing: bool = True) -> Path:
        """Compute metrics and persist them as JSON."""
        result = self.compute(smoothing=smoothing)
        formatted_metrics = format_metrics_for_output(result.metrics)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(formatted_metrics, indent=2), encoding="utf-8")

        log(f"num_samples={result.num_samples}")
        for name, value in formatted_metrics.items():
            log(f"{name}={value:.4f}")
        log(f"metrics_save_path={path}")
        return path


def synthetic_evaluation_example() -> dict[str, float]:
    """Run a tiny synthetic example for quick validation."""
    references = [
        "[START] there is no focal consolidation or pleural effusion [END]",
        "[START] mild cardiomegaly without acute abnormality [END]",
    ]
    predictions = [
        "there is no focal consolidation or pleural effusion",
        "mild cardiomegaly without acute abnormality",
    ]
    evaluator = Evaluator.from_texts(references=references, predictions=predictions)
    result = evaluator.compute()
    return result.metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for report evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate generated radiology reports.")
    parser.add_argument("--references_path", default=None, help="Path to reference reports")
    parser.add_argument("--predictions_path", default=None, help="Path to generated reports")
    parser.add_argument("--output_path", default=str(DEFAULT_METRICS_PATH), help="Path to metrics.json output")
    parser.add_argument("--demo", action="store_true", help="Run a small synthetic evaluation example")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for evaluation."""
    args = parse_args()
    try:
        if args.demo:
            metrics = synthetic_evaluation_example()
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(format_metrics_for_output(metrics), indent=2), encoding="utf-8")
            log(f"num_samples=2")
            for name, value in format_metrics_for_output(metrics).items():
                log(f"{name}={value:.4f}")
            log(f"metrics_save_path={output_path}")
            return 0

        if not args.references_path or not args.predictions_path:
            raise ValueError("Both --references_path and --predictions_path are required unless --demo is used")

        evaluator = Evaluator.from_files(args.references_path, args.predictions_path)
        evaluator.save_metrics(args.output_path)
        return 0
    except Exception as exc:
        print(f"[evaluator][error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
