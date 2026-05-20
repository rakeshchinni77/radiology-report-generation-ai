"""Production-style evaluation CLI for multimodal radiology report generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import Evaluator
from src.evaluation.evaluator import format_metrics_for_output
from src.evaluation.metrics import clean_text


DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "evaluation_summary.txt"

DEMO_REFERENCES = [
    "The cardiac silhouette is mildly enlarged. No focal consolidation, pleural effusion, or pneumothorax. Low lung volumes with minimal bibasilar atelectatic change.",
    "Mild cardiomegaly is present. No focal air-space consolidation or pleural effusion. Bibasilar atelectasis is noted with low lung volumes.",
    "The chest radiograph shows low lung volumes and mild bibasilar atelectasis. No focal consolidation or pleural effusion. The heart is mildly enlarged.",
]


def log(message: str) -> None:
    """Print a consistent evaluation log line."""
    print(f"[evaluate] {message}")


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    """Resolve and create the output directory."""
    path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_csv_reports(test_file: str | Path) -> list[str]:
    """Load reference reports from a lightweight CSV file."""
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Invalid CSV format: missing header row in {path}")

            report_field = None
            for candidate in ("report", "reference", "text"):
                if candidate in reader.fieldnames:
                    report_field = candidate
                    break

            if report_field is None:
                raise ValueError(
                    f"Invalid CSV format in {path}: expected one of the columns 'report', 'reference', or 'text'"
                )

            reports: list[str] = []
            for row in reader:
                value = (row.get(report_field) or "").strip()
                if value:
                    reports.append(value)
            return reports
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV format in {path}: {exc}") from exc


def _generate_placeholder_prediction(reference_report: str) -> str:
    """Generate a lightweight, deterministic placeholder prediction."""
    text = clean_text(reference_report)
    if not text:
        return "no focal consolidation or pleural effusion. low lung volumes are present."

    findings: list[str] = []

    if "cardiomegaly" in text:
        findings.append("mild cardiomegaly")
    if "consolidation" in text:
        if "no focal consolidation" in text or "without focal consolidation" in text or "no consolidation" in text:
            findings.append("no focal consolidation")
        else:
            findings.append("focal consolidation")
    if "pleural effusion" in text:
        if "no pleural effusion" in text or "without pleural effusion" in text:
            findings.append("no pleural effusion")
        else:
            findings.append("pleural effusion")
    if "low lung volumes" in text:
        findings.append("low lung volumes")
    if "atelectasis" in text:
        findings.append("bibasilar atelectasis")

    if not findings:
        tokens = text.split()
        return " ".join(tokens[:16]) if tokens else "no acute cardiopulmonary abnormality"

    return ". ".join(findings) + "."


def _build_demo_predictions(references: list[str]) -> list[str]:
    """Create a tiny demo prediction set for empty or demo runs."""
    if not references:
        return [
            "mild cardiomegaly. no focal consolidation. low lung volumes.",
            "no focal consolidation or pleural effusion. bibasilar atelectasis.",
            "low lung volumes with bibasilar atelectasis. mild cardiomegaly.",
        ]
    return [_generate_placeholder_prediction(report) for report in references]


def _load_evaluation_inputs(
    test_file: str | Path | None,
    predictions_file: str | Path | None,
    references_file: str | Path | None,
    demo: bool,
) -> tuple[list[str], list[str], str]:
    """Load or synthesize evaluation inputs based on the requested mode."""
    if demo:
        predictions = _build_demo_predictions(DEMO_REFERENCES)
        return DEMO_REFERENCES, predictions, "demo"

    if predictions_file is not None or references_file is not None:
        if predictions_file is None or references_file is None:
            raise ValueError("Both --predictions_file and --references_file are required for direct evaluation")

        predictions_path = Path(predictions_file)
        references_path = Path(references_file)
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        if not references_path.exists():
            raise FileNotFoundError(f"References file not found: {references_path}")

        evaluator = Evaluator.from_files(references_path, predictions_path)
        return evaluator.references, evaluator.predictions, "direct-files"

    if test_file is None:
        raise ValueError("Provide --test_file, or both --predictions_file and --references_file, or use --demo")

    references = _load_csv_reports(test_file)
    if not references:
        log(f"test_file={test_file} contained no report rows; using built-in demo evaluation inputs")
        predictions = _build_demo_predictions([])
        return DEMO_REFERENCES, predictions, "demo-fallback"

    predictions = [_generate_placeholder_prediction(report) for report in references]
    return references, predictions, "test-file-placeholder"


def _format_summary(metrics: dict[str, float], num_samples: int, mode: str, duration_seconds: float) -> str:
    """Create a human-readable evaluation summary."""
    lines = [
        "Radiology Report Evaluation Summary",
        f"Mode: {mode}",
        f"Samples: {num_samples}",
        f"Evaluation duration: {duration_seconds:.2f} seconds",
        "",
        f"BLEU-1: {metrics['bleu_1']:.4f}",
        f"BLEU-2: {metrics['bleu_2']:.4f}",
        f"BLEU-3: {metrics['bleu_3']:.4f}",
        f"BLEU-4: {metrics['bleu_4']:.4f}",
        f"ROUGE-L: {metrics['rouge_l']:.4f}",
        f"CIDEr: {metrics['cider']:.4f}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate radiology report generation outputs.")
    parser.add_argument("--test_file", default=None, help="CSV file containing test reports")
    parser.add_argument("--predictions_file", default=None, help="Generated report text file")
    parser.add_argument("--references_file", default=None, help="Reference report text file")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for metrics and summary outputs")
    parser.add_argument("--demo", action="store_true", help="Run the built-in lightweight demo evaluation")
    return parser.parse_args()


def main() -> int:
    """Run the evaluation pipeline and save metrics outputs."""
    args = parse_args()
    start_time = time.perf_counter()

    try:
        output_dir = _resolve_output_dir(args.output_dir)
        references, predictions, mode = _load_evaluation_inputs(
            test_file=args.test_file,
            predictions_file=args.predictions_file,
            references_file=args.references_file,
            demo=args.demo,
        )

        evaluator = Evaluator.from_texts(references=references, predictions=predictions)
        result = evaluator.compute()

        metrics_path = output_dir / DEFAULT_METRICS_PATH.name
        summary_path = output_dir / DEFAULT_SUMMARY_PATH.name

        formatted_metrics = format_metrics_for_output(result.metrics)
        metrics_path.write_text(json.dumps(formatted_metrics, indent=2), encoding="utf-8")

        duration_seconds = time.perf_counter() - start_time
        summary_text = _format_summary(
            metrics=result.metrics,
            num_samples=result.num_samples,
            mode=mode,
            duration_seconds=duration_seconds,
        )
        summary_path.write_text(summary_text, encoding="utf-8")

        log(f"num_samples={result.num_samples}")
        for metric_name in ("BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "ROUGE_L", "CIDEr"):
            log(f"{metric_name}={formatted_metrics[metric_name]:.4f}")
        log(f"metrics_save_path={metrics_path}")
        log(f"summary_save_path={summary_path}")
        log(f"evaluation_duration_sec={duration_seconds:.2f}")
        return 0
    except Exception as exc:
        print(f"[evaluate][error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
