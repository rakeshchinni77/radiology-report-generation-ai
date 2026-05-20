"""GPT-2 tokenizer preparation for OpenI radiology report generation."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from transformers import GPT2Tokenizer


SPECIAL_TOKENS = {
    "bos_token": "[START]",
    "eos_token": "[END]",
    "pad_token": "[PAD]",
}
DEFAULT_MAX_LENGTH = 256


@dataclass(frozen=True)
class TokenizerStats:
    vocabulary_size: int
    average_token_length: float
    max_token_length: int
    reports_processed: int


def log(message: str) -> None:
    print(f"[tokenizer] {message}")


def warn(message: str) -> None:
    print(f"[tokenizer][warning] {message}")


def error(message: str) -> None:
    print(f"[tokenizer][error] {message}", file=sys.stderr)


def iter_reports(csv_path: Path, max_reports: int | None = None) -> Iterator[str]:
    """Yield cleaned reports from the processed train CSV without loading everything into memory."""
    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            report = (row.get("report") or "").strip()
            if not report:
                continue
            yield report
            count += 1
            if max_reports is not None and count >= max_reports:
                break


def load_tokenizer(tokenizer_dir: Path) -> GPT2Tokenizer | None:
    """Load a previously saved tokenizer if it exists."""
    if not tokenizer_dir.exists():
        return None

    has_local_artifact = any(
        (tokenizer_dir / artifact_name).exists()
        for artifact_name in ("tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json")
    )
    if not has_local_artifact:
        return None

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_dir))
        tokenizer.padding_side = "right"
        return tokenizer
    except Exception as exc:  # pragma: no cover - defensive CLI logging
        warn(f"Failed to load saved tokenizer from {tokenizer_dir}: {exc}")
        return None


def validate_special_tokens(tokenizer: GPT2Tokenizer) -> bool:
    """Validate that the required GPT-2 special tokens are present and unique."""
    required_tokens = [SPECIAL_TOKENS["bos_token"], SPECIAL_TOKENS["eos_token"], SPECIAL_TOKENS["pad_token"]]
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in required_tokens]

    if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in token_ids):
        return False

    return len(set(token_ids)) == len(required_tokens)


def build_tokenizer(tokenizer_dir: Path) -> GPT2Tokenizer:
    """Create a GPT-2 tokenizer and attach OpenI special tokens."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = DEFAULT_MAX_LENGTH
    tokenizer.save_pretrained(str(tokenizer_dir))
    return tokenizer


def load_or_build_tokenizer(tokenizer_dir: Path, force_rebuild: bool = False) -> GPT2Tokenizer:
    """Load a saved tokenizer when valid, otherwise build and save a fresh one."""
    if not force_rebuild:
        tokenizer = load_tokenizer(tokenizer_dir)
        if tokenizer is not None and validate_special_tokens(tokenizer):
            return tokenizer
        if tokenizer is not None:
            warn("Saved tokenizer is missing required special tokens; rebuilding")

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(tokenizer_dir)
    if not validate_special_tokens(tokenizer):
        raise RuntimeError("Tokenizer validation failed after build")
    return tokenizer


def encode_report(tokenizer: GPT2Tokenizer, report: str, max_length: int) -> list[int]:
    """Encode a report with START/END tokens and truncation support."""
    text = f"{SPECIAL_TOKENS['bos_token']} {report} {SPECIAL_TOKENS['eos_token']}"
    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )


def prepare_tokenizer(
    train_csv_path: Path,
    tokenizer_dir: Path,
    max_length: int = DEFAULT_MAX_LENGTH,
    preview_examples: int = 3,
    max_reports: int | None = None,
    force_rebuild: bool = False,
) -> TokenizerStats:
    """Prepare a GPT-2 tokenizer and gather lightweight vocabulary statistics."""
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")

    tokenizer = load_or_build_tokenizer(tokenizer_dir, force_rebuild=force_rebuild)
    report_lengths: list[int] = []
    preview_remaining = max(0, preview_examples)

    for index, report in enumerate(iter_reports(train_csv_path, max_reports=max_reports), start=1):
        encoded_ids = encode_report(tokenizer, report, max_length=max_length)
        report_lengths.append(len(encoded_ids))

        if preview_remaining > 0:
            preview_remaining -= 1
            log("Encoded preview")
            log(f"  Report {index}: {report[:200]}")
            log(f"  Token IDs: {encoded_ids[:40]}{' ...' if len(encoded_ids) > 40 else ''}")
            log(f"  Decoded : {tokenizer.decode(encoded_ids, skip_special_tokens=False)[:200]}")

    if not report_lengths:
        warn("No reports were found in train.csv; tokenizer statistics will be empty")

    vocabulary_size = len(tokenizer)
    average_token_length = sum(report_lengths) / len(report_lengths) if report_lengths else 0.0
    max_token_length = max(report_lengths) if report_lengths else 0

    stats = TokenizerStats(
        vocabulary_size=vocabulary_size,
        average_token_length=average_token_length,
        max_token_length=max_token_length,
        reports_processed=len(report_lengths),
    )

    log("Tokenizer statistics")
    log(f"Vocabulary size: {stats.vocabulary_size}")
    log(f"Average token length: {stats.average_token_length:.2f}")
    log(f"Max token length: {stats.max_token_length}")
    log(f"Reports processed: {stats.reports_processed}")

    tokenizer.save_pretrained(str(tokenizer_dir))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and validate a GPT-2 tokenizer for OpenI radiology report generation."
    )
    parser.add_argument("--train-csv", default="data/processed/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--tokenizer-dir",
        default="data/processed/tokenizer",
        help="Directory where tokenizer artifacts are saved",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length")
    parser.add_argument("--preview-examples", type=int, default=3, help="Number of encoded preview examples")
    parser.add_argument("--max-reports", type=int, default=None, help="Optional cap on reports to process")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild tokenizer even if a valid saved tokenizer already exists",
    )
    args = parser.parse_args()

    train_csv_path = Path(args.train_csv).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()

    try:
        stats = prepare_tokenizer(
            train_csv_path=train_csv_path,
            tokenizer_dir=tokenizer_dir,
            max_length=args.max_length,
            preview_examples=args.preview_examples,
            max_reports=args.max_reports,
            force_rebuild=args.force_rebuild,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        error(str(exc))
        return 1

    if stats.reports_processed == 0:
        warn("Tokenizer was prepared, but no reports were processed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())