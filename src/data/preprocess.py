"""OpenI XML-to-CSV preprocessing pipeline."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from src.data.parser import (
    build_image_index,
    iter_xml_files,
    parse_xml_file,
    resolve_image_reference,
)


SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_LIMITS = {"development": 200, "training": 1000, "evaluation": 50}


def log(message: str) -> None:
    print(f"[preprocess] {message}")


def warn(message: str) -> None:
    print(f"[preprocess][warning] {message}")


def error(message: str) -> None:
    print(f"[preprocess][error] {message}", file=sys.stderr)


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_sample_limit(preset: str | None, max_samples: int | None) -> int | None:
    if max_samples is not None:
        return max_samples
    if preset is None:
        return DEFAULT_LIMITS["development"]
    return DEFAULT_LIMITS.get(preset, DEFAULT_LIMITS["development"])


def open_writer(csv_path: Path):
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=["image_path", "report"])
    writer.writeheader()
    return handle, writer


def build_record_key(image_path: str, report: str) -> str:
    return f"{image_path}\t{report}"


def split_records(records: list[tuple[str, str]], seed: int) -> dict[str, list[tuple[str, str]]]:
    shuffled_records = records[:]
    random.Random(seed).shuffle(shuffled_records)

    total = len(shuffled_records)
    train_count = int(total * SPLIT_RATIOS["train"])
    val_count = int(total * SPLIT_RATIOS["val"])
    test_count = total - train_count - val_count

    train_records = shuffled_records[:train_count]
    val_records = shuffled_records[train_count : train_count + val_count]
    test_records = shuffled_records[train_count + val_count : train_count + val_count + test_count]

    return {"train": train_records, "val": val_records, "test": test_records}


def validate_split_integrity(split_records_map: dict[str, list[tuple[str, str]]]) -> bool:
    seen_keys: set[str] = set()
    ok = True

    for split_name in ("train", "val", "test"):
        split_keys = {build_record_key(image_path, report) for image_path, report in split_records_map[split_name]}
        overlap = seen_keys.intersection(split_keys)
        if overlap:
            ok = False
            warn(f"Duplicate image/report pairs detected across splits for {split_name}: {len(overlap)}")
        seen_keys.update(split_keys)

    total_rows = sum(len(rows) for rows in split_records_map.values())
    if len(seen_keys) != total_rows:
        ok = False
        warn("Split integrity check failed: duplicate rows were found across splits")

    if total_rows == 0:
        warn("Split integrity check found no rows to validate")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess OpenI XML files into deterministic train/val/test CSV metadata."
    )
    parser.add_argument("--data-root", default="data/raw", help="Raw dataset root (default: data/raw)")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for generated CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DEFAULT_LIMITS.keys()),
        default="development",
        help="Dataset size preset: development=200, training=1000, evaluation=50",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the number of samples processed from XML files",
    )
    parser.add_argument(
        "--preview-examples",
        type=int,
        default=0,
        help="Print N before/after report cleaning previews for inspection",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = data_root / "images"
    xml_dir = data_root / "xml"

    if not images_dir.exists():
        error(f"Missing required folder: {images_dir}")
        return 1
    if not xml_dir.exists():
        error(f"Missing required folder: {xml_dir}")
        return 1

    ensure_directories(output_dir)

    sample_limit = resolve_sample_limit(args.preset, args.max_samples)
    project_root = data_root.parent.parent
    image_index = build_image_index(images_dir, project_root=project_root)
    log(f"Image index built from {len(image_index)} keys")
    log(f"Using sample limit: {sample_limit if sample_limit is not None else 'all available records'}")

    split_paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    split_handles: dict[str, object] = {}
    split_writers: dict[str, csv.DictWriter] = {}
    split_counts: Counter[str] = Counter()
    report_section_counts: Counter[str] = Counter()
    total_candidates = 0
    accepted_rows = 0
    skipped_missing_image = 0
    skipped_incomplete = 0
    skipped_corrupted = 0
    missing_impression_count = 0
    heavily_cleaned_count = 0
    preview_examples_remaining = max(0, args.preview_examples)
    duplicate_pair_count = 0
    seen_record_keys: set[str] = set()
    collected_records: list[tuple[str, str, str, str, float, str]] = []

    try:
        for xml_path in iter_xml_files(xml_dir):
            if sample_limit is not None and accepted_rows >= sample_limit:
                break

            total_candidates += 1
            parsed = parse_xml_file(xml_path)
            if parsed is None:
                skipped_incomplete += 1
                continue

            resolved_image = resolve_image_reference(parsed.image_reference, image_index)
            if resolved_image is None:
                skipped_missing_image += 1
                warn(f"Missing image for XML: {xml_path.name} -> {parsed.image_reference}")
                continue

            if not parsed.report.strip():
                skipped_incomplete += 1
                warn(f"Empty or clinically unusable report skipped: {xml_path.name}")
                continue

            if parsed.cleaning_ratio >= 0.35:
                heavily_cleaned_count += 1
                warn(
                    f"Heavily cleaned report in {xml_path.name}: removed_or_normalized={parsed.cleaning_ratio:.2f}"
                )

            if preview_examples_remaining > 0:
                preview_examples_remaining -= 1
                log("Cleaning preview")
                log(f"  XML: {xml_path.name}")
                log(f"  Before: {parsed.raw_report[:300]}")
                log(f"  After:  {parsed.report[:300]}")

            if parsed.report_section != "impression":
                missing_impression_count += 1
                warn(
                    f"Missing impression section in {xml_path.name}; using fallback section: {parsed.report_section or 'unknown'}"
                )

            relative_image = Path(resolved_image).as_posix()
            record_key = build_record_key(relative_image, parsed.report)
            if record_key in seen_record_keys:
                duplicate_pair_count += 1
                warn(f"Duplicate image/report pair skipped: {xml_path.name} -> {relative_image}")
                continue

            seen_record_keys.add(record_key)
            collected_records.append(
                (
                    relative_image,
                    parsed.report,
                    parsed.raw_report,
                    parsed.report_section,
                    parsed.cleaning_ratio,
                    xml_path.name,
                )
            )
            report_section_counts[parsed.report_section or "unknown"] += 1
            accepted_rows += 1

    except Exception as exc:  # pragma: no cover - defensive logging for CLI runs
        skipped_corrupted += 1
        error(f"Preprocessing failed: {exc}")
        return 1

    split_output = split_records([(image_path, report) for image_path, report, *_ in collected_records], args.seed)

    for split_name, csv_path in split_paths.items():
        handle, writer = open_writer(csv_path)
        split_handles[split_name] = handle
        split_writers[split_name] = writer

    try:
        for split_name, rows in split_output.items():
            for image_path, report in rows:
                split_writers[split_name].writerow({"image_path": image_path, "report": report})
                split_counts[split_name] += 1

        split_ok = validate_split_integrity(split_output)
        if not split_ok:
            warn("Split integrity validation detected issues")
    finally:
        for handle in split_handles.values():
            handle.close()

    total_split_rows = sum(split_counts.values())

    def split_ratio(count: int) -> str:
        if total_split_rows == 0:
            return "0.00%"
        return f"{(count / total_split_rows):.2%}"

    log("Preprocessing summary")
    log(f"XML candidates scanned: {total_candidates}")
    log(f"Rows written: {accepted_rows}")
    log(f"Train rows: {split_counts['train']} ({split_ratio(split_counts['train'])})")
    log(f"Val rows: {split_counts['val']} ({split_ratio(split_counts['val'])})")
    log(f"Test rows: {split_counts['test']} ({split_ratio(split_counts['test'])})")
    log(f"Impression sections extracted: {report_section_counts['impression']}")
    log(f"Fallback sections extracted: {accepted_rows - report_section_counts['impression']}")
    log(f"Missing impression sections: {missing_impression_count}")
    log(f"Heavily cleaned reports: {heavily_cleaned_count}")
    log(f"Duplicate image/report pairs skipped: {duplicate_pair_count}")
    log(f"Random seed used for splitting: {args.seed}")
    log(f"Skipped missing image references: {skipped_missing_image}")
    log(f"Skipped incomplete XML files: {skipped_incomplete}")
    log(f"Skipped corrupted XML files: {skipped_corrupted}")
    log(f"CSV outputs written to: {output_dir}")

    if accepted_rows == 0:
        warn("No valid rows were generated. Check dataset extraction and XML structure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())