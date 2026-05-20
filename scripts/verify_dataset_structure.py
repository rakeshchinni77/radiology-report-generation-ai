#!/usr/bin/env python3
"""Verify the OpenI dataset layout and cross-check XML image references.

The script is intentionally lightweight:
- scans files one by one
- parses XML files with iterparse
- does not load the full dataset into memory
- reports structure issues, counts, and mismatches
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


REQUIRED_DIRS = (
    Path("data/raw/images"),
    Path("data/raw/xml"),
    Path("data/raw/reports"),
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}
REPORT_EXTENSIONS = {".txt", ".csv", ".json"}
REFERENCE_HINTS = {"image", "img", "filename", "file", "path", "href", "source"}


def log(message: str) -> None:
    print(f"[verify_dataset] {message}")


def warn(message: str) -> None:
    print(f"[verify_dataset][warning] {message}")


def error(message: str) -> None:
    print(f"[verify_dataset][error] {message}", file=sys.stderr)


def iter_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def normalize_reference(value: str) -> str:
    value = value.strip().replace("\\", "/")
    return Path(value).name


def collect_image_names(images_dir: Path) -> tuple[set[str], set[str], int]:
    stems: set[str] = set()
    names: set[str] = set()
    count = 0
    for image_path in iter_files(images_dir, IMAGE_EXTENSIONS):
        count += 1
        names.add(image_path.name.lower())
        stems.add(image_path.stem.lower())
        stems.add(image_path.name.lower())
    return stems, names, count


def extract_references(xml_path: Path) -> set[str]:
    references: set[str] = set()
    try:
        for _, element in ET.iterparse(xml_path, events=("end",)):
            tag_name = element.tag.rsplit("}", 1)[-1].lower()

            if element.text and any(hint in tag_name for hint in REFERENCE_HINTS):
                candidate = normalize_reference(element.text)
                if candidate:
                    references.add(candidate.lower())

            for attr_name, attr_value in element.attrib.items():
                normalized_attr = attr_name.rsplit("}", 1)[-1].lower()
                if any(hint in normalized_attr for hint in REFERENCE_HINTS):
                    candidate = normalize_reference(attr_value)
                    if candidate:
                        references.add(candidate.lower())

            element.clear()
    except ET.ParseError as exc:
        warn(f"XML parse failed for {xml_path}: {exc}")
    return references


def collect_xml_references(xml_dir: Path) -> tuple[set[str], int, Counter[str]]:
    refs: set[str] = set()
    parse_errors: Counter[str] = Counter()
    count = 0

    for xml_path in iter_files(xml_dir, {".xml"}):
        count += 1
        extracted = extract_references(xml_path)
        if not extracted:
            parse_errors["no_image_references"] += 1
        refs.update(extracted)

    return refs, count, parse_errors


def validate_required_dirs(base_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for relative_dir in REQUIRED_DIRS:
        folder = base_dir / relative_dir
        if not folder.exists() or not folder.is_dir():
            missing.append(folder)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify OpenI dataset folders, counts, and XML/image reference consistency."
    )
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="Path to the raw dataset root (default: data/raw)",
    )
    args = parser.parse_args()

    base_dir = Path(args.data_root).resolve()
    missing_dirs = validate_required_dirs(base_dir.parent.parent)

    log(f"Checking dataset root: {base_dir}")
    if missing_dirs:
        for folder in missing_dirs:
            error(f"Missing required folder: {folder}")
        return 1

    images_dir = base_dir / "images"
    xml_dir = base_dir / "xml"
    reports_dir = base_dir / "reports"

    log("Scanning files without loading the full dataset into memory")
    image_keys, image_names, image_count = collect_image_names(images_dir)
    xml_refs, xml_count, xml_warnings = collect_xml_references(xml_dir)

    report_count = sum(1 for _ in iter_files(reports_dir, REPORT_EXTENSIONS))

    unmatched_refs = sorted(ref for ref in xml_refs if ref not in image_keys)
    orphan_images = sorted(
        image_name for image_name in image_names if image_name not in xml_refs
    )

    log("Dataset summary")
    log(f"Images discovered: {image_count}")
    log(f"XML files discovered: {xml_count}")
    log(f"Report files discovered: {report_count}")
    log(f"Unique image references found in XML: {len(xml_refs)}")

    if xml_warnings:
        for warning_name, warning_count in xml_warnings.items():
            warn(f"{warning_name}: {warning_count}")

    if unmatched_refs:
        warn(f"XML references without matching image files: {len(unmatched_refs)}")
        for ref in unmatched_refs[:20]:
            warn(f"  missing image reference: {ref}")

    if orphan_images:
        warn(f"Image files without matching XML references: {len(orphan_images)}")
        for image_name in orphan_images[:20]:
            warn(f"  unmatched image file: {image_name}")

    if image_count == 0 or xml_count == 0:
        warn("Dataset appears incomplete. Verify that OpenI files were extracted correctly.")

    log("Verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())