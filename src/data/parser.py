"""Streaming XML parsing helpers for the OpenI chest X-ray dataset."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}
IMAGE_HINTS = {"image", "img", "href", "file", "filename", "path", "source"}
REPORT_HINTS = {
    "impression",
    "findings",
    "finding",
    "summary",
    "abstract",
    "report",
    "conclusion",
    "description",
    "body",
}
REPORT_SECTION_PRIORITY = ("impression", "findings", "finding", "summary", "report", "conclusion", "abstract")
REPORT_SECTION_IGNORE = {"technique", "comparison", "history", "indication", "procedure", "exam"}
TEXT_CLEAN_PATTERN = re.compile(r"[^\w\s.,;:()\-/]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
DEID_PLACEHOLDER_PATTERN = re.compile(r"\[(?:\s*\*+[^\]]*\*+\s*|[^\]]*?)\]")
REPEATED_SYMBOL_PATTERN = re.compile(r"([_*#=@~^`|\\])\1{1,}")
PUNCTUATION_RUN_PATTERN = re.compile(r"([!?.,;:])\1{1,}")
ANONYMIZATION_TOKEN_PATTERN = re.compile(
    r"\b(?:anon(?:ymized)?|de[- ]?id(?:entified)?|patient\s+name|mrn|dob|name|id)\b",
    re.IGNORECASE,
)
IMAGE_REFERENCE_PATTERN = re.compile(r"([\w.-]+\.(?:jpg|jpeg|png|dcm))", re.IGNORECASE)
OPENI_IMAGE_PATTERN = re.compile(r"(CXR\d+_IM-\d+-\d+\.(?:jpg|jpeg|png))", re.IGNORECASE)
PARENT_IMAGE_HINTS = {"parentimage", "parentimg", "imagefile"}


@dataclass(frozen=True)
class ParsedRecord:
    image_reference: str
    raw_report: str
    report: str
    report_section: str
    cleaning_ratio: float
    xml_path: Path


def clean_text(text: str) -> str:
    """Normalize clinical report text for downstream CSV generation."""
    if not text:
        return ""

    text = text.lower()
    text = text.replace("\x00", " ")
    text = DEID_PLACEHOLDER_PATTERN.sub(" ", text)
    text = text.replace("_", " ")
    text = REPEATED_SYMBOL_PATTERN.sub(" ", text)
    text = PUNCTUATION_RUN_PATTERN.sub(r"\1", text)
    text = ANONYMIZATION_TOKEN_PATTERN.sub(" ", text)
    text = TEXT_CLEAN_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_clinical_text(text: str) -> tuple[str, float]:
    """Clean clinical text and return a lightweight cleaning ratio.

    The ratio reflects how much text was removed or normalized, which is used
    for logging heavily cleaned reports without loading all data into memory.
    """
    cleaned_text = clean_text(text)
    if not text:
        return cleaned_text, 0.0

    original_tokens = len(WHITESPACE_PATTERN.split(text.strip())) if text.strip() else 0
    cleaned_tokens = len(cleaned_text.split()) if cleaned_text else 0
    if original_tokens == 0:
        return cleaned_text, 0.0
    removed_ratio = max(0.0, 1.0 - (cleaned_tokens / original_tokens))
    return cleaned_text, removed_ratio


def normalize_reference(reference: str) -> str:
    reference = reference.strip().replace("\\", "/")
    if not reference:
        return ""
    return Path(reference).name.lower()


def _iter_xml_text_nodes(xml_path: Path) -> Iterator[tuple[str, str, dict[str, str]]]:
    for _, element in ET.iterparse(xml_path, events=("end",)):
        tag_name = element.tag.rsplit("}", 1)[-1].lower()
        text = (element.text or "").strip()
        attributes = {key.rsplit("}", 1)[-1].lower(): value for key, value in element.attrib.items()}
        yield tag_name, text, attributes
        element.clear()


def extract_image_references(xml_path: Path) -> list[str]:
    """Extract candidate image references from an OpenI XML file.
    
    Prioritizes:
    1. Actual image filenames matching OpenI naming convention (CXRxxxx_IM-xxxx.png)
    2. URL attributes in parentImage elements
    3. Text content with image file extensions
    4. Falls back to generic image references as last resort
    """
    priority_refs: set[str] = set()
    fallback_refs: set[str] = set()

    try:
        for tag_name, text, attributes in _iter_xml_text_nodes(xml_path):
            if any(hint in tag_name for hint in PARENT_IMAGE_HINTS):
                for attr_name, attr_value in attributes.items():
                    if any(hint in attr_name for hint in {"url", "file", "src", "href", "filename", "path"}):
                        normalized = normalize_reference(attr_value)
                        if normalized and "." in normalized:
                            if OPENI_IMAGE_PATTERN.match(normalized):
                                priority_refs.add(normalized)
                            elif IMAGE_REFERENCE_PATTERN.match(normalized):
                                priority_refs.add(normalized)
                            else:
                                fallback_refs.add(normalized)
                    if any(hint in attr_name for hint in {"id", "uid", "ref"}) and attr_value:
                        fallback_refs.add(attr_value.strip().lower())

            if text and IMAGE_REFERENCE_PATTERN.search(text):
                for match in IMAGE_REFERENCE_PATTERN.findall(text):
                    priority_refs.add(match.lower())

            if text and any(hint in tag_name for hint in IMAGE_HINTS):
                normalized = normalize_reference(text)
                if normalized and "." in normalized:
                    priority_refs.add(normalized)
                elif normalized:
                    fallback_refs.add(normalized)

            for attr_name, attr_value in attributes.items():
                if any(hint in attr_name for hint in {"url", "file", "src", "href", "filename", "path"}):
                    normalized = normalize_reference(attr_value)
                    if normalized and "." in normalized:
                        priority_refs.add(normalized)
    except ET.ParseError as exc:
        print(f"[parser][warning] Skipping corrupted XML file {xml_path}: {exc}", file=sys.stderr)
        return []

    result = sorted(priority_refs) if priority_refs else sorted(fallback_refs)
    return result


def _section_name(tag_name: str, attributes: dict[str, str]) -> str | None:
    label = " ".join(attributes.values()).lower()

    if any(ignore in tag_name or ignore in label for ignore in REPORT_SECTION_IGNORE):
        return None

    for section in REPORT_SECTION_PRIORITY:
        if section in tag_name or section in label:
            return section
    return None


def extract_report_text(xml_path: Path) -> tuple[str, str]:
    """Extract the most relevant radiology report text from the XML file.

    Returns a tuple of (raw_text, section_name). The section name is usually
    "impression" and falls back to a clinically meaningful section when
    impression is missing.
    """
    section_texts: dict[str, list[str]] = {section: [] for section in REPORT_SECTION_PRIORITY}
    fallback_texts: list[str] = []

    try:
        for tag_name, text, attributes in _iter_xml_text_nodes(xml_path):
            if not text:
                continue

            normalized_tag = tag_name.lower()
            candidate_section = _section_name(normalized_tag, attributes)

            if candidate_section:
                section_texts[candidate_section].append(text)
            elif any(hint in normalized_tag for hint in REPORT_HINTS) and not any(
                ignore in normalized_tag for ignore in REPORT_SECTION_IGNORE
            ):
                fallback_texts.append(text)
    except ET.ParseError as exc:
        print(f"[parser][warning] Skipping corrupted XML file {xml_path}: {exc}", file=sys.stderr)
        return "", ""

    for section in REPORT_SECTION_PRIORITY:
        if section_texts[section]:
            return " ".join(section_texts[section]), section

    return " ".join(fallback_texts), ("fallback" if fallback_texts else "")


def parse_xml_file(xml_path: Path) -> ParsedRecord | None:
    """Parse a single XML file into a normalized OpenI metadata record."""
    if not xml_path.exists() or xml_path.suffix.lower() != ".xml":
        return None

    image_references = extract_image_references(xml_path)
    raw_report_text, report_section = extract_report_text(xml_path)
    report_text, cleaning_ratio = clean_clinical_text(raw_report_text)

    if not image_references or not report_text:
        return None

    return ParsedRecord(
        image_reference=image_references[0],
        raw_report=raw_report_text,
        report=report_text,
        report_section=report_section,
        cleaning_ratio=cleaning_ratio,
        xml_path=xml_path,
    )


def iter_xml_files(xml_dir: Path) -> Iterator[Path]:
    for root, dirnames, filenames in os.walk(xml_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(".xml"):
                yield Path(root) / filename


def build_image_index(images_dir: Path, project_root: Path | None = None) -> dict[str, str]:
    """Map image names and stems to relative paths for lightweight validation."""
    image_index: dict[str, str] = {}
    project_root = project_root or images_dir.parent.parent

    for root, dirnames, filenames in os.walk(images_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            image_path = Path(root) / filename
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                relative_path = image_path.relative_to(project_root).as_posix()
            except ValueError:
                relative_path = image_path.name

            image_index.setdefault(image_path.name.lower(), relative_path)
            image_index.setdefault(image_path.stem.lower(), relative_path)
    return image_index


def resolve_image_reference(reference: str, image_index: dict[str, str]) -> str | None:
    """Resolve an XML image reference to a relative path under data/raw/images.
    
    Tries in order:
    1. Direct filename match
    2. Stem (filename without extension) match
    3. Prefix match for numeric or partial IDs
    """
    normalized = normalize_reference(reference)
    if not normalized:
        return None

    if normalized in image_index:
        return image_index[normalized]

    stem = Path(normalized).stem.lower()
    if stem in image_index:
        return image_index[stem]

    for indexed_name, indexed_path in image_index.items():
        if indexed_name.startswith(stem) or stem in indexed_name:
            return indexed_path

    return None
