"""PyTorch dataset for multimodal OpenI radiology report generation."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.transforms import (
    CLIP_IMAGE_SIZE,
    build_validation_transforms,
    preview_transform,
    safe_load_image_tensor,
)
from src.data.tokenizer import (
    DEFAULT_MAX_LENGTH,
    SPECIAL_TOKENS,
    load_tokenizer,
    validate_special_tokens,
)


@dataclass(frozen=True)
class MetadataRow:
    image_path: str
    report: str


def log(message: str) -> None:
    print(f"[dataset] {message}")


def warn(message: str) -> None:
    print(f"[dataset][warning] {message}", file=sys.stderr)


def _resolve_project_root(csv_path: Path, project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()

    resolved_csv = csv_path.resolve()
    if len(resolved_csv.parents) >= 3:
        return resolved_csv.parents[2]
    return Path.cwd().resolve()


def _load_metadata(csv_path: Path) -> list[MetadataRow]:
    rows: list[MetadataRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        missing_columns = {"image_path", "report"} - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns {sorted(missing_columns)}: {csv_path}")

        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            report = (row.get("report") or "").strip()
            if not image_path or not report:
                continue
            rows.append(MetadataRow(image_path=image_path, report=report))
    return rows


class OpenIRadiologyDataset(Dataset[dict[str, Any]]):
    """Lazy multimodal dataset for OpenI chest X-ray report generation.

    Each item returns a dictionary containing:
    - image: normalized image tensor for a CLIP-style encoder
    - input_ids: GPT-2 token IDs with [START]/[END]
    - attention_mask: token attention mask
    - report: cleaned report text from CSV
    - raw_report: alias for report
    - image_path: relative image path from metadata
    - image_valid: whether the image loaded successfully
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer_dir: str | Path,
        project_root: str | Path | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        image_size: int = CLIP_IMAGE_SIZE,
        debug: bool = False,
        preview_examples: int = 0,
        placeholder_on_error: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.project_root = _resolve_project_root(self.csv_path, Path(project_root) if project_root is not None else None)
        self.tokenizer_dir = Path(tokenizer_dir).resolve()
        self.max_length = max_length
        self.image_size = image_size
        self.debug = debug
        self.preview_examples = max(0, preview_examples)
        self.placeholder_on_error = placeholder_on_error

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")

        self.records = _load_metadata(self.csv_path)
        if not self.records:
            warn(f"No usable rows were found in {self.csv_path}")

        self.tokenizer = load_tokenizer(self.tokenizer_dir)
        if self.tokenizer is None:
            raise FileNotFoundError(
                f"Tokenizer artifacts not found in {self.tokenizer_dir}. Run src.data.tokenizer first."
            )
        if not validate_special_tokens(self.tokenizer):
            raise ValueError(f"Tokenizer in {self.tokenizer_dir} is missing required special tokens")

        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = max_length

        self.image_transform = build_validation_transforms(image_size=image_size)
        self.placeholder_image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
        self._missing_image_paths: set[str] = set()
        self._corrupted_image_paths: set[str] = set()
        self._debug_preview_remaining = max(0, preview_examples)

    @classmethod
    def from_split(
        cls,
        split: str,
        data_dir: str | Path = "data/processed",
        tokenizer_dir: str | Path = "data/processed/tokenizer",
        **kwargs: Any,
    ) -> "OpenIRadiologyDataset":
        split_csv = Path(data_dir) / f"{split}.csv"
        return cls(csv_path=split_csv, tokenizer_dir=tokenizer_dir, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _tokenize_report(self, report: str) -> tuple[Tensor, Tensor]:
        text = f"{SPECIAL_TOKENS['bos_token']} {report} {SPECIAL_TOKENS['eos_token']}"
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def _log_debug_sample(self, index: int, image_path: Path, report: str, image_valid: bool) -> None:
        if not self.debug:
            return

        log(f"Sample {index}: image_valid={image_valid}, image_path={image_path.as_posix()}")
        log(f"  report_preview={report[:180]}")

    def _maybe_preview(self, index: int, image_path: Path, report: str, input_ids: Tensor) -> None:
        if self._debug_preview_remaining <= 0:
            return

        self._debug_preview_remaining -= 1
        log("Encoded sample preview")
        log(f"  index: {index}")
        log(f"  image_path: {image_path.as_posix()}")
        log(f"  report: {report[:220]}")
        log(f"  token_count: {int(input_ids.numel())}")
        log(f"  token_ids: {input_ids[:40].tolist()}{' ...' if input_ids.numel() > 40 else ''}")

    def preview_sample(self, index: int = 0) -> dict[str, Any]:
        """Return a lightweight preview for one dataset sample without mutating state."""
        item = self[index]
        return {
            "index": index,
            "image_path": item["image_path"],
            "report": item["report"],
            "image_valid": item["image_valid"],
            "input_ids_preview": item["input_ids"][:32].tolist(),
            "attention_mask_preview": item["attention_mask"][:32].tolist(),
            "image_shape": tuple(item["image"].shape),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self._resolve_image_path(record.image_path)
        image_tensor, image_valid = safe_load_image_tensor(
            image_path=image_path,
            image_transform=self.image_transform,
            placeholder=self.placeholder_image,
        )

        if not image_valid and self.placeholder_on_error:
            image_tensor = self.placeholder_image.clone()

        if not image_valid and record.image_path not in self._missing_image_paths and image_path.exists() is False:
            self._missing_image_paths.add(record.image_path)
            warn(f"Missing image referenced by metadata: {record.image_path}")
        elif not image_valid and record.image_path not in self._corrupted_image_paths and image_path.exists():
            self._corrupted_image_paths.add(record.image_path)
            warn(f"Corrupted image skipped: {record.image_path}")

        input_ids, attention_mask = self._tokenize_report(record.report)

        self._log_debug_sample(index, image_path, record.report, image_valid)
        self._maybe_preview(index, image_path, record.report, input_ids)

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "report": record.report,
            "raw_report": record.report,
            "image_path": record.image_path,
            "image_valid": image_valid,
        }


def load_openi_dataset(
    split: str,
    data_dir: str | Path = "data/processed",
    tokenizer_dir: str | Path = "data/processed/tokenizer",
    **kwargs: Any,
) -> OpenIRadiologyDataset:
    """Convenience loader for train/val/test OpenI metadata splits."""
    return OpenIRadiologyDataset.from_split(split=split, data_dir=data_dir, tokenizer_dir=tokenizer_dir, **kwargs)


def preview_dataset_image(
    image_path: str | Path,
    image_size: int = CLIP_IMAGE_SIZE,
    transform: Any | None = None,
) -> dict[str, Any]:
    """Preview a single image transform result for debugging image preprocessing."""
    preview = preview_transform(image_path=image_path, transform=transform or build_validation_transforms(image_size))
    return {
        "image_path": preview.image_path,
        "image_valid": preview.image_valid,
        "tensor_shape": preview.tensor_shape,
        "tensor_mean": preview.tensor_mean,
        "tensor_std": preview.tensor_std,
    }
