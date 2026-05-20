"""PyTorch DataLoader helpers for OpenI multimodal batching."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.dataset import OpenIRadiologyDataset, load_openi_dataset


DEFAULT_BATCH_SIZE = 2
DEFAULT_BATCH_SIZE_GPU = 4
DEFAULT_NUM_WORKERS_CPU = 0
DEFAULT_NUM_WORKERS_WINDOWS = 2


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS_CPU
    pin_memory: bool = False
    seed: int = 42
    drop_last: bool = False
    debug: bool = False


def log(message: str) -> None:
    print(f"[dataloader] {message}")


def warn(message: str) -> None:
    print(f"[dataloader][warning] {message}", file=sys.stderr)


def _default_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {
            "image": torch.empty(0),
            "input_ids": torch.empty(0, dtype=torch.long),
            "attention_mask": torch.empty(0, dtype=torch.long),
            "report": [],
            "raw_report": [],
            "image_path": [],
            "image_valid": [],
        }

    images = torch.stack([item["image"] for item in batch], dim=0)
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)

    return {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "report": [item["report"] for item in batch],
        "raw_report": [item["raw_report"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "image_valid": [item["image_valid"] for item in batch],
    }


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is not None:
        return max(0, num_workers)

    if sys.platform.startswith("win"):
        return DEFAULT_NUM_WORKERS_WINDOWS
    return DEFAULT_NUM_WORKERS_CPU


def _resolve_pin_memory(pin_memory: bool | None) -> bool:
    if pin_memory is not None:
        return pin_memory
    return False


def _build_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_dataloader(
    dataset: OpenIRadiologyDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int = 42,
    drop_last: bool = False,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Build a low-RAM, CPU-friendly DataLoader for the OpenI dataset."""
    resolved_num_workers = _resolve_num_workers(num_workers)
    resolved_pin_memory = _resolve_pin_memory(pin_memory)
    resolved_collate_fn = collate_fn or _default_collate_fn
    generator = _build_generator(seed) if shuffle else None

    return DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        num_workers=resolved_num_workers,
        pin_memory=resolved_pin_memory,
        drop_last=drop_last,
        collate_fn=resolved_collate_fn,
        generator=generator,
        persistent_workers=resolved_num_workers > 0,
    )


def build_train_dataloader(
    dataset: OpenIRadiologyDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int = 42,
    drop_last: bool = False,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> DataLoader:
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def build_validation_dataloader(
    dataset: OpenIRadiologyDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int = 42,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> DataLoader:
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_test_dataloader(
    dataset: OpenIRadiologyDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int = 42,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> DataLoader:
    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=False,
        collate_fn=collate_fn,
    )


def preview_batch_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight summary of batch tensor shapes for debugging."""
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            summary[key] = tuple(value.shape)
        elif isinstance(value, list):
            summary[key] = len(value)
        else:
            summary[key] = type(value).__name__
    return summary


def log_dataloader_config(config: DataLoaderConfig) -> None:
    log(
        "Config: "
        f"batch_size={config.batch_size}, num_workers={config.num_workers}, "
        f"pin_memory={config.pin_memory}, seed={config.seed}, drop_last={config.drop_last}, "
        f"debug={config.debug}"
    )


def preview_dataloader_batch(dataloader: DataLoader) -> dict[str, Any]:
    """Pull one batch and print shape metadata for inspection."""
    batch = next(iter(dataloader))
    summary = preview_batch_shapes(batch)
    log(f"Batch preview: {summary}")
    return summary


def create_openi_dataloaders(
    split: str,
    data_dir: str | Path = "data/processed",
    tokenizer_dir: str | Path = "data/processed/tokenizer",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int = 42,
    debug: bool = False,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Convenience factory for a split-specific OpenI DataLoader."""
    dataset = load_openi_dataset(
        split=split,
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        debug=debug,
        **dataset_kwargs,
    )

    is_train_split = split.lower() == "train"
    loader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train_split,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if debug:
        log_dataloader_config(
            DataLoaderConfig(
                batch_size=batch_size,
                num_workers=_resolve_num_workers(num_workers),
                pin_memory=_resolve_pin_memory(pin_memory),
                seed=seed,
                drop_last=False,
                debug=debug,
            )
        )

    return loader
