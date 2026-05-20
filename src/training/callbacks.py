"""Training callbacks, checkpoint helpers, and lightweight logging utilities."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class TrainingRunSummary:
    epochs: int
    total_steps: int
    best_val_loss: float
    initial_train_loss: float
    final_train_loss: float
    total_training_time_sec: float
    checkpoint_dir: str
    best_checkpoint_path: str | None
    latest_checkpoint_path: str | None


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return file_path


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return file_path


def move_batch_to_device(batch: dict[str, Any], device: torch.device, non_blocking: bool = False) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def get_memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024 / 1024,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }


def format_memory_snapshot(snapshot: dict[str, float]) -> str:
    return (
        f"gpu_allocated_mb={snapshot['allocated_mb']:.2f} "
        f"gpu_reserved_mb={snapshot['reserved_mb']:.2f} "
        f"gpu_peak_mb={snapshot['peak_allocated_mb']:.2f}"
    )


def save_checkpoint(state: dict[str, Any], checkpoint_path: str | Path, drive_checkpoint_path: str | Path | None = None) -> Path:
    path = Path(checkpoint_path)
    ensure_directory(path.parent)
    torch.save(state, path)

    if drive_checkpoint_path is not None:
        drive_path = Path(drive_checkpoint_path)
        ensure_directory(drive_path.parent)
        shutil.copy2(path, drive_path)

    return path


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=map_location)


def log_epoch_metrics(prefix: str, epoch: int, train_loss: float, val_loss: float, lr: float, grad_norm: float, device: torch.device) -> None:
    memory = format_memory_snapshot(get_memory_snapshot(device))
    print(
        f"[{prefix}] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
        f"lr={lr:.8f} grad_norm={grad_norm:.4f} {memory}"
    )


def log_batch_metrics(prefix: str, epoch: int, step: int, loss: float, lr: float, grad_norm: float, device: torch.device) -> None:
    memory = format_memory_snapshot(get_memory_snapshot(device))
    print(
        f"[{prefix}] epoch={epoch} batch={step} loss={loss:.6f} lr={lr:.8f} "
        f"grad_norm={grad_norm:.4f} {memory}"
    )
