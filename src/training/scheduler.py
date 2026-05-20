"""Lightweight learning-rate scheduler helpers."""

from __future__ import annotations

import math

import torch


def _cosine_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps and warmup_steps > 0:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _linear_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps and warmup_steps > 0:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 1.0 - progress)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    schedule_type: str = "cosine",
) -> tuple[torch.optim.lr_scheduler.LambdaLR, dict[str, int]]:
    warmup_steps = int(total_steps * warmup_ratio)
    schedule_type = schedule_type.lower().strip()

    if schedule_type not in {"cosine", "linear"}:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    lr_lambda = (
        (lambda step: _cosine_lambda(step, warmup_steps, total_steps))
        if schedule_type == "cosine"
        else (lambda step: _linear_lambda(step, warmup_steps, total_steps))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {"total_steps": total_steps, "warmup_steps": warmup_steps}
