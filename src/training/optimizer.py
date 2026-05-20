"""Optimizer helpers for the multimodal training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class OptimizerSummary:
    trainable_params: int
    decay_params: int
    no_decay_params: int
    learning_rate: float
    weight_decay: float


def get_trainable_parameter_groups(model: nn.Module) -> tuple[list[dict[str, object]], OptimizerSummary]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.endswith("bias") or "ln_" in name or "layer_norm" in name or "norm" in name:
            no_decay_params.append(parameter)
            no_decay_names.append(name)
        else:
            decay_params.append(parameter)
            decay_names.append(name)

    param_groups = [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    summary = OptimizerSummary(
        trainable_params=sum(p.numel() for p in decay_params + no_decay_params),
        decay_params=sum(p.numel() for p in decay_params),
        no_decay_params=sum(p.numel() for p in no_decay_params),
        learning_rate=0.0,
        weight_decay=0.0,
    )
    return param_groups, summary


def build_adamw_optimizer(model: nn.Module, learning_rate: float = 1e-4, weight_decay: float = 0.01) -> tuple[torch.optim.Optimizer, OptimizerSummary]:
    param_groups, summary = get_trainable_parameter_groups(model)
    if not param_groups[0]["params"] and not param_groups[1]["params"]:
        raise ValueError("No trainable parameters found for AdamW optimizer")

    param_groups[0]["weight_decay"] = weight_decay
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.999))

    summary = OptimizerSummary(
        trainable_params=summary.trainable_params,
        decay_params=summary.decay_params,
        no_decay_params=summary.no_decay_params,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer, summary
