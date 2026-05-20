"""Tiny overfit sanity test for the multimodal OpenI pipeline.

This script uses two training samples, keeps the vision encoder and GPT-2
transformer frozen, and trains only the cross-attention block plus the LM head.
It is intentionally CPU-friendly and prints clear PASS/FAIL diagnostics.
"""

from __future__ import annotations

import math
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataloader import build_dataloader, preview_batch_shapes
from src.data.dataset import load_openi_dataset
from src.models.multimodal_model import build_multimodal_model


def log(message: str) -> None:
    print(message)


def _choose_split() -> str:
    for split in ("train", "val", "test"):
        try:
            if len(load_openi_dataset(split=split)) > 0:
                return split
        except Exception:
            continue
    raise RuntimeError("No usable rows were found in train/val/test processed CSVs")


def _select_two_indices(length: int) -> list[int]:
    if length <= 0:
        raise ValueError("Dataset is empty")
    if length == 1:
        return [0, 0]
    return [0, 1]


def _summarize_trainable_params(model: torch.nn.Module) -> dict[str, int]:
    summary = {"cross_attention": 0, "lm_head": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if "cross_attention" in name:
            summary["cross_attention"] += count
        elif "lm_head" in name:
            summary["lm_head"] += count
        else:
            summary["other"] += count
    return summary


def _validate_only_expected_grads(model: torch.nn.Module) -> dict[str, int]:
    counts = {"cross_attention": 0, "lm_head": 0, "other_trainable": 0, "unexpected_grad": 0}
    for name, parameter in model.named_parameters():
        has_grad = parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        if parameter.requires_grad:
            if "cross_attention" in name:
                counts["cross_attention"] += 1 if has_grad else 0
            elif "lm_head" in name:
                counts["lm_head"] += 1 if has_grad else 0
            else:
                counts["other_trainable"] += 1 if has_grad else 0
                if has_grad:
                    counts["unexpected_grad"] += 1
        elif has_grad:
            counts["unexpected_grad"] += 1
    return counts


def _device_check(batch: dict[str, object], device: torch.device) -> bool:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.device != device:
            return False
    return True


def _build_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous()

    flat_logits = shifted_logits.view(-1, vocab_size)
    flat_labels = shifted_labels.view(-1)
    flat_mask = shifted_mask.view(-1).bool()

    ignore_labels = flat_labels.clone()
    ignore_labels[~flat_mask] = pad_token_id
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)(flat_logits, ignore_labels)


def run_tiny_overfit(steps: int = 30, lr: float = 5e-5) -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    tracemalloc.start()

    split = _choose_split()
    log(f"[0] Using split: {split}")

    try:
        dataset = load_openi_dataset(split=split)
        indices = _select_two_indices(len(dataset))
        subset = Subset(dataset, indices)
        loader = build_dataloader(subset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False, seed=7)
        batch = next(iter(loader))
        log("[1] Dataset/batch setup: PASS")
        log(f"    batch_shapes={preview_batch_shapes(batch)}")
    except Exception:
        log("[1] Dataset/batch setup: FAIL")
        traceback.print_exc()
        raise

    try:
        tokenizer = dataset.tokenizer
        model = build_multimodal_model(tokenizer, local_files_only=False)
        model.to(device)
        model.train()

        for name, parameter in model.named_parameters():
            parameter.requires_grad = ("cross_attention" in name) or ("lm_head" in name)

        if not model.validate_frozen():
            raise AssertionError("Frozen parameter validation failed before training")

        trainable_summary = _summarize_trainable_params(model)
        log("[2] Model setup: PASS")
        log(f"    trainable_summary={trainable_summary}")
        log(f"    total_params={model.total_parameter_count()}")
        log(f"    trainable_params={model.trainable_parameter_count()}")
    except Exception:
        log("[2] Model setup: FAIL")
        traceback.print_exc()
        raise

    try:
        images = batch["image"].to(device=device, dtype=torch.float32)
        input_ids = batch["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long)

        if not _device_check({"image": images, "input_ids": input_ids, "attention_mask": attention_mask}, device):
            raise AssertionError("Device mismatch detected in batch tensors")

        out = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask, return_attention=True)
        logits = out["logits"]
        attention_weights = out["attention_weights"]
        patch_embeddings = out["patch_embeddings"]

        log("[3] Forward pass: PASS")
        log(f"    logits_shape={tuple(logits.shape)}")
        log(f"    attention_weights_shape={tuple(attention_weights.shape)}")
        log(f"    patch_embeddings_shape={tuple(patch_embeddings.shape)}")

        if not torch.isfinite(logits).all():
            raise FloatingPointError("NaNs or infs detected in logits")
        if not torch.isfinite(attention_weights).all():
            raise FloatingPointError("NaNs or infs detected in attention weights")
        if not torch.isfinite(patch_embeddings).all():
            raise FloatingPointError("NaNs or infs detected in patch embeddings")

        loss = _build_loss(logits, input_ids, attention_mask, tokenizer.pad_token_id)
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN/inf loss detected")
        log(f"[4] Initial loss: PASS - {float(loss.detach()):.6f}")
    except Exception:
        log("[3-4] Forward/loss setup: FAIL")
        traceback.print_exc()
        raise

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    log(f"[5] Optimizer setup: PASS - AdamW lr={lr}")
    log(f"    optimizer_param_groups={len(optimizer.param_groups)}")
    log(f"    optimizer_trainable_params={sum(p.numel() for p in optimizer.param_groups[0]['params'])}")

    initial_loss = float(loss.detach())
    final_loss = initial_loss
    grad_norm = 0.0
    last_report_time = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        out = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask, return_attention=True)
        logits = out["logits"]
        loss = _build_loss(logits, input_ids, attention_mask, tokenizer.pad_token_id)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"NaN/inf loss detected at step {step}")

        loss.backward()

        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm detected at step {step}")
        if grad_norm > 100.0:
            raise FloatingPointError(f"Exploding gradient detected at step {step}: grad_norm={grad_norm:.4f}")

        grad_counts = _validate_only_expected_grads(model)
        if grad_counts["unexpected_grad"] > 0:
            raise AssertionError(f"Unexpected gradients detected at step {step}: {grad_counts}")
        if grad_counts["cross_attention"] == 0 or grad_counts["lm_head"] == 0:
            raise AssertionError(f"Missing gradients at step {step}: {grad_counts}")

        optimizer.step()
        final_loss = float(loss.detach())

        if step == 1 or step % 5 == 0 or step == steps:
            current, peak = tracemalloc.get_traced_memory()
            log(
                f"[step {step:02d}] loss={final_loss:.6f} grad_norm={grad_norm:.4f} "
                f"mem_current_mb={current / 1024 / 1024:.2f} mem_peak_mb={peak / 1024 / 1024:.2f}"
            )

        if time.time() - last_report_time > 30:
            current, peak = tracemalloc.get_traced_memory()
            log(f"[memory] current_mb={current / 1024 / 1024:.2f} peak_mb={peak / 1024 / 1024:.2f}")
            last_report_time = time.time()

    loss_reduction = initial_loss - final_loss
    percent_reduction = (loss_reduction / initial_loss * 100.0) if initial_loss > 0 else 0.0

    if final_loss >= initial_loss:
        raise AssertionError(f"Loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}")

    log("\n[6] Overfit summary: PASS")
    log(f"    initial_loss={initial_loss:.6f}")
    log(f"    final_loss={final_loss:.6f}")
    log(f"    percent_loss_reduction={percent_reduction:.2f}%")
    log(f"    final_grad_norm={grad_norm:.4f}")
    log("    PASS: loss decreased and gradients stayed finite")


if __name__ == "__main__":
    try:
        run_tiny_overfit()
    except Exception:
        print("OVERFIT TEST FAILED")
        traceback.print_exc()
        sys.exit(2)