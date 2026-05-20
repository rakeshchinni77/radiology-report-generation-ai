"""Production-style multimodal trainer for the OpenI report generator."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.dataloader import build_dataloader, preview_batch_shapes
from src.data.dataset import load_openi_dataset
from src.models.multimodal_model import MultimodalReportGenerator, build_multimodal_model
from src.training.callbacks import (
    TrainingRunSummary,
    ensure_directory,
    move_batch_to_device,
    save_checkpoint,
    save_json,
    save_yaml,
)
from src.training.optimizer import build_adamw_optimizer
from src.training.scheduler import build_scheduler
from src.data.tokenizer import load_or_build_tokenizer


@dataclass
class TrainingConfig:
    train_split: str = "train"
    val_split: str = "val"
    data_dir: str = "data/processed"
    tokenizer_dir: str = "data/processed/tokenizer"
    output_dir: str = "checkpoints"
    drive_checkpoint_dir: str | None = None
    batch_size: int = 2
    epochs: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.5
    seed: int = 42
    num_workers: int | None = None
    pin_memory: bool | None = None
    log_every: int = 1
    local_files_only: bool = False
    clip_model_name: str | None = None
    gpt2_model_name: str | None = None
    num_attention_heads: int = 8
    trainable_top_blocks: int = 2
    use_amp: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    val_fraction_if_missing: float = 0.2


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    initial_train_loss: float = float("inf")
    final_train_loss: float = float("inf")
    latest_checkpoint_path: str | None = None
    best_checkpoint_path: str | None = None
    total_training_time_sec: float = 0.0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass


def _detect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _has_rows(dataset: Dataset[Any]) -> bool:
    try:
        return len(dataset) > 0
    except Exception:
        return False


def _split_train_val_if_needed(train_dataset: Dataset[Any], val_dataset: Dataset[Any] | None, seed: int, val_fraction: float) -> tuple[Dataset[Any], Dataset[Any]]:
    if val_dataset is not None and _has_rows(val_dataset):
        return train_dataset, val_dataset

    if len(train_dataset) < 2:
        return train_dataset, train_dataset

    val_size = max(1, int(len(train_dataset) * val_fraction))
    train_size = max(1, len(train_dataset) - val_size)
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, len(train_dataset) - train_size], generator=generator)
    return train_subset, val_subset


def _prepare_frozen_trainable_state(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ("cross_attention" in name) or ("lm_head" in name)


def _assert_trainable_policy(model: MultimodalReportGenerator) -> None:
    expected = {"cross_attention", "lm_head"}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not any(token in name for token in expected):
            raise AssertionError(f"Unexpected trainable parameter: {name}")


def _compute_language_model_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous().bool()

    targets = shifted_labels.clone()
    targets[~shifted_mask] = pad_token_id
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)(shifted_logits.view(-1, vocab_size), targets.view(-1))


def _is_finite_tensor(value: torch.Tensor) -> bool:
    return torch.isfinite(value).all().item()


def _grad_stats(model: nn.Module) -> dict[str, float | int]:
    total_norm_sq = 0.0
    max_grad = 0.0
    invalid_tensors = 0
    grad_tensors = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        grad_tensors += 1
        if not torch.isfinite(grad).all():
            invalid_tensors += 1
            continue
        param_norm = float(grad.norm(2).item())
        total_norm_sq += param_norm * param_norm
        current_max = float(grad.abs().max().item())
        if current_max > max_grad:
            max_grad = current_max

    return {
        "grad_norm": math.sqrt(total_norm_sq),
        "max_grad": max_grad,
        "invalid_tensors": invalid_tensors,
        "grad_tensors": grad_tensors,
    }


class MultimodalTrainer:
    def __init__(self, model: MultimodalReportGenerator, tokenizer, train_dataset: Dataset[Any], val_dataset: Dataset[Any] | None, config: TrainingConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = _detect_device()
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.output_dir = ensure_directory(config.output_dir)
        self.drive_checkpoint_dir = Path(config.drive_checkpoint_dir) if config.drive_checkpoint_dir else None
        self.state = TrainState()

        _set_seed(config.seed)
        self.model.to(self.device)
        _prepare_frozen_trainable_state(self.model)
        if not self.model.validate_frozen():
            raise ValueError("Model frozen-state validation failed before training")
        _assert_trainable_policy(self.model)
        self._set_mode_for_stability()

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.optimizer, self.optimizer_summary = build_adamw_optimizer(
            self.model, learning_rate=config.learning_rate, weight_decay=config.weight_decay
        )
        total_steps = max(1, math.ceil(len(self.train_loader) / max(1, config.gradient_accumulation_steps)) * config.epochs)
        self.scheduler, self.scheduler_summary = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            schedule_type=config.scheduler_type,
        )

        save_yaml(self.output_dir / "training_config.yaml", asdict(config))
        save_json(self.output_dir / "training_config.json", asdict(config))

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        if not _has_rows(train_dataset):
            raise ValueError("Training dataset is empty")

        if val_dataset is None or not _has_rows(val_dataset):
            train_dataset, val_dataset = _split_train_val_if_needed(
                train_dataset,
                val_dataset,
                seed=self.config.seed,
                val_fraction=self.config.val_fraction_if_missing,
            )

        pin_memory = self.config.pin_memory if self.config.pin_memory is not None else self.device.type == "cuda"
        num_workers = self.config.num_workers if self.config.num_workers is not None else (2 if self.device.type == "cuda" else 0)

        train_loader = build_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=self.config.seed,
        )
        val_loader = build_dataloader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=self.config.seed,
        )
        print(f"[trainer] train_batch_preview={preview_batch_shapes(next(iter(train_loader)))}")
        print(f"[trainer] val_batch_preview={preview_batch_shapes(next(iter(val_loader)))}")
        return train_loader, val_loader

    def _set_mode_for_stability(self) -> None:
        self.model.train()
        self.model.vision_encoder.eval()
        self.model.text_decoder.model.transformer.eval()
        if hasattr(self.model.text_decoder.model, "ln_f"):
            self.model.text_decoder.model.ln_f.eval()
        self.model.text_decoder.lm_head.train()
        self.model.cross_attention.train()
        if getattr(self.model, "patch_projection", None) is not None:
            self.model.patch_projection.eval()

    def _autocast_context(self):
        if self.use_amp:
            return torch.amp.autocast("cuda")
        return nullcontext()

    def _forward_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._set_mode_for_stability()
        batch = move_batch_to_device(batch, self.device, non_blocking=self.device.type == "cuda")
        images = batch["image"].to(dtype=torch.float32)
        input_ids = batch["input_ids"].to(dtype=torch.long)
        attention_mask = batch["attention_mask"].to(dtype=torch.long)

        if not _is_finite_tensor(images):
            raise FloatingPointError("Non-finite input image tensor detected")

        with self._autocast_context():
            output = self.model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask, return_attention=False)
            logits = output["logits"]
            if not _is_finite_tensor(logits):
                raise FloatingPointError("Non-finite logits detected")
            loss = _compute_language_model_loss(logits, input_ids, attention_mask, self.tokenizer.pad_token_id)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss detected")
        return loss, {"logits": logits}

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        losses: list[float] = []

        for step, batch in enumerate(self.val_loader, start=1):
            batch = move_batch_to_device(batch, self.device, non_blocking=self.device.type == "cuda")
            images = batch["image"].to(dtype=torch.float32)
            input_ids = batch["input_ids"].to(dtype=torch.long)
            attention_mask = batch["attention_mask"].to(dtype=torch.long)

            with self._autocast_context():
                output = self.model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask, return_attention=False)
                logits = output["logits"]
                loss = _compute_language_model_loss(logits, input_ids, attention_mask, self.tokenizer.pad_token_id)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss detected")
            losses.append(float(loss.detach().cpu()))

            if self.config.max_val_batches is not None and step >= self.config.max_val_batches:
                break

        self.model.train()
        self._set_mode_for_stability()
        return sum(losses) / max(1, len(losses))

    def _save_state(self, epoch: int, val_loss: float, is_best: bool) -> str:
        state = {
            "epoch": epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "best_val_loss": self.state.best_val_loss,
            "val_loss": val_loss,
            "training_config": asdict(self.config),
            "optimizer_summary": asdict(self.optimizer_summary),
            "scheduler_summary": self.scheduler_summary,
        }

        latest_path = self.output_dir / "latest_model.pt"
        save_checkpoint(state, latest_path, self.drive_checkpoint_dir / "latest_model.pt" if self.drive_checkpoint_dir else None)

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            save_checkpoint(state, best_path, self.drive_checkpoint_dir / "best_model.pt" if self.drive_checkpoint_dir else None)
            self.state.best_checkpoint_path = str(best_path)

        self.state.latest_checkpoint_path = str(latest_path)
        return str(latest_path)

    def fit(self) -> TrainingRunSummary:
        self.model.train()
        start_time = time.perf_counter()
        self.state.initial_train_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            epoch_losses: list[float] = []
            grad_norm = 0.0
            max_grad = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            self._set_mode_for_stability()

            for step, batch in enumerate(self.train_loader, start=1):
                loss, _ = self._forward_loss(batch)
                raw_loss = float(loss.detach().cpu())
                if not math.isfinite(raw_loss):
                    print(f"[train][warning] Skipping invalid loss at epoch={epoch}, step={step}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.state.initial_train_loss == float("inf"):
                    self.state.initial_train_loss = raw_loss

                scaled_loss = loss / max(1, self.config.gradient_accumulation_steps)
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                epoch_losses.append(raw_loss)

                should_step = step % self.config.gradient_accumulation_steps == 0 or step == len(self.train_loader)
                if should_step:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_stats = _grad_stats(self.model)
                    if grad_stats["invalid_tensors"] > 0:
                        print(
                            f"[train][warning] Skipping optimizer step due to invalid gradients: "
                            f"invalid_tensors={grad_stats['invalid_tensors']} grad_tensors={grad_stats['grad_tensors']}"
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scaler is not None:
                            self.scaler.update()
                        continue

                    max_grad = float(grad_stats["max_grad"])
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
                    if not math.isfinite(grad_norm):
                        print(f"[train][warning] Skipping optimizer step due to non-finite grad norm at epoch={epoch}, step={step}")
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scaler is not None:
                            self.scaler.update()
                        continue

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.global_step += 1

                if self.config.log_every > 0 and step % self.config.log_every == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    memory = f"gpu_allocated_mb={torch.cuda.memory_allocated(self.device) / 1024 / 1024:.2f} gpu_reserved_mb={torch.cuda.memory_reserved(self.device) / 1024 / 1024:.2f} gpu_peak_mb={torch.cuda.max_memory_allocated(self.device) / 1024 / 1024:.2f}" if self.device.type == "cuda" and torch.cuda.is_available() else "gpu_allocated_mb=0.00 gpu_reserved_mb=0.00 gpu_peak_mb=0.00"
                    print(
                        f"[train] epoch={epoch} batch={step} loss={raw_loss:.6f} lr={current_lr:.8f} "
                        f"grad_norm={grad_norm:.4f} max_grad={max_grad:.4f} {memory}"
                    )

                if self.config.max_train_batches is not None and step >= self.config.max_train_batches:
                    break

            train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            val_loss = self._validate()
            if self.state.final_train_loss == float("inf"):
                self.state.final_train_loss = train_loss
            self.state.final_train_loss = train_loss

            is_best = val_loss < self.state.best_val_loss
            if is_best:
                self.state.best_val_loss = val_loss

            latest_path = self._save_state(epoch, val_loss, is_best=is_best)
            current_lr = self.scheduler.get_last_lr()[0]
            memory = f"gpu_allocated_mb={torch.cuda.memory_allocated(self.device) / 1024 / 1024:.2f} gpu_reserved_mb={torch.cuda.memory_reserved(self.device) / 1024 / 1024:.2f} gpu_peak_mb={torch.cuda.max_memory_allocated(self.device) / 1024 / 1024:.2f}" if self.device.type == "cuda" and torch.cuda.is_available() else "gpu_allocated_mb=0.00 gpu_reserved_mb=0.00 gpu_peak_mb=0.00"
            print(
                f"[epoch] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.8f} "
                f"grad_norm={grad_norm:.4f} max_grad={max_grad:.4f} {memory}"
            )
            print(f"[trainer] latest_checkpoint={latest_path}")
            if is_best:
                print(f"[trainer] best_checkpoint={self.state.best_checkpoint_path}")

        self.state.total_training_time_sec = time.perf_counter() - start_time
        summary = TrainingRunSummary(
            epochs=self.config.epochs,
            total_steps=self.state.global_step,
            best_val_loss=self.state.best_val_loss,
            initial_train_loss=self.state.initial_train_loss,
            final_train_loss=self.state.final_train_loss,
            total_training_time_sec=self.state.total_training_time_sec,
            checkpoint_dir=str(self.output_dir),
            best_checkpoint_path=self.state.best_checkpoint_path,
            latest_checkpoint_path=self.state.latest_checkpoint_path,
        )
        save_yaml(self.output_dir / "training_summary.yaml", asdict(summary))
        save_json(self.output_dir / "training_summary.json", asdict(summary))
        print("[trainer] training complete")
        print(f"[trainer] best_val_loss={summary.best_val_loss:.6f}")
        print(f"[trainer] total_training_time_sec={summary.total_training_time_sec:.2f}")
        return summary


def build_trainer(config: TrainingConfig) -> MultimodalTrainer:
    tokenizer = load_or_build_tokenizer(Path(config.tokenizer_dir))
    train_dataset = load_openi_dataset(split=config.train_split, data_dir=config.data_dir, tokenizer_dir=config.tokenizer_dir)
    val_dataset = None
    try:
        val_dataset = load_openi_dataset(split=config.val_split, data_dir=config.data_dir, tokenizer_dir=config.tokenizer_dir)
    except Exception:
        val_dataset = None

    model = build_multimodal_model(
        tokenizer,
        clip_model_name=config.clip_model_name,
        gpt2_model_name=config.gpt2_model_name,
        trainable_top_blocks=config.trainable_top_blocks,
        num_attention_heads=config.num_attention_heads,
        local_files_only=config.local_files_only,
    )
    return MultimodalTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset, config=config)
