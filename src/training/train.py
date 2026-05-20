"""CLI entrypoint for multimodal OpenI training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import TrainingConfig, build_trainer


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the OpenI multimodal radiology report generator.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--tokenizer-dir", default="data/processed/tokenizer")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--drive-checkpoint-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler-type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--trainable-top-blocks", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--clip-model-name", default=None)
    parser.add_argument("--gpt2-model-name", default=None)
    parser.add_argument("--local-files-only", action="store_true")

    args = parser.parse_args()
    return TrainingConfig(
        train_split=args.train_split,
        val_split=args.val_split,
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        drive_checkpoint_dir=args.drive_checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        scheduler_type=args.scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True if args.pin_memory else None,
        log_every=1,
        local_files_only=args.local_files_only,
        clip_model_name=args.clip_model_name,
        gpt2_model_name=args.gpt2_model_name,
        num_attention_heads=args.num_attention_heads,
        trainable_top_blocks=args.trainable_top_blocks,
        use_amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )


def main() -> int:
    config = parse_args()
    trainer = build_trainer(config)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())