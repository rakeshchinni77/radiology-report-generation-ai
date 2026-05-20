"""Pre-training validation smoke test for the multimodal pipeline.

This script validates dataset/dataloader/tokenizer/model integration on a tiny
CPU-friendly batch. It prints PASS/FAIL logs for each step.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from src.data.dataloader import build_dataloader, preview_batch_shapes
from src.data.dataset import load_openi_dataset
from src.models.multimodal_model import build_multimodal_model


def log(msg: str) -> None:
    print(msg)


def _first_non_empty_split() -> str:
    for split in ("train", "val", "test"):
        try:
            if len(load_openi_dataset(split=split)) > 0:
                return split
        except Exception:
            continue
    raise RuntimeError("No usable rows were found in train/val/test processed CSVs")


def run_one_batch_validation(split: str | None = None) -> None:
    torch.set_num_threads(1)
    root = Path.cwd()
    selected_split = split or _first_non_empty_split()
    log(f"[0] Using split: {selected_split}")

    try:
        # 1) dataset loading
        ds = load_openi_dataset(split=selected_split)
        log("[1] Dataset load: PASS")
    except Exception:
        log("[1] Dataset load: FAIL")
        traceback.print_exc()
        raise

    try:
        # 2) dataloader batching (force a tiny two-sample batch when possible)
        indices = [0, 0] if len(ds) == 1 else [0, 1]
        loader = build_dataloader(Subset(ds, indices), batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        shapes = preview_batch_shapes(batch)
        log(f"[2] Dataloader batch: PASS - shapes={shapes}")
    except Exception:
        log("[2] Dataloader batch: FAIL")
        traceback.print_exc()
        raise

    try:
        # 3) tokenizer integration (dataset already uses tokenizer)
        tokenizer = ds.tokenizer
        assert tokenizer is not None
        log("[3] Tokenizer integration: PASS")
    except Exception:
        log("[3] Tokenizer integration: FAIL")
        traceback.print_exc()
        raise

    try:
        # 4) multimodal forward pass (build real multimodal model)
        model = build_multimodal_model(tokenizer, local_files_only=False)
        model.eval()
        images = batch["image"].to(torch.float32)
        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].long()

        out = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask, return_attention=True)
        log("[4] Multimodal forward: PASS")
    except Exception:
        log("[4] Multimodal forward: FAIL")
        traceback.print_exc()
        raise

    try:
        # 5) logits generation
        logits = out.get("logits")
        assert logits is not None
        assert not torch.isnan(logits).any()
        log(f"[5] Logits generation: PASS - logits_shape={tuple(logits.shape)}")
    except Exception:
        log("[5] Logits generation: FAIL")
        traceback.print_exc()
        raise

    try:
        # 6) attention weight generation
        attn = out.get("attention_weights")
        assert attn is not None
        assert not torch.isnan(attn).any()
        log(f"[6] Attention weights: PASS - shape={tuple(attn.shape)}")
    except Exception:
        log("[6] Attention weights: FAIL")
        traceback.print_exc()
        raise

    try:
        # 7) loss computation (cross-entropy over vocab)
        vocab_size = logits.shape[-1]
        labels = input_ids
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=tokenizer.pad_token_id)
        assert not torch.isnan(loss).any()
        log(f"[7] Loss computation: PASS - loss={float(loss.detach()):.6f}")
    except Exception:
        log("[7] Loss computation: FAIL")
        traceback.print_exc()
        raise

    try:
        # 8) backward pass
        model.train()
        # zero gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        loss.backward()
        log("[8] Backward pass: PASS")
    except Exception:
        log("[8] Backward pass: FAIL")
        traceback.print_exc()
        raise

    try:
        # 9) gradient flow: grads should exist only for cross-attention and LM head
        grads = {"cross_attention": 0, "lm_head": 0, "encoder": 0, "transformer": 0}

        for name, param in model.named_parameters():
            has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
            if "cross_attention" in name:
                grads["cross_attention"] += 1 if has_grad else 0
            elif "lm_head" in name or name.startswith("text_decoder.model.lm_head"):
                grads["lm_head"] += 1 if has_grad else 0
            elif "vision_encoder" in name or "vision_model" in name:
                grads["encoder"] += 1 if has_grad else 0
            elif name.startswith("text_decoder.model.transformer"):
                grads["transformer"] += 1 if has_grad else 0

        ok = grads["cross_attention"] > 0 and grads["lm_head"] > 0 and grads["encoder"] == 0 and grads["transformer"] == 0
        if ok:
            log(f"[9] Gradient flow: PASS - grads={grads}")
        else:
            log(f"[9] Gradient flow: FAIL - grads={grads}")
            raise AssertionError("Unexpected gradient flow pattern")
    except Exception:
        log("[9] Gradient flow: FAIL")
        traceback.print_exc()
        raise

    try:
        # 10) frozen parameter validation
        frozen_ok = model.validate_frozen()
        assert frozen_ok
        log("[10] Frozen parameter validation: PASS")
    except Exception:
        log("[10] Frozen parameter validation: FAIL")
        traceback.print_exc()
        raise

    log("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        run_one_batch_validation()
    except Exception as exc:  # pragma: no cover - test runner
        print("One or more checks failed; see traceback above.")
        sys.exit(2)