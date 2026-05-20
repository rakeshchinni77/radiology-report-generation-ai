"""Production-grade inference entrypoint for radiology report generation."""

from __future__ import annotations

import argparse
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import load_tokenizer, validate_special_tokens
from src.data.transforms import CLIP_IMAGE_SIZE, build_inference_transforms, load_image_rgb
from src.inference.beam_search import generate_beam_report, generate_greedy_report
from src.models.multimodal_model import build_multimodal_model
from src.visualization.attention_hooks import (
    save_attention_artifacts,
    save_processed_attention_maps,
    summarize_attention_tensor,
    process_cross_attention_maps,
)
from src.training.callbacks import load_checkpoint


DEFAULT_CHECKPOINT = Path("checkpoints/best_model.pt")
DEFAULT_OUTPUT_PATH = Path("output/generated_report.txt")
DEFAULT_TOKENIZER_CANDIDATES = [Path("checkpoints/tokenizer"), Path("data/processed/tokenizer")]


def log(message: str) -> None:
    print(f"[predict] {message}")


def warn(message: str) -> None:
    print(f"[predict][warning] {message}", file=sys.stderr)


def error(message: str) -> None:
    print(f"[predict][error] {message}", file=sys.stderr)


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate = torch.device(device_arg)
    if candidate.type == "cuda" and not torch.cuda.is_available():
        warn(f"Requested CUDA device '{device_arg}' is unavailable; falling back to CPU")
        return torch.device("cpu")
    return candidate


def _resolve_checkpoint_path(checkpoint_arg: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_arg)
    if checkpoint_path.is_dir():
        for candidate in (checkpoint_path / "best_model.pt", checkpoint_path / "latest_model.pt"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No checkpoint file found in directory: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_tokenizer_from_candidates(candidates: list[Path]) -> Any:
    for tokenizer_dir in candidates:
        tokenizer = load_tokenizer(tokenizer_dir)
        if tokenizer is not None and validate_special_tokens(tokenizer):
            tokenizer.padding_side = "right"
            return tokenizer
    raise FileNotFoundError(
        "Tokenizer artifacts not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_model(
    tokenizer: Any,
    checkpoint_path: Path,
    device: torch.device,
    local_files_only: bool = False,
) -> torch.nn.Module:
    model = build_multimodal_model(tokenizer, local_files_only=local_files_only)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if isinstance(state_dict, dict) and state_dict:
        if all(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            warn(f"Strict checkpoint loading failed, retrying with strict=False: {exc}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                warn(f"Missing checkpoint keys: {missing_keys[:10]}{' ...' if len(missing_keys) > 10 else ''}")
            if unexpected_keys:
                warn(f"Unexpected checkpoint keys: {unexpected_keys[:10]}{' ...' if len(unexpected_keys) > 10 else ''}")

    model.to(device)
    model.eval()
    return model


def _load_image_tensor(image_path: str | Path, device: torch.device) -> torch.Tensor:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    transform = build_inference_transforms(image_size=CLIP_IMAGE_SIZE)
    try:
        image = load_image_rgb(path)
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(device=device, dtype=torch.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Failed to load image '{path}': {exc}") from exc


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _resolve_output_path(output_path: str | Path | None) -> Path:
    if output_path is None:
        return DEFAULT_OUTPUT_PATH

    path = Path(output_path)
    if path.suffix:
        return path
    return path / DEFAULT_OUTPUT_PATH.name


def run_inference(
    image_path: str | Path,
    checkpoint: str | Path,
    max_length: int,
    device_arg: str | None,
    output_path: str | Path | None,
    decode_strategy: str = "beam",
    beam_size: int = 3,
    temperature: float = 1.0,
    top_k: int = 0,
    return_attentions: bool = False,
) -> str:
    device = _resolve_device(device_arg)
    checkpoint_path = _resolve_checkpoint_path(checkpoint)

    log(f"device={device}")
    log(f"checkpoint={checkpoint_path}")

    tokenizer_candidates = [checkpoint_path.parent / "tokenizer", *DEFAULT_TOKENIZER_CANDIDATES]
    tokenizer = _load_tokenizer_from_candidates(tokenizer_candidates)
    model = _load_model(tokenizer, checkpoint_path, device=device)

    image_tensor = _load_image_tensor(image_path, device=device)
    log(f"image_tensor_shape={tuple(image_tensor.shape)}")
    log(f"decode_strategy={decode_strategy}")
    log(f"beam_size={beam_size}")
    log(f"attention_enabled={return_attentions}")

    start_time = time.perf_counter()
    if decode_strategy == "greedy":
        result = generate_greedy_report(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            max_length=max_length,
            device=device,
            temperature=temperature,
            top_k=top_k,
        )
    else:
        result = generate_beam_report(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            max_length=max_length,
            device=device,
            beam_size=beam_size,
            temperature=temperature,
            top_k=top_k,
        )
    inference_time = time.perf_counter() - start_time

    attention_save_path: Path | None = None
    if return_attentions:
        with torch.inference_mode():
            token_ids = torch.tensor([result.token_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(token_ids)
            attention_out = model(
                pixel_values=image_tensor,
                input_ids=token_ids,
                attention_mask=attention_mask,
                return_attention=True,
                return_attentions=True,
            )
            attention_weights = attention_out.get("attention_weights")
            token_attention_maps = attention_out.get("token_attention_maps")
            decoder_attentions = attention_out.get("decoder_attentions")
            attention_save_path = save_attention_artifacts(
                attention_weights=attention_weights,
                token_attention_maps=token_attention_maps,
                decoder_attentions=decoder_attentions,
                output_dir=Path("results/attention_scores"),
            )
            log(f"attention_tensor_shapes={summarize_attention_tensor(attention_weights)}")
            if attention_weights is not None:
                processed_maps = process_cross_attention_maps(
                    cross_attention_weights=attention_weights,
                    head_aggregation="mean",
                    token_aggregation="mean",
                )
                processed_paths = save_processed_attention_maps(processed_maps, output_dir=Path("results/attention_scores"))
                log(f"aggregated_attention_shape={tuple(processed_maps['aggregated_attention'].shape)}")
                log(f"normalized_attention_shape={tuple(processed_maps['normalized_attention'].shape)}")
                log(f"aggregated_attention_save_path={processed_paths['aggregated_path']}")
                log(f"spatial_attention_save_path={processed_paths['spatial_path']}")
            if decoder_attentions is not None:
                log(f"decoder_attention_layers={len(decoder_attentions)}")
                log(f"decoder_attention_shapes={[tuple(attn.shape) for attn in decoder_attentions]}")
            log(f"attention_save_path={attention_save_path}")

    report = result.text
    output_file = _resolve_output_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report + "\n", encoding="utf-8")

    print(report)
    log(f"generated_token_count={max(0, len(result.token_ids) - 1)}")
    log(f"stop_reason={result.stop_reason}")
    log(f"beam_score={result.score:.4f}")
    log(f"normalized_score={result.normalized_score:.4f}")
    if return_attentions:
        log("attention_enabled=True")
    log(f"inference_time_sec={inference_time:.4f}")
    log(f"output_path={output_file}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a radiology report from a chest X-ray image.")
    parser.add_argument("--image_path", required=True, help="Path to the chest X-ray image")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum generated token length")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument("--output_path", default=None, help="Output file path or directory")
    parser.add_argument("--decode_strategy", choices=["greedy", "beam"], default="beam", help="Decoding strategy to use")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (1.0 keeps greedy behavior when top_k=0)")
    parser.add_argument("--top_k", type=int, default=0, help="Apply top-k filtering before token selection")
    parser.add_argument("--return_attentions", action="store_true", help="Return and save attention tensors during inference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_inference(
            image_path=args.image_path,
            checkpoint=args.checkpoint,
            max_length=args.max_length,
            device_arg=args.device,
            output_path=args.output_path,
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size,
            temperature=args.temperature,
            top_k=args.top_k,
            return_attentions=args.return_attentions,
        )
        return 0
    except Exception as exc:
        error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())