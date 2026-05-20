"""High-level explainable AI visualization pipeline for attention heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.visualization.attention_hooks import DEFAULT_ATTENTION_DIR
from src.visualization.heatmap import (
    colorize_heatmap,
    load_spatial_attention_tensor,
    normalize_heatmap,
    resize_attention_map,
    save_heatmap_image,
)
from src.visualization.overlay import (
    blend_heatmap_with_image,
    load_image_bgr,
    save_attention_figure,
    save_overlay_image,
)


DEFAULT_SPATIAL_ATTENTION_PATH = DEFAULT_ATTENTION_DIR / "spatial_attention.pt"
DEFAULT_HEATMAP_OUTPUT = Path("output/attention_heatmap.png")
DEFAULT_OVERLAY_OUTPUT = Path("output/attention_overlay.png")
DEFAULT_FIGURE_OUTPUT = Path("output/attention_figure.png")


def log(message: str) -> None:
    print(f"[visualize_attention] {message}")


def load_cross_attention_tensor(file_path: str | Path, key: str = "cross_attention_weights") -> torch.Tensor:
    """Load the raw cross-attention tensor from a saved bundle or tensor file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Attention file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = payload.get(key)
        if tensor is None:
            tensor = payload.get("spatial_attention")
        if tensor is None:
            raise KeyError(f"Missing '{key}' in attention bundle: {file_path}")
    else:
        raise TypeError(f"Unsupported attention payload type: {type(payload)!r}")

    return tensor


def process_saved_attention_bundle(
    file_path: str | Path,
    output_dir: str | Path = DEFAULT_ATTENTION_DIR,
) -> dict[str, Any]:
    """Load a saved attention file and create aggregated/spatial outputs if needed."""
    tensor = load_cross_attention_tensor(file_path)
    if tensor.dim() == 3:
        spatial_attention = tensor
        aggregated_attention = tensor.reshape(tensor.shape[0], -1)
    elif tensor.dim() == 4:
        from src.visualization.attention_hooks import process_cross_attention_maps

        processed = process_cross_attention_maps(cross_attention_weights=tensor)
        spatial_attention = processed["spatial_attention"]
        aggregated_attention = processed["aggregated_attention"]
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(tensor.shape)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = output_dir / "aggregated_attention.pt"
    spatial_path = output_dir / "spatial_attention.pt"
    torch.save(aggregated_attention.detach().cpu(), aggregated_path)
    torch.save(spatial_attention.detach().cpu(), spatial_path)
    return {"aggregated_path": aggregated_path, "spatial_path": spatial_path}


def build_attention_visualization(
    image_path: str | Path,
    attention_path: str | Path = DEFAULT_SPATIAL_ATTENTION_PATH,
    output_dir: str | Path = Path("output"),
    output_path: str | Path | None = None,
    alpha: float = 0.6,
    show: bool = False,
    save_intermediates: bool = True,
) -> dict[str, Any]:
    """Generate and save heatmap, overlay, and figure artifacts."""
    image_bgr = load_image_bgr(image_path)
    attention_tensor = load_spatial_attention_tensor(attention_path)
    log(f"loaded_image_path={Path(image_path)}")
    log(f"loaded_attention_shape={tuple(attention_tensor.shape)}")

    normalized_heatmap = normalize_heatmap(attention_tensor)
    resized_heatmap = resize_attention_map(normalized_heatmap, image_shape=image_bgr.shape[:2])
    log(f"resized_attention_shape={tuple(resized_heatmap.shape)}")

    colored_heatmap = colorize_heatmap(resized_heatmap)
    overlay_bgr = blend_heatmap_with_image(image_bgr, colored_heatmap, alpha=alpha)

    output_dir = Path(output_dir)
    overlay_target = Path(output_path) if output_path is not None else output_dir / DEFAULT_OVERLAY_OUTPUT.name
    overlay_path = save_overlay_image(overlay_bgr, overlay_target)
    heatmap_path = None
    figure_path = None
    if save_intermediates:
        heatmap_path = save_heatmap_image(resized_heatmap, output_dir / DEFAULT_HEATMAP_OUTPUT.name)
        figure_path = save_attention_figure(
            image_bgr=image_bgr,
            heatmap_bgr=colored_heatmap,
            overlay_bgr=overlay_bgr,
            output_path=output_dir / DEFAULT_FIGURE_OUTPUT.name,
            show=show,
        )
        log(f"heatmap_save_path={heatmap_path}")
        log(f"figure_save_path={figure_path}")

    log(f"overlay_save_path={overlay_path}")
    log(f"overlay_resolution={overlay_bgr.shape[1]}x{overlay_bgr.shape[0]}")

    return {
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "figure_path": figure_path,
        "overlay_resolution": overlay_bgr.shape[:2],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps over chest X-ray images.")
    parser.add_argument("--image_path", required=True, help="Path to the chest X-ray image")
    parser.add_argument(
        "--attention_path",
        default=str(DEFAULT_SPATIAL_ATTENTION_PATH),
        help="Path to the saved spatial attention tensor",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to save the final overlay PNG; defaults to output/attention_overlay.png",
    )
    parser.add_argument("--output_dir", default="output", help="Directory for saved visualization artifacts")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay blending factor")
    parser.add_argument("--show", action="store_true", help="Display the matplotlib visualization interactively")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        build_attention_visualization(
            image_path=args.image_path,
            attention_path=args.attention_path,
            output_dir=args.output_dir,
            output_path=args.output_path,
            alpha=args.alpha,
            show=args.show,
        )
        return 0
    except Exception as exc:
        print(f"[visualize_attention][error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
