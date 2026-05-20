"""Overlay utilities for visualizing attention on chest X-ray images."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image_bgr(image_path: str | Path) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def blend_heatmap_with_image(image_bgr: np.ndarray, colored_heatmap_bgr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend a colored heatmap with the original image."""
    if image_bgr.shape[:2] != colored_heatmap_bgr.shape[:2]:
        raise ValueError(
            f"Image and heatmap shapes must match, got {image_bgr.shape[:2]} and {colored_heatmap_bgr.shape[:2]}"
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in the range [0, 1]")

    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, colored_heatmap_bgr, alpha, 0.0)
    return overlay


def save_overlay_image(overlay_bgr: np.ndarray, output_path: str | Path) -> Path:
    """Save an overlay image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay_bgr):
        raise IOError(f"Failed to save overlay image to {path}")
    return path


def save_attention_figure(
    image_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Create and save a professional matplotlib figure showing the attention visualization."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    panels = [
        (image_rgb, "Original X-ray"),
        (heatmap_rgb, "Attention Heatmap"),
        (overlay_rgb, "Overlay"),
    ]

    for axis, (panel, title) in zip(axes, panels, strict=True):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return path
