"""Heatmap generation utilities for attention-based explainability."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_OUTPUT_DIR = Path("output")


def load_spatial_attention_tensor(file_path: str | Path) -> torch.Tensor:
    """Load a spatial attention tensor saved on disk.

    Supports either a raw tensor or a dict containing "spatial_attention".
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Attention file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = payload.get("spatial_attention")
        if tensor is None:
            tensor = payload.get("normalized_attention")
        if tensor is None:
            raise KeyError(f"File '{path}' does not contain a spatial_attention tensor")
    else:
        raise TypeError(f"Unsupported attention payload type: {type(payload)!r}")

    if tensor.dim() != 3 or tensor.shape != (1, 7, 7):
        raise ValueError(f"Expected attention tensor shape (1, 7, 7), got {tuple(tensor.shape)}")
    return tensor.detach().cpu().float()


def normalize_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    """Normalize a 7x7 tensor to a uint8 heatmap in the range 0..255."""
    if heatmap.dim() != 3:
        raise ValueError(f"Expected heatmap shape (1, 7, 7), got {tuple(heatmap.shape)}")

    array = heatmap.squeeze(0).numpy().astype(np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    denominator = max(maximum - minimum, 1e-8)
    normalized = (array - minimum) / denominator
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def resize_attention_map(heatmap_uint8: np.ndarray, image_shape: tuple[int, int], interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Upsample a 7x7 attention map to match the image resolution."""
    if heatmap_uint8.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, got shape {heatmap_uint8.shape}")

    height, width = image_shape
    resized = cv2.resize(heatmap_uint8, (width, height), interpolation=interpolation)
    return resized


def colorize_heatmap(resized_heatmap: np.ndarray) -> np.ndarray:
    """Convert a grayscale heatmap into a colored JET heatmap."""
    if resized_heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap array, got shape {resized_heatmap.shape}")
    return cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)


def save_heatmap_image(heatmap_uint8: np.ndarray, output_path: str | Path) -> Path:
    """Save a raw heatmap image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), heatmap_uint8):
        raise IOError(f"Failed to save heatmap image to {path}")
    return path
