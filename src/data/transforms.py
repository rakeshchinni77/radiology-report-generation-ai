"""Shared image preprocessing transforms for OpenI chest X-ray workflows."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class TransformPreview:
    image_path: str
    image_valid: bool
    tensor_shape: tuple[int, ...]
    tensor_mean: float
    tensor_std: float


def log(message: str) -> None:
    print(f"[transforms] {message}")


def warn(message: str) -> None:
    print(f"[transforms][warning] {message}", file=sys.stderr)


def build_clip_image_transform(image_size: int = CLIP_IMAGE_SIZE) -> transforms.Compose:
    """Build a CLIP-compatible image normalization pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def build_training_transforms(image_size: int = CLIP_IMAGE_SIZE, use_light_augmentation: bool = False) -> transforms.Compose:
    """Build CPU-friendly training transforms for chest X-ray images.

    The default path keeps preprocessing stable and lightweight. A small
    horizontal flip can be enabled explicitly, but heavy augmentations are
    intentionally omitted.
    """
    transforms_list: list[Any] = [
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ]

    if use_light_augmentation:
        transforms_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    return transforms.Compose(transforms_list)


def build_validation_transforms(image_size: int = CLIP_IMAGE_SIZE) -> transforms.Compose:
    """Build deterministic validation transforms."""
    return build_clip_image_transform(image_size=image_size)


def build_inference_transforms(image_size: int = CLIP_IMAGE_SIZE) -> transforms.Compose:
    """Build deterministic inference transforms compatible with CLIP ViT-B/32 style encoders."""
    return build_clip_image_transform(image_size=image_size)


def load_image_rgb(image_path: str | Path) -> Image.Image:
    """Load an image from disk and convert it to RGB."""
    path = Path(image_path)
    with Image.open(path) as image:
        return image.convert("RGB")


def safe_load_image_tensor(
    image_path: str | Path,
    image_transform: transforms.Compose | None = None,
    placeholder: Tensor | None = None,
) -> tuple[Tensor, bool]:
    """Safely load and transform an image, returning a placeholder on failure."""
    transform = image_transform or build_validation_transforms()
    placeholder_tensor = placeholder if placeholder is not None else torch.zeros(3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)

    try:
        image = load_image_rgb(image_path)
        return transform(image), True
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        warn(f"Malformed or missing image skipped: {image_path} ({exc})")
        return placeholder_tensor.clone(), False


def preview_transform(image_path: str | Path, transform: transforms.Compose | None = None) -> TransformPreview:
    """Preview the output tensor shape and statistics for a single image."""
    tensor, valid = safe_load_image_tensor(image_path=image_path, image_transform=transform)
    tensor = tensor.float()
    return TransformPreview(
        image_path=str(image_path),
        image_valid=valid,
        tensor_shape=tuple(tensor.shape),
        tensor_mean=float(tensor.mean().item()) if tensor.numel() else 0.0,
        tensor_std=float(tensor.std().item()) if tensor.numel() else 0.0,
    )
