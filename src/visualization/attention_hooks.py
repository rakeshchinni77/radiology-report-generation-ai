"""Attention extraction and processing helpers for multimodal inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_ATTENTION_DIR = Path("results/attention_scores")
DEFAULT_PATCH_GRID_SIZE = 7


@dataclass(frozen=True)
class AttentionArtifacts:
    """Container for attention tensors produced during inference."""

    cross_attention_weights: torch.Tensor | None
    token_attention_maps: torch.Tensor | None
    decoder_attentions: tuple[torch.Tensor, ...] | None = None
    save_path: Path | None = None


def ensure_attention_dir(output_dir: str | Path = DEFAULT_ATTENTION_DIR) -> Path:
    """Create and return the directory used to persist attention tensors."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_attention_artifacts(file_path: str | Path) -> dict[str, Any]:
    """Load a saved attention bundle from disk."""
    return torch.load(Path(file_path), map_location="cpu")


def save_attention_artifacts(
    attention_weights: torch.Tensor | None,
    token_attention_maps: torch.Tensor | None = None,
    decoder_attentions: tuple[torch.Tensor, ...] | None = None,
    output_dir: str | Path = DEFAULT_ATTENTION_DIR,
    file_name: str = "attention_weights.pt",
) -> Path:
    """Save raw attention tensors in a compact `.pt` bundle."""
    directory = ensure_attention_dir(output_dir)
    save_path = directory / file_name
    payload: dict[str, Any] = {
        "cross_attention_weights": attention_weights.detach().cpu() if attention_weights is not None else None,
        "token_attention_maps": token_attention_maps.detach().cpu() if token_attention_maps is not None else None,
        "decoder_attentions": tuple(attn.detach().cpu() for attn in decoder_attentions) if decoder_attentions is not None else None,
    }
    torch.save(payload, save_path)
    return save_path


def summarize_attention_tensor(tensor: torch.Tensor | None) -> str:
    """Return a compact tensor-shape summary for logging."""
    if tensor is None:
        return "None"
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def normalize_attention_tensor(attention_weights: torch.Tensor | None) -> torch.Tensor | None:
    """Detach and move attention tensors to CPU for persistence or visualization."""
    if attention_weights is None:
        return None
    return attention_weights.detach().cpu()


def aggregate_attention_heads(attention_weights: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Aggregate attention over heads.

    Expected input shape is (batch, heads, seq_len, patch_count).
    Returns (batch, seq_len, patch_count).
    """
    if attention_weights.dim() != 4:
        raise ValueError(f"Expected 4D attention tensor, got shape {tuple(attention_weights.shape)}")

    method = method.lower().strip()
    if method == "mean":
        return attention_weights.mean(dim=1)
    if method == "max":
        return attention_weights.max(dim=1).values
    raise ValueError(f"Unsupported head aggregation method: {method}")


def aggregate_token_attentions(
    attention_weights: torch.Tensor,
    method: str = "mean",
    token_index: int | None = None,
) -> torch.Tensor:
    """Aggregate attention over tokens.

    Accepts a tensor shaped (batch, seq_len, patch_count) and returns
    (batch, patch_count).
    """
    if attention_weights.dim() != 3:
        raise ValueError(f"Expected 3D token attention tensor, got shape {tuple(attention_weights.shape)}")

    method = method.lower().strip()
    if method == "mean":
        return attention_weights.mean(dim=1)
    if method == "single":
        index = -1 if token_index is None else token_index
        return attention_weights[:, index, :]
    raise ValueError(f"Unsupported token aggregation method: {method}")


def reshape_spatial_attention(attention_vector: torch.Tensor, patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE) -> torch.Tensor:
    """Reshape a flat patch vector into a square spatial grid."""
    if attention_vector.dim() != 2:
        raise ValueError(f"Expected 2D tensor [batch, patch_count], got shape {tuple(attention_vector.shape)}")

    expected_patch_count = patch_grid_size * patch_grid_size
    if attention_vector.shape[-1] != expected_patch_count:
        raise ValueError(
            f"Expected patch_count={expected_patch_count} for a {patch_grid_size}x{patch_grid_size} grid, "
            f"got {attention_vector.shape[-1]}"
        )

    return attention_vector.reshape(attention_vector.shape[0], patch_grid_size, patch_grid_size)


def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the [0, 1] range using min-max scaling."""
    if tensor.numel() == 0:
        return tensor

    tensor = tensor.float()
    minimum = tensor.amin(dim=tuple(range(1, tensor.dim())), keepdim=True) if tensor.dim() > 1 else tensor.min()
    maximum = tensor.amax(dim=tuple(range(1, tensor.dim())), keepdim=True) if tensor.dim() > 1 else tensor.max()
    denominator = (maximum - minimum).clamp_min(1e-8)
    return (tensor - minimum) / denominator


def process_cross_attention_maps(
    cross_attention_weights: torch.Tensor,
    head_aggregation: str = "mean",
    token_aggregation: str = "mean",
    token_index: int | None = None,
    patch_grid_size: int = DEFAULT_PATCH_GRID_SIZE,
) -> dict[str, torch.Tensor]:
    """Convert raw cross-attention into aggregated and spatial attention maps.

    Returns a dictionary containing raw, aggregated, spatial, and normalized tensors.
    """
    if cross_attention_weights.dim() != 4:
        raise ValueError(
            f"Expected raw cross-attention shape (batch, heads, seq_len, patch_count), got {tuple(cross_attention_weights.shape)}"
        )

    aggregated_heads = aggregate_attention_heads(cross_attention_weights, method=head_aggregation)
    aggregated_tokens = aggregate_token_attentions(aggregated_heads, method=token_aggregation, token_index=token_index)
    spatial_attention = reshape_spatial_attention(aggregated_tokens, patch_grid_size=patch_grid_size)
    normalized_spatial = min_max_normalize(spatial_attention)

    return {
        "raw_attention": cross_attention_weights,
        "aggregated_attention": aggregated_tokens,
        "spatial_attention": spatial_attention,
        "normalized_attention": normalized_spatial,
    }


def save_processed_attention_maps(
    processed_maps: dict[str, torch.Tensor],
    output_dir: str | Path = DEFAULT_ATTENTION_DIR,
) -> dict[str, Path]:
    """Persist aggregated and spatial attention maps to disk."""
    directory = ensure_attention_dir(output_dir)
    aggregated_path = directory / "aggregated_attention.pt"
    spatial_path = directory / "spatial_attention.pt"

    aggregated_attention = processed_maps["aggregated_attention"].detach().cpu()
    spatial_attention = processed_maps["spatial_attention"].detach().cpu()

    torch.save(aggregated_attention, aggregated_path)
    torch.save(spatial_attention, spatial_path)

    return {"aggregated_path": aggregated_path, "spatial_path": spatial_path}
