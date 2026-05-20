"""Cross-attention fusion between GPT-2 hidden states and CLIP image patches."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_MODEL_DIM = 768
DEFAULT_NUM_HEADS = 8
DEFAULT_DROPOUT = 0.0


@dataclass(frozen=True)
class AttentionConfig:
    hidden_size: int = DEFAULT_MODEL_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    dropout: float = DEFAULT_DROPOUT
    batch_first: bool = True
    debug: bool = False


@dataclass(frozen=True)
class AttentionPreview:
    hidden_states_shape: tuple[int, ...]
    encoder_hidden_states_shape: tuple[int, ...]
    fused_hidden_states_shape: tuple[int, ...]
    attention_weights_shape: tuple[int, ...]
    device: str
    dtype: str


def log(message: str) -> None:
    print(f"[attention] {message}")


def warn(message: str) -> None:
    print(f"[attention][warning] {message}", file=sys.stderr)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def count_total_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def validate_attention_trainability(module: nn.Module) -> bool:
    """Validate that only cross-attention parameters are trainable.

    This module intentionally has no extra trainable layers beyond MultiheadAttention.
    """
    allowed_trainable_prefixes = {"cross_attention."}

    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            if not any(name.startswith(prefix) for prefix in allowed_trainable_prefixes):
                return False

    return True


class OpenICrossAttentionFusion(nn.Module):
    """Batch-first cross-attention fusion for GPT-2 and CLIP features.

    Language hidden states act as queries.
    Image patch embeddings act as keys and values.
    The module preserves the patch sequence for future heatmap extraction.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_MODEL_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        batch_first: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.config = AttentionConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            debug=debug,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.cross_attention.dropout = 0.0
        self.debug = debug
        self.eval()

    def train(self, mode: bool = True) -> "OpenICrossAttentionFusion":
        """Keep deterministic CPU-friendly behavior by default.

        The fusion module is intended to be used in a stable, low-variance manner.
        """
        super().train(False)
        self.cross_attention.train(False)
        return self

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        need_weights: bool = True,
    ) -> dict[str, Tensor]:
        """Fuse language states with image patch embeddings.

        Args:
            hidden_states: GPT-2 hidden states shaped [batch, seq, hidden].
            encoder_hidden_states: CLIP patch embeddings shaped [batch, patches, hidden].
            attention_mask: Optional query-side mask shaped [batch, seq].
            encoder_attention_mask: Optional key-side mask shaped [batch, patches].
            need_weights: Whether to return attention weights for visualization.
        """
        self._validate_inputs(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask)

        key_padding_mask = self._mask_to_key_padding_mask(encoder_attention_mask)

        attn_output, attn_weights = self.cross_attention(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        fused_hidden_states = hidden_states + attn_output
        if attention_mask is not None:
            fused_hidden_states = self._apply_query_mask(fused_hidden_states, attention_mask)

        if self.debug:
            log(
                "Forward shapes: "
                f"hidden_states={tuple(hidden_states.shape)}, "
                f"encoder_hidden_states={tuple(encoder_hidden_states.shape)}, "
                f"fused_hidden_states={tuple(fused_hidden_states.shape)}, "
                f"attention_weights={tuple(attn_weights.shape) if attn_weights is not None else None}"
            )

        return {
            "fused_hidden_states": fused_hidden_states,
            "attention_weights": attn_weights,
        }

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> None:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        if encoder_hidden_states.dim() != 3:
            raise ValueError(
                f"encoder_hidden_states must be [batch, patches, hidden], got {tuple(encoder_hidden_states.shape)}"
            )
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("Batch size must match between hidden_states and encoder_hidden_states")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states last dim must equal hidden_size={self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if encoder_hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "encoder_hidden_states last dim must match hidden_size="
                f"{self.config.hidden_size}, got {encoder_hidden_states.shape[-1]}"
            )
        if attention_mask is not None and attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match hidden_states batch and sequence dimensions")
        if encoder_attention_mask is not None and encoder_attention_mask.shape[:2] != encoder_hidden_states.shape[:2]:
            raise ValueError("encoder_attention_mask must match encoder_hidden_states batch and sequence dimensions")

    def _mask_to_key_padding_mask(self, encoder_attention_mask: Tensor | None) -> Tensor | None:
        if encoder_attention_mask is None:
            return None
        if encoder_attention_mask.dtype == torch.bool:
            return ~encoder_attention_mask
        return encoder_attention_mask == 0

    def _apply_query_mask(self, fused_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask != 0
        return fused_hidden_states * attention_mask.unsqueeze(-1).to(dtype=fused_hidden_states.dtype)

    def validate_frozen_state(self) -> bool:
        """Validate that only the cross-attention module is trainable."""
        return validate_attention_trainability(self)

    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self)

    def total_parameter_count(self) -> int:
        return count_total_parameters(self)

    def log_attention_config(self) -> None:
        log(
            "Config: "
            f"hidden_size={self.config.hidden_size}, "
            f"num_heads={self.config.num_heads}, "
            f"dropout={self.config.dropout}, "
            f"batch_first={self.config.batch_first}, "
            f"debug={self.config.debug}"
        )
        log(f"Total parameters: {self.total_parameter_count()}")
        log(f"Trainable parameters: {self.trainable_parameter_count()}")
        log(f"Frozen validation: {self.validate_frozen_state()}")

    def preview_attention_shapes(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> AttentionPreview:
        with torch.no_grad():
            outputs = self.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                need_weights=True,
            )

        attention_weights = outputs["attention_weights"]
        if attention_weights is None:
            raise RuntimeError("Attention weights were not returned")

        return AttentionPreview(
            hidden_states_shape=tuple(hidden_states.shape),
            encoder_hidden_states_shape=tuple(encoder_hidden_states.shape),
            fused_hidden_states_shape=tuple(outputs["fused_hidden_states"].shape),
            attention_weights_shape=tuple(attention_weights.shape),
            device=str(hidden_states.device),
            dtype=str(outputs["fused_hidden_states"].dtype),
        )


def build_cross_attention_fusion(
    hidden_size: int = DEFAULT_MODEL_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    dropout: float = DEFAULT_DROPOUT,
    debug: bool = False,
) -> OpenICrossAttentionFusion:
    """Factory for the multimodal cross-attention fusion module."""
    module = OpenICrossAttentionFusion(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
        debug=debug,
    )
    module.log_attention_config()
    return module


def preview_cross_attention_shapes(
    fusion: OpenICrossAttentionFusion,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    encoder_attention_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Return a compact shape summary for debugging and notebook previews."""
    preview = fusion.preview_attention_shapes(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        encoder_attention_mask=encoder_attention_mask,
    )
    summary = {
        "hidden_states_shape": preview.hidden_states_shape,
        "encoder_hidden_states_shape": preview.encoder_hidden_states_shape,
        "fused_hidden_states_shape": preview.fused_hidden_states_shape,
        "attention_weights_shape": preview.attention_weights_shape,
        "device": preview.device,
        "dtype": preview.dtype,
        "trainable_parameters": fusion.trainable_parameter_count(),
        "frozen": fusion.validate_frozen_state(),
    }
    log(f"Preview: {summary}")
    return summary
