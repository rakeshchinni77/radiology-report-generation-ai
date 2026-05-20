"""Frozen CLIP vision encoder for multimodal radiology report generation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import CLIPVisionModel


DEFAULT_CLIP_VISION_MODEL = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = DEFAULT_CLIP_VISION_MODEL
    output_hidden_states: bool = False
    output_attentions: bool = False
    local_files_only: bool = False


@dataclass(frozen=True)
class EncoderPreview:
    image_shape: tuple[int, ...]
    patch_embeddings_shape: tuple[int, ...]
    pooled_embeddings_shape: tuple[int, ...] | None
    device: str
    dtype: str


def log(message: str) -> None:
    print(f"[encoder] {message}")


def warn(message: str) -> None:
    print(f"[encoder][warning] {message}", file=sys.stderr)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def validate_encoder_frozen(module: nn.Module) -> bool:
    return all(not parameter.requires_grad for parameter in module.parameters())


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class OpenIClipVisionEncoder(nn.Module):
    """Frozen CLIP Vision encoder that returns patch-level embeddings.

    The module is configured for inference-style usage during training:
    - all parameters are frozen
    - forward pass uses ``torch.inference_mode``
    - outputs preserve spatial patch embeddings for future attention maps
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_VISION_MODEL,
        local_files_only: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        self.config = EncoderConfig(
            model_name=model_name,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            local_files_only=local_files_only,
        )
        self.vision_model = CLIPVisionModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.vision_model.config.output_hidden_states = output_hidden_states
        self.vision_model.config.output_attentions = output_attentions
        _freeze_module(self.vision_model)
        self.vision_model.eval()

        if not validate_encoder_frozen(self.vision_model):
            raise RuntimeError("CLIP vision encoder must be fully frozen")

    def train(self, mode: bool = True) -> "OpenIClipVisionEncoder":
        """Keep the encoder in eval mode to preserve frozen inference-style behavior."""
        super().train(False)
        self.vision_model.eval()
        return self

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor | None]:
        """Encode a batch of CLIP-normalized image tensors.

        Args:
            pixel_values: Tensor shaped [batch, 3, 224, 224]

        Returns:
            A dictionary with patch embeddings and optional pooled embeddings.
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected 4D tensor [batch, 3, H, W], got shape {tuple(pixel_values.shape)}")

        if pixel_values.shape[1] != 3:
            raise ValueError(f"Expected 3 image channels, got {pixel_values.shape[1]}")

        with torch.inference_mode():
            outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else None

        # CLIPVisionModel returns [CLS] + patch tokens. We preserve both the full
        # sequence and the patch-only embeddings for attention visualization.
        cls_embedding = last_hidden_state[:, :1, :]
        patch_embeddings = last_hidden_state[:, 1:, :]

        return {
            "patch_embeddings": patch_embeddings,
            "cls_embedding": cls_embedding,
            "pooled_embedding": pooled_output,
            "last_hidden_state": last_hidden_state,
        }

    def encode(self, pixel_values: Tensor) -> dict[str, Tensor | None]:
        """Alias for forward for encoder-centric call sites."""
        return self.forward(pixel_values)

    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self.vision_model)

    def validate_frozen(self) -> bool:
        return validate_encoder_frozen(self.vision_model)

    def log_encoder_config(self) -> None:
        log(
            "Config: "
            f"model_name={self.config.model_name}, "
            f"output_hidden_states={self.config.output_hidden_states}, "
            f"output_attentions={self.config.output_attentions}, "
            f"local_files_only={self.config.local_files_only}"
        )
        log(f"Trainable parameters: {self.trainable_parameter_count()}")
        log(f"Frozen validation: {self.validate_frozen()}")

    def preview_embeddings(self, pixel_values: Tensor) -> EncoderPreview:
        """Run a small forward pass preview and summarize embedding shapes."""
        outputs = self.forward(pixel_values)
        patch_embeddings = outputs["patch_embeddings"]
        pooled_embedding = outputs["pooled_embedding"]

        return EncoderPreview(
            image_shape=tuple(pixel_values.shape),
            patch_embeddings_shape=tuple(patch_embeddings.shape),
            pooled_embeddings_shape=tuple(pooled_embedding.shape) if pooled_embedding is not None else None,
            device=str(pixel_values.device),
            dtype=str(pixel_values.dtype),
        )


def build_clip_vision_encoder(
    model_name: str = DEFAULT_CLIP_VISION_MODEL,
    local_files_only: bool = False,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
) -> OpenIClipVisionEncoder:
    """Factory for the frozen CLIP vision encoder."""
    encoder = OpenIClipVisionEncoder(
        model_name=model_name,
        local_files_only=local_files_only,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    encoder.log_encoder_config()
    return encoder


def preview_encoder_embeddings(encoder: OpenIClipVisionEncoder, pixel_values: Tensor) -> dict[str, Any]:
    """Return a concise preview of encoder outputs for debugging."""
    preview = encoder.preview_embeddings(pixel_values)
    summary = {
        "image_shape": preview.image_shape,
        "patch_embeddings_shape": preview.patch_embeddings_shape,
        "pooled_embeddings_shape": preview.pooled_embeddings_shape,
        "device": preview.device,
        "dtype": preview.dtype,
        "trainable_parameters": encoder.trainable_parameter_count(),
        "frozen": encoder.validate_frozen(),
    }
    log(f"Embedding preview: {summary}")
    return summary
