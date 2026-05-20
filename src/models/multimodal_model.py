"""End-to-end multimodal model: CLIP vision encoder + cross-attention + GPT-2 decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.models.encoder import OpenIClipVisionEncoder, build_clip_vision_encoder
from src.models.attention import OpenICrossAttentionFusion, build_cross_attention_fusion
from src.models.decoder import OpenIGPT2Decoder, build_gpt2_decoder


@dataclass(frozen=True)
class MultimodalConfig:
    clip_model_name: str | None = None
    gpt2_model_name: str | None = None
    trainable_top_blocks: int = 2
    num_attention_heads: int = 8
    debug: bool = False
    local_files_only: bool = False


class MultimodalReportGenerator(nn.Module):
    """Compose CLIP encoder, cross-attention fusion, and GPT-2 decoder.

    Usage:
        encoder = build_clip_vision_encoder(local_files_only=True)
        decoder = build_gpt2_decoder(tokenizer, local_files_only=True)
        fusion = build_cross_attention_fusion()
        model = MultimodalReportGenerator(encoder, fusion, decoder)
    """

    def __init__(
        self,
        vision_encoder: OpenIClipVisionEncoder,
        cross_attention: OpenICrossAttentionFusion,
        text_decoder: OpenIGPT2Decoder,
        config: MultimodalConfig | None = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.cross_attention = cross_attention
        self.text_decoder = text_decoder
        self.config = config or MultimodalConfig()

        # If encoder projection dim differs from decoder, add a frozen linear mapping
        encoder_dim = (vision_encoder.vision_model.config.projection_dim if hasattr(vision_encoder.vision_model.config, "projection_dim") else 768)
        decoder_dim = text_decoder.model.config.n_embd
        if encoder_dim != decoder_dim:
            proj = nn.Linear(encoder_dim, decoder_dim, bias=False)
            # keep projection frozen to adhere to "train only cross-attention and LM head"
            for p in proj.parameters():
                p.requires_grad = False
            self.patch_projection = proj
        else:
            self.patch_projection = None

        # Ensure CPU-friendly deterministic default
        self.eval()

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        return_attention: bool = False,
        return_attentions: bool = False,
    ) -> dict[str, Any]:
        """Run end-to-end forward: images -> encoder -> fusion -> decoder logits.

        Args:
            pixel_values: [batch, 3, H, W] image tensors (CLIP-normalized)
            input_ids: [batch, seq] token ids for decoder input
            attention_mask: optional [batch, seq]
            encoder_attention_mask: optional [batch, patches]
            return_attention: whether to include attention weights in the output

        Returns:
            dict with 'logits' (batch, seq, vocab) and optional 'attention_weights'
        """
        # Run CLIP encoder in inference mode to save memory and keep it frozen
        with torch.inference_mode():
            enc_out = self.vision_encoder(pixel_values)
        # Clone inference-mode outputs into normal tensors before autograd uses them.
        patch_embeddings = enc_out["patch_embeddings"].clone()  # [batch, patches, hidden]

        # Run GPT-2 transformer backbone under no_grad because it is frozen.
        with torch.no_grad():
            decoder_outputs = self.text_decoder.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=return_attentions,
            )
        hidden_states = decoder_outputs.hidden_states[-1]  # [batch, seq, hidden]
        decoder_attentions = decoder_outputs.attentions if return_attentions else None

        # Fuse via cross-attention (language queries, image keys/values)
        fusion_out = self.cross_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=patch_embeddings,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            need_weights=return_attention,
        )

        fused_hidden = fusion_out["fused_hidden_states"]
        attn_weights = fusion_out.get("attention_weights")

        # Project fused hidden states to logits via the LM head
        logits = self.text_decoder.lm_head(fused_hidden)

        out: dict[str, Any] = {"logits": logits}
        if return_attention:
            out["attention_weights"] = attn_weights
            out["patch_embeddings"] = patch_embeddings
        if return_attentions:
            out["decoder_attentions"] = decoder_attentions
            out["token_attention_maps"] = attn_weights

        return out

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def validate_frozen(self) -> bool:
        # Encoder must be fully frozen
        enc_frozen = self.vision_encoder.validate_frozen()
        # Decoder frozen pattern validated by its own helper
        dec_frozen = self.text_decoder.validate_frozen()
        # Attention should be trainable-only
        attn_valid = self.cross_attention.validate_frozen_state()
        return enc_frozen and dec_frozen and attn_valid

    def log_model_summary(self) -> None:
        print(f"[multimodal] Total params: {self.total_parameter_count()}")
        print(f"[multimodal] Trainable params: {self.trainable_parameter_count()}")
        print(f"[multimodal] Frozen check: {self.validate_frozen()}")

    def preview_shapes(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        with torch.no_grad():
            enc_out = self.vision_encoder(pixel_values)
            patch_embeddings = enc_out["patch_embeddings"]
            decoder_outputs = self.text_decoder.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            hidden_states = decoder_outputs.hidden_states[-1]
            preview = self.cross_attention.preview_attention_shapes(
                hidden_states=hidden_states,
                encoder_hidden_states=patch_embeddings,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        return {
            "patch_embeddings_shape": preview.encoder_hidden_states_shape,
            "hidden_states_shape": preview.hidden_states_shape,
            "fused_hidden_states_shape": preview.fused_hidden_states_shape,
            "attention_weights_shape": preview.attention_weights_shape,
            "trainable_parameters": self.trainable_parameter_count(),
            "frozen": self.validate_frozen(),
        }


def build_multimodal_model(
    tokenizer, 
    clip_model_name: str | None = None,
    gpt2_model_name: str | None = None,
    trainable_top_blocks: int = 2,
    num_attention_heads: int = 8,
    debug: bool = False,
    local_files_only: bool = False,
) -> MultimodalReportGenerator:
    """Factory to build a multimodal generator from components.

    tokenizer: GPT2Tokenizer already loaded with special tokens.
    """
    if clip_model_name:
        vision_encoder = build_clip_vision_encoder(model_name=clip_model_name, local_files_only=local_files_only)
    else:
        vision_encoder = build_clip_vision_encoder(local_files_only=local_files_only)

    # Determine encoder and decoder hidden dims and align cross-attention to decoder dim
    encoder_dim = (vision_encoder.vision_model.config.projection_dim if hasattr(vision_encoder.vision_model.config, "projection_dim") else 768)
    text_decoder = build_gpt2_decoder(
        tokenizer,
        model_name=gpt2_model_name or "gpt2",
        trainable_top_blocks=trainable_top_blocks,
        local_files_only=local_files_only,
    )
    decoder_dim = text_decoder.model.config.n_embd

    cross_attention = build_cross_attention_fusion(
        hidden_size=decoder_dim,
        num_heads=num_attention_heads,
        debug=debug,
    )


    model = MultimodalReportGenerator(vision_encoder=vision_encoder, cross_attention=cross_attention, text_decoder=text_decoder, config=MultimodalConfig(clip_model_name=clip_model_name, gpt2_model_name=gpt2_model_name, trainable_top_blocks=trainable_top_blocks, num_attention_heads=num_attention_heads, debug=debug, local_files_only=local_files_only))
    model.log_model_summary()
    return model
