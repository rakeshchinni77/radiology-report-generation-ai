"""GPT-2 language decoder for OpenI radiology report generation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


DEFAULT_GPT2_MODEL = "gpt2"
DEFAULT_TRAINABLE_TOP_BLOCKS = 2
DEFAULT_MAX_POSITION_EMBEDDINGS = 256

SPECIAL_TOKENS = {
    "bos_token": "[START]",
    "eos_token": "[END]",
    "pad_token": "[PAD]",
}


@dataclass(frozen=True)
class DecoderConfig:
    model_name: str = DEFAULT_GPT2_MODEL
    vocab_size: int | None = None
    tokenizer_length: int | None = None
    trainable_top_blocks: int = DEFAULT_TRAINABLE_TOP_BLOCKS
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS
    add_cross_attention: bool = True
    local_files_only: bool = False
    use_cache: bool = False


@dataclass(frozen=True)
class DecoderPreview:
    input_ids_shape: tuple[int, ...]
    attention_mask_shape: tuple[int, ...] | None
    logits_shape: tuple[int, ...]
    hidden_state_shape: tuple[int, ...] | None
    attention_shapes: tuple[tuple[int, ...], ...] | None
    encoder_hidden_state_shape: tuple[int, ...] | None
    device: str
    dtype: str


def log(message: str) -> None:
    print(f"[decoder] {message}")


def warn(message: str) -> None:
    print(f"[decoder][warning] {message}", file=sys.stderr)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def count_total_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def validate_decoder_frozen(model: GPT2LMHeadModel, trainable_top_blocks: int = DEFAULT_TRAINABLE_TOP_BLOCKS) -> bool:
    """Validate the intended frozen pattern for the GPT-2 decoder.

    Expected behavior:
    - embeddings are frozen
    - lower transformer blocks are frozen
    - top transformer blocks are trainable
    - LM head is trainable
    """
    """Validate that only the LM head remains trainable inside GPT-2.

    This enforces a strict freeze: embeddings, positional embeddings, all
    transformer blocks and layer norms must be frozen; LM head must be trainable.
    """
    transformer = model.transformer

    # All transformer blocks must be frozen
    for block in transformer.h:
        for parameter in block.parameters():
            if parameter.requires_grad:
                return False

    # Embeddings and final layernorm must be frozen
    for parameter in transformer.wte.parameters():
        if parameter.requires_grad:
            return False
    for parameter in transformer.wpe.parameters():
        if parameter.requires_grad:
            return False
    for parameter in transformer.ln_f.parameters():
        if parameter.requires_grad:
            return False

    # LM head must be trainable
    lm_head_parameters = list(model.lm_head.parameters())
    if not lm_head_parameters:
        return False
    if any(not parameter.requires_grad for parameter in lm_head_parameters):
        return False

    return True


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def _untie_lm_head(model: GPT2LMHeadModel) -> None:
    """Detach the LM head from token embeddings so it can stay trainable while embeddings are frozen."""
    hidden_size = model.config.n_embd
    vocab_size = model.config.vocab_size
    old_weight = model.lm_head.weight.detach().clone()
    new_head = nn.Linear(hidden_size, vocab_size, bias=False)
    # Slightly damp the initial output layer magnitude for more stable early updates.
    new_head.weight = nn.Parameter(old_weight * 0.5)
    model.lm_head = new_head


def _resolve_vocab_size(
    tokenizer: GPT2Tokenizer | None,
    vocab_size: int | None,
) -> int:
    if tokenizer is not None:
        return len(tokenizer)
    if vocab_size is not None:
        return vocab_size
    raise ValueError("Provide either a tokenizer or an explicit vocab_size")


class OpenIGPT2Decoder(nn.Module):
    """Mostly frozen GPT-2 decoder with optional cross-attention support."""

    def __init__(
        self,
        tokenizer: GPT2Tokenizer | None = None,
        model_name: str = DEFAULT_GPT2_MODEL,
        vocab_size: int | None = None,
        tokenizer_length: int | None = None,
        trainable_top_blocks: int = DEFAULT_TRAINABLE_TOP_BLOCKS,
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        local_files_only: bool = False,
        add_cross_attention: bool = True,
        use_cache: bool = False,
    ) -> None:
        super().__init__()
        resolved_vocab_size = _resolve_vocab_size(tokenizer, vocab_size)
        if tokenizer_length is not None:
            resolved_vocab_size = tokenizer_length

        self.config = DecoderConfig(
            model_name=model_name,
            vocab_size=resolved_vocab_size,
            tokenizer_length=tokenizer_length,
            trainable_top_blocks=trainable_top_blocks,
            max_position_embeddings=max_position_embeddings,
            add_cross_attention=add_cross_attention,
            local_files_only=local_files_only,
            use_cache=use_cache,
        )

        gpt2_config = GPT2Config.from_pretrained(model_name, local_files_only=local_files_only)
        gpt2_config.add_cross_attention = add_cross_attention
        gpt2_config.use_cache = use_cache
        gpt2_config.n_positions = max_position_embeddings
        gpt2_config.n_ctx = max_position_embeddings
        gpt2_config.vocab_size = resolved_vocab_size

        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            config=gpt2_config,
            local_files_only=local_files_only,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )

        self.model.resize_token_embeddings(resolved_vocab_size)
        self.model.config.pad_token_id = tokenizer.pad_token_id if tokenizer is not None else None
        self.model.config.bos_token_id = tokenizer.bos_token_id if tokenizer is not None else None
        self.model.config.eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None
        self.model.config.use_cache = use_cache
        self.model.config.output_attentions = True
        self.model.config.output_hidden_states = True
        self.model.config._attn_implementation = "eager"
        self.model.transformer.config._attn_implementation = "eager"
        self.model.eval()

        _untie_lm_head(self.model)

        self._freeze_lower_stack(trainable_top_blocks=trainable_top_blocks)
        self._validate_special_tokens(tokenizer)

        if not validate_decoder_frozen(self.model, trainable_top_blocks=trainable_top_blocks):
            raise RuntimeError("GPT-2 decoder frozen-layer validation failed")

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def transformer(self) -> nn.Module:
        return self.model.transformer

    def train(self, mode: bool = True) -> "OpenIGPT2Decoder":
        """Keep the decoder in eval mode by default to preserve deterministic behavior."""
        super().train(False)
        self.model.eval()
        return self

    def _freeze_lower_stack(self, trainable_top_blocks: int) -> None:
        """Freeze all GPT-2 transformer blocks, embeddings and norms.

        Only the LM head remains trainable inside the GPT-2 model.
        """
        transformer = self.model.transformer
        # Freeze embeddings, position embeddings, dropout and final ln
        _freeze_module(transformer.wte)
        _freeze_module(transformer.wpe)
        _freeze_module(transformer.drop)
        _freeze_module(transformer.ln_f)

        # Freeze every transformer block (including any cross-attn submodules added by config)
        for block in transformer.h:
            _freeze_module(block)

        # Ensure LM head is trainable
        for parameter in self.model.lm_head.parameters():
            parameter.requires_grad = True

    def _validate_special_tokens(self, tokenizer: GPT2Tokenizer | None) -> None:
        if tokenizer is None:
            return

        required_tokens = [SPECIAL_TOKENS["bos_token"], SPECIAL_TOKENS["eos_token"], SPECIAL_TOKENS["pad_token"]]
        missing_tokens: list[str] = []
        for token in required_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                missing_tokens.append(token)

        if missing_tokens:
            raise ValueError(f"Tokenizer is missing required special tokens: {missing_tokens}")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool = True,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
    ) -> Any:
        """Run a GPT-2 forward pass with optional future cross-attention inputs."""
        if input_ids.dim() != 2:
            raise ValueError(f"Expected 2D input_ids [batch, seq], got shape {tuple(input_ids.shape)}")

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids shape, "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )

        if encoder_hidden_states is not None and encoder_hidden_states.dim() != 3:
            raise ValueError(
                "encoder_hidden_states must be 3D [batch, seq, hidden], "
                f"got shape {tuple(encoder_hidden_states.shape)}"
            )

        if encoder_attention_mask is not None and encoder_hidden_states is None:
            raise ValueError("encoder_attention_mask requires encoder_hidden_states")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        return outputs

    def decode(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs: Any) -> Any:
        """Alias for forward to keep call sites readable."""
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self.model)

    def total_parameter_count(self) -> int:
        return count_total_parameters(self.model)

    def validate_frozen(self) -> bool:
        return validate_decoder_frozen(self.model, trainable_top_blocks=self.config.trainable_top_blocks)

    def log_decoder_config(self) -> None:
        log(
            "Config: "
            f"model_name={self.config.model_name}, "
            f"vocab_size={self.config.vocab_size}, "
            f"tokenizer_length={self.config.tokenizer_length}, "
            f"trainable_top_blocks={self.config.trainable_top_blocks}, "
            f"max_position_embeddings={self.config.max_position_embeddings}, "
            f"add_cross_attention={self.config.add_cross_attention}, "
            f"local_files_only={self.config.local_files_only}, "
            f"use_cache={self.config.use_cache}"
        )
        log(f"Total parameters: {self.total_parameter_count()}")
        log(f"Trainable parameters: {self.trainable_parameter_count()}")
        log(f"Frozen validation: {self.validate_frozen()}")
        summary = self.module_freeze_summary()
        log(f"Module freeze summary: {summary}")

    def module_freeze_summary(self) -> dict[str, int | bool]:
        """Return a breakdown of parameter counts and trainable flags for major submodules."""
        transformer = self.model.transformer
        emb_total = sum(p.numel() for p in transformer.wte.parameters()) + sum(p.numel() for p in transformer.wpe.parameters())
        emb_trainable = sum(p.numel() for p in transformer.wte.parameters() if p.requires_grad) + sum(p.numel() for p in transformer.wpe.parameters() if p.requires_grad)
        transformer_total = sum(p.numel() for p in transformer.parameters())
        transformer_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        lm_total = sum(p.numel() for p in self.model.lm_head.parameters())
        lm_trainable = sum(p.numel() for p in self.model.lm_head.parameters() if p.requires_grad)

        return {
            "embeddings_total": emb_total,
            "embeddings_trainable": emb_trainable,
            "transformer_total": transformer_total,
            "transformer_trainable": transformer_trainable,
            "lm_head_total": lm_total,
            "lm_head_trainable": lm_trainable,
        }

    def preview_output_shapes(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> DecoderPreview:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        logits = outputs.logits
        hidden_state = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else None
        attention_shapes = tuple(tuple(attn.shape) for attn in outputs.attentions) if getattr(outputs, "attentions", None) else None

        return DecoderPreview(
            input_ids_shape=tuple(input_ids.shape),
            attention_mask_shape=tuple(attention_mask.shape) if attention_mask is not None else None,
            logits_shape=tuple(logits.shape),
            hidden_state_shape=tuple(hidden_state.shape) if hidden_state is not None else None,
            attention_shapes=attention_shapes,
            encoder_hidden_state_shape=tuple(encoder_hidden_states.shape) if encoder_hidden_states is not None else None,
            device=str(input_ids.device),
            dtype=str(logits.dtype),
        )


def build_gpt2_decoder(
    tokenizer: GPT2Tokenizer,
    model_name: str = DEFAULT_GPT2_MODEL,
    trainable_top_blocks: int = DEFAULT_TRAINABLE_TOP_BLOCKS,
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
    local_files_only: bool = False,
    use_cache: bool = False,
) -> OpenIGPT2Decoder:
    """Factory for the GPT-2 decoder that aligns with the saved tokenizer."""
    decoder = OpenIGPT2Decoder(
        tokenizer=tokenizer,
        model_name=model_name,
        tokenizer_length=len(tokenizer),
        trainable_top_blocks=trainable_top_blocks,
        max_position_embeddings=max_position_embeddings,
        local_files_only=local_files_only,
        add_cross_attention=True,
        use_cache=use_cache,
    )
    decoder.log_decoder_config()
    return decoder


def preview_decoder_output_shapes(
    decoder: OpenIGPT2Decoder,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    encoder_hidden_states: Tensor | None = None,
    encoder_attention_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Return a concise shape summary for debugging and notebook previews."""
    preview = decoder.preview_output_shapes(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
    )
    summary = {
        "input_ids_shape": preview.input_ids_shape,
        "attention_mask_shape": preview.attention_mask_shape,
        "logits_shape": preview.logits_shape,
        "hidden_state_shape": preview.hidden_state_shape,
        "attention_shapes": preview.attention_shapes,
        "encoder_hidden_state_shape": preview.encoder_hidden_state_shape,
        "device": preview.device,
        "dtype": preview.dtype,
        "trainable_parameters": decoder.trainable_parameter_count(),
        "frozen": decoder.validate_frozen(),
    }
    log(f"Output preview: {summary}")
    return summary
