"""Reusable multimodal decoding utilities for greedy and beam search generation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import re

import torch


DEFAULT_MAX_CONSECUTIVE_REPEATS = 3


@dataclass(frozen=True)
class GenerationResult:
    """Container for a generated report and the decoding metadata."""

    token_ids: list[int]
    text: str
    score: float
    normalized_score: float
    stop_reason: str


@dataclass(frozen=True)
class BeamHypothesis:
    """Internal beam-search hypothesis."""

    token_ids: list[int]
    score: float
    stop_reason: str = "max_length"

    @property
    def generated_length(self) -> int:
        return max(1, len(self.token_ids) - 1)

    @property
    def normalized_score(self) -> float:
        return self.score / float(self.generated_length**0.7)


def clean_generated_text(text: str) -> str:
    """Remove special tokens and collapse duplicated whitespace."""
    cleaned = text
    for token in ("[START]", "[END]", "[PAD]"):
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _validate_tokenizer(tokenizer: Any) -> None:
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define bos, eos, and pad token ids")


def _filter_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    filtered = logits / temperature
    if top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        values, _ = torch.topk(filtered, top_k, dim=-1)
        threshold = values[..., -1, None]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, float("-inf")), filtered)
    return filtered


def _choose_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> tuple[int, float]:
    filtered = _filter_logits(logits, temperature=temperature, top_k=top_k)
    log_probs = torch.log_softmax(filtered, dim=-1)

    if top_k > 0 or temperature != 1.0:
        probabilities = torch.softmax(filtered, dim=-1)
        next_token = int(torch.multinomial(probabilities, num_samples=1).item())
    else:
        next_token = int(torch.argmax(filtered, dim=-1).item())

    return next_token, float(log_probs[next_token].item())


def _repetition_guard(sequence: list[int], max_consecutive_repeats: int) -> bool:
    if max_consecutive_repeats <= 0:
        return False
    if len(sequence) < max_consecutive_repeats:
        return False

    last_token = sequence[-1]
    return all(token == last_token for token in sequence[-max_consecutive_repeats:])


def _extract_patch_embeddings(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        enc_out = model.vision_encoder(image_tensor)
    return enc_out["patch_embeddings"]


def _next_token_logits(
    model: torch.nn.Module,
    patch_embeddings: torch.Tensor,
    token_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        transformer_outputs = model.text_decoder.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        fusion_out = model.cross_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=patch_embeddings,
            attention_mask=attention_mask,
            need_weights=False,
        )
        logits = model.text_decoder.lm_head(fusion_out["fused_hidden_states"])
    return logits[:, -1, :]


def generate_greedy_report(
    model: torch.nn.Module,
    tokenizer: Any,
    image_tensor: torch.Tensor,
    max_length: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 0,
    max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
) -> GenerationResult:
    """Generate a report using greedy autoregressive decoding."""
    _validate_tokenizer(tokenizer)

    model.eval()
    model.vision_encoder.eval()
    model.text_decoder.model.transformer.eval()
    model.text_decoder.lm_head.eval()
    model.cross_attention.eval()

    patch_embeddings = _extract_patch_embeddings(model, image_tensor)
    token_ids = [int(tokenizer.bos_token_id)]
    score = 0.0
    stop_reason = "max_length"

    with _autocast_context(device):
        for _ in range(max(1, max_length) - 1):
            logits = _next_token_logits(model, patch_embeddings, token_ids, device)
            next_token_id, token_log_prob = _choose_next_token(logits[0], temperature=temperature, top_k=top_k)
            token_ids.append(next_token_id)
            score += token_log_prob

            if next_token_id == tokenizer.eos_token_id:
                stop_reason = "eos_token"
                break
            if _repetition_guard(token_ids, max_consecutive_repeats=max_consecutive_repeats):
                stop_reason = "repetition_guard"
                break

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = clean_generated_text(decoded)
    generated_length = max(1, len(token_ids) - 1)
    normalized_score = score / float(generated_length**0.7)
    return GenerationResult(
        token_ids=token_ids,
        text=text,
        score=score,
        normalized_score=normalized_score,
        stop_reason=stop_reason,
    )


def generate_beam_report(
    model: torch.nn.Module,
    tokenizer: Any,
    image_tensor: torch.Tensor,
    max_length: int,
    device: torch.device,
    beam_size: int = 3,
    temperature: float = 1.0,
    top_k: int = 0,
    max_consecutive_repeats: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
) -> GenerationResult:
    """Generate a report using beam search with normalized score ranking."""
    _validate_tokenizer(tokenizer)
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")

    model.eval()
    model.vision_encoder.eval()
    model.text_decoder.model.transformer.eval()
    model.text_decoder.lm_head.eval()
    model.cross_attention.eval()

    patch_embeddings = _extract_patch_embeddings(model, image_tensor)
    active_beams: list[BeamHypothesis] = [BeamHypothesis(token_ids=[int(tokenizer.bos_token_id)], score=0.0)]

    with _autocast_context(device):
        for _ in range(max(1, max_length) - 1):
            candidates: list[BeamHypothesis] = []
            all_finished = True

            for beam in active_beams:
                if beam.stop_reason != "max_length":
                    candidates.append(beam)
                    continue

                all_finished = False
                logits = _next_token_logits(model, patch_embeddings, beam.token_ids, device)
                filtered = _filter_logits(logits[0], temperature=temperature, top_k=top_k)
                log_probs = torch.log_softmax(filtered, dim=-1)

                candidate_pool = max(beam_size, top_k if top_k > 0 else beam_size)
                candidate_pool = min(candidate_pool, log_probs.size(-1))
                top_scores, top_tokens = torch.topk(log_probs, candidate_pool)

                for token_log_prob, token_id in zip(top_scores.tolist(), top_tokens.tolist()):
                    next_token_id = int(token_id)
                    next_sequence = beam.token_ids + [next_token_id]
                    next_score = beam.score + float(token_log_prob)
                    next_reason = "max_length"

                    if next_token_id == tokenizer.eos_token_id:
                        next_reason = "eos_token"
                    elif _repetition_guard(next_sequence, max_consecutive_repeats=max_consecutive_repeats):
                        next_reason = "repetition_guard"

                    candidates.append(
                        BeamHypothesis(
                            token_ids=next_sequence,
                            score=next_score,
                            stop_reason=next_reason,
                        )
                    )

            candidates.sort(key=lambda candidate: candidate.normalized_score, reverse=True)
            active_beams = candidates[:beam_size]

            if all_finished:
                break

            if all(beam.stop_reason != "max_length" for beam in active_beams):
                break

    best_beam = max(active_beams, key=lambda candidate: candidate.normalized_score)
    decoded = tokenizer.decode(best_beam.token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = clean_generated_text(decoded)
    return GenerationResult(
        token_ids=best_beam.token_ids,
        text=text,
        score=best_beam.score,
        normalized_score=best_beam.normalized_score,
        stop_reason=best_beam.stop_reason if best_beam.stop_reason != "max_length" or len(best_beam.token_ids) >= max_length else "max_length",
    )
