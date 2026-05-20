"""Inference utilities for the OpenI multimodal report generator."""

from .beam_search import GenerationResult, generate_beam_report, generate_greedy_report

__all__ = ["GenerationResult", "generate_beam_report", "generate_greedy_report"]
