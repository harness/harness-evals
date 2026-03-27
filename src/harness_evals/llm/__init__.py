"""Pluggable LLM and embedding interfaces for evaluation metrics."""

from harness_evals.llm.base import BaseLLM
from harness_evals.llm.embedding import BaseEmbedding

__all__ = ["BaseLLM", "BaseEmbedding"]

# Provider imports are deferred to avoid hard dependency on openai/anthropic.
# Use: from harness_evals.llm.openai import OpenAILLM
# Use: from harness_evals.llm.anthropic import AnthropicLLM
# Use: from harness_evals.llm.openai_embedding import OpenAIEmbedding
