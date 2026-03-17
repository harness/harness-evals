"""Pluggable LLM interface for metrics that need a judge."""

from harness_evals.llm.base import BaseLLM

__all__ = ["BaseLLM"]

# Provider imports are deferred to avoid hard dependency on openai/anthropic.
# Use: from harness_evals.llm.openai import OpenAILLM
# Use: from harness_evals.llm.anthropic import AnthropicLLM
