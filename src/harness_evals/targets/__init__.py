"""Targets — the system under test abstraction for harness-evals."""

from harness_evals.targets.auth import ApiKeyAuth, AuthConfig, BasicAuth, BearerAuth, NoAuth
from harness_evals.targets.base import BaseTarget, ConversationTarget
from harness_evals.targets.conversational_streaming_http import ConversationalStreamingHttpTarget
from harness_evals.targets.http import HttpTarget
from harness_evals.targets.prompt import PromptTarget
from harness_evals.targets.streaming_http import StreamingHttpTarget

__all__ = [
    "BaseTarget",
    "ConversationTarget",
    "PromptTarget",
    "HttpTarget",
    "StreamingHttpTarget",
    "ConversationalStreamingHttpTarget",
    "AuthConfig",
    "NoAuth",
    "BearerAuth",
    "ApiKeyAuth",
    "BasicAuth",
]
