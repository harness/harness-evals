"""Targets — the system under test abstraction for harness-evals."""

from harness_evals.targets.auth import ApiKeyAuth, AuthConfig, BasicAuth, BearerAuth, NoAuth
from harness_evals.targets.base import BaseTarget
from harness_evals.targets.http import HttpTarget
from harness_evals.targets.prompt import PromptTarget
from harness_evals.targets.streaming_http import StreamingHttpTarget

__all__ = [
    "BaseTarget",
    "PromptTarget",
    "HttpTarget",
    "StreamingHttpTarget",
    "AuthConfig",
    "NoAuth",
    "BearerAuth",
    "ApiKeyAuth",
    "BasicAuth",
]
