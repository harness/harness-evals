"""Targets — the system under test abstraction for harness-evals."""

from harness_evals.targets.auth import ApiKeyAuth, AuthConfig, BasicAuth, BearerAuth, NoAuth
from harness_evals.targets.base import BaseTarget
from harness_evals.targets.http import HttpTarget
from harness_evals.targets.prompt import PromptTarget

__all__ = [
    "BaseTarget",
    "PromptTarget",
    "HttpTarget",
    "AuthConfig",
    "NoAuth",
    "BearerAuth",
    "ApiKeyAuth",
    "BasicAuth",
]
