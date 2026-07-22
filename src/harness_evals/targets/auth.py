"""Authentication configuration for HTTP targets."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass

from harness_evals.env import resolve_env_value


class AuthConfig(ABC):
    """Base class for HTTP authentication strategies."""

    @abstractmethod
    def apply(self, headers: dict[str, str], params: dict[str, str]) -> None:
        """Mutate *headers* and/or *params* to attach credentials."""


@dataclass(frozen=True)
class NoAuth(AuthConfig):
    """No authentication — pass-through."""

    def apply(self, headers: dict[str, str], params: dict[str, str]) -> None:
        return None


@dataclass(frozen=True)
class BearerAuth(AuthConfig):
    """Bearer token authentication. Supports ``${VAR}`` interpolation."""

    token: str

    def apply(self, headers: dict[str, str], params: dict[str, str]) -> None:
        headers["Authorization"] = f"Bearer {resolve_env_value(self.token)}"


@dataclass(frozen=True)
class ApiKeyAuth(AuthConfig):
    """API key authentication in a header or query parameter.

    Supports ``${VAR}`` interpolation on the key value.
    """

    key: str
    header: str = "X-API-Key"
    location: str = "header"

    def __post_init__(self) -> None:
        if self.location not in {"header", "query"}:
            raise ValueError(f"location must be 'header' or 'query', got {self.location!r}")

    def apply(self, headers: dict[str, str], params: dict[str, str]) -> None:
        resolved = resolve_env_value(self.key)
        if self.location == "query":
            params[self.header] = resolved
        else:
            headers[self.header] = resolved


@dataclass(frozen=True)
class BasicAuth(AuthConfig):
    """HTTP Basic authentication. Supports ``${VAR}`` interpolation."""

    username: str
    password: str

    def apply(self, headers: dict[str, str], params: dict[str, str]) -> None:
        user = resolve_env_value(self.username)
        pwd = resolve_env_value(self.password)
        credentials = base64.b64encode(f"{user}:{pwd}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {credentials}"
