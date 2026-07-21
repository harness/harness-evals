"""Harness environment profiles and login token generation for eval runs.

Example-only utility — not part of the generic harness-evals package.
Used by examples/load_harness_env.py and the harness-agent-conversation eval.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

logger = logging.getLogger(__name__)

REQUIRED_PROFILE_KEYS = ("url", "org", "project", "account_id", "username", "password")
DEFAULT_SSE_PATH_TEMPLATE = (
    "{base_url}/gateway/chat/unified?orgIdentifier={org}&projectIdentifier={project}&accountIdentifier={account_id}"
)
DEFAULT_LOGIN_PATH = "login"
DEFAULT_PROFILE_PATH = Path(".harness-evals/environments.yaml")
ENV_EXPORT_KEYS = ("SSE_ENDPOINT_URL", "HARNESS_ACCOUNT", "HARNESS_ORG", "HARNESS_PROJECT", "TOKEN")


class HarnessEnvError(Exception):
    """Raised when profile loading, login, or env resolution fails."""


def load_profiles(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load environment profiles from YAML and validate required keys."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise HarnessEnvError(f"Profile file not found: {profile_path}")

    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HarnessEnvError(f"Profile file must contain a YAML mapping: {profile_path}")

    defaults = raw.get("_default", {})
    if defaults is not None and not isinstance(defaults, dict):
        raise HarnessEnvError("_default entry must be a mapping when present")

    profiles: dict[str, dict[str, Any]] = {}
    for name, profile in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(profile, dict):
            raise HarnessEnvError(f"Environment {name!r} must be a mapping")
        merged = {**(defaults or {}), **profile}
        missing = [key for key in REQUIRED_PROFILE_KEYS if not merged.get(key)]
        if missing:
            raise HarnessEnvError(f"Environment {name!r} missing required keys: {', '.join(missing)}")
        profiles[name] = merged

    if not profiles:
        raise HarnessEnvError(f"No environments found in profile file: {profile_path}")

    return profiles


def get_profile(profiles: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """Return a single environment profile by name."""

    try:
        return profiles[name]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise HarnessEnvError(f"Unknown environment {name!r}. Available: {available}") from exc


def derive_sse_url(profile: dict[str, Any]) -> str:
    """Build SSE endpoint URL from base URL and profile scope."""

    base_url = str(profile["url"]).rstrip("/")
    template = str(profile.get("sse_path_template") or DEFAULT_SSE_PATH_TEMPLATE)
    try:
        return template.format(
            base_url=base_url,
            org=profile["org"],
            project=profile["project"],
            account_id=profile["account_id"],
        )
    except KeyError as exc:
        raise HarnessEnvError(f"sse_path_template references unknown field: {exc}") from exc


def _find_json_value(obj: Any, key: str) -> str | None:
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            value = obj[key]
            if isinstance(value, str):
                return value
        for item in obj.values():
            found = _find_json_value(item, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_json_value(item, key)
            if found is not None:
                return found
    return None


def parse_login_response(payload: dict[str, Any]) -> dict[str, str]:
    """Extract token and defaultAccountId from a Harness login JSON response."""

    token = _find_json_value(payload, "token")
    if not token:
        raise HarnessEnvError("Login response did not contain a token")

    default_account_id = _find_json_value(payload, "defaultAccountId")
    result: dict[str, str] = {"token": token}
    if default_account_id:
        result["default_account_id"] = default_account_id
    return result


def login(
    base_url: str,
    username: str,
    password: str,
    *,
    login_path: str = DEFAULT_LOGIN_PATH,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """Authenticate via POST /gateway/api/users/login and return token metadata."""

    credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    endpoint = f"{base_url.rstrip('/')}/gateway/api/users/{login_path.lstrip('/')}"
    body = json.dumps({"authorization": f"Basic {credentials}"}).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_s) as response:
            status = response.status
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise HarnessEnvError(f"Login failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise HarnessEnvError(f"Login request failed: {exc.reason}") from exc

    if status != 200:
        raise HarnessEnvError(f"Login failed with HTTP {status}: {raw[:500]}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarnessEnvError(f"Login response was not valid JSON: {raw[:200]}") from exc

    if not isinstance(payload, dict):
        raise HarnessEnvError("Login response must be a JSON object")

    return parse_login_response(payload)


def resolve_env(profile: dict[str, Any], *, token: str | None = None) -> dict[str, str]:
    """Map a profile to eval runtime environment variables."""

    resolved_token = token or profile.get("token")
    if not resolved_token:
        raise HarnessEnvError("No token available. Run with --login or save a cached token with --save.")

    return {
        "SSE_ENDPOINT_URL": derive_sse_url(profile),
        "HARNESS_ACCOUNT": str(profile["account_id"]),
        "HARNESS_ORG": str(profile["org"]),
        "HARNESS_PROJECT": str(profile["project"]),
        "TOKEN": str(resolved_token),
    }


def format_shell_exports(env: dict[str, str]) -> str:
    """Format environment variables as shell export statements safe for eval."""

    lines = []
    for key in ENV_EXPORT_KEYS:
        if key not in env:
            continue
        lines.append(f"export {key}={shlex.quote(env[key])}")
    return "\n".join(lines)


def mask_secret(value: str, *, visible: int = 4) -> str:
    """Mask a secret for display, keeping only the last few characters."""

    if len(value) <= visible:
        return "*" * len(value)
    return f"{'*' * (len(value) - visible)}{value[-visible:]}"


def format_show_output(env: dict[str, str]) -> str:
    """Format resolved env vars with TOKEN masked."""

    lines = []
    for key in ENV_EXPORT_KEYS:
        if key not in env:
            continue
        value = env[key]
        if key == "TOKEN":
            value = mask_secret(value)
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def save_profile_token(path: Path | str, env_name: str, token: str) -> None:
    """Write refreshed token back into the profile YAML file."""

    profile_path = Path(path)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HarnessEnvError(f"Profile file must contain a YAML mapping: {profile_path}")

    env_entry = raw.get(env_name)
    if not isinstance(env_entry, dict):
        raise HarnessEnvError(f"Environment {env_name!r} not found in {profile_path}")

    env_entry["token"] = token
    profile_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def warn_account_mismatch(profile: dict[str, Any], login_result: dict[str, str]) -> None:
    """Log when login defaultAccountId differs from profile account_id."""

    default_account_id = login_result.get("default_account_id")
    profile_account_id = str(profile["account_id"])
    if default_account_id and default_account_id != profile_account_id:
        logger.warning(
            "Login defaultAccountId (%s) differs from profile account_id (%s); using profile value.",
            default_account_id,
            profile_account_id,
        )


def validate_profile_name(name: str) -> None:
    """Reject profile names that could corrupt YAML structure."""

    if name.startswith("_") or not name.strip():
        raise HarnessEnvError(f"Invalid environment name: {name!r}")


def _resolve_token(
    profile_path: Path,
    env_name: str,
    profile: dict[str, Any],
    *,
    do_login: bool,
    do_save: bool,
    auto_login: bool = False,
) -> str:
    if do_save and not do_login and not auto_login:
        raise HarnessEnvError("--save requires --login")

    cached = profile.get("token")
    should_login = do_login or (auto_login and not cached)

    if should_login:
        login_path = str(profile.get("login_path") or DEFAULT_LOGIN_PATH)
        login_result = login(
            str(profile["url"]),
            str(profile["username"]),
            str(profile["password"]),
            login_path=login_path,
        )
        warn_account_mismatch(profile, login_result)
        token = login_result["token"]
        if do_save:
            save_profile_token(profile_path, env_name, token)
        return token

    if cached:
        return str(cached)

    raise HarnessEnvError(
        f"No cached token for {env_name!r}. Run: python examples/load_harness_env.py export {env_name} --login --save"
    )


def load_and_resolve(
    name: str,
    profile_path: Path | str,
    *,
    login: bool = False,
    save: bool = False,
    auto_login: bool = False,
) -> dict[str, str]:
    """Load a named profile, resolve token, and return eval environment variables."""

    validate_profile_name(name)
    path = Path(profile_path)
    profiles = load_profiles(path)
    profile = get_profile(profiles, name)
    token = _resolve_token(
        path,
        name,
        profile,
        do_login=login,
        do_save=save,
        auto_login=auto_login,
    )
    return resolve_env(profile, token=token)


def apply_env_to_process(env: dict[str, str]) -> None:
    """Apply resolved environment variables to the current process."""

    os.environ.update(env)
