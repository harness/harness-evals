#!/usr/bin/env python3
"""Load Harness environment profiles and export eval runtime variables.

Example-only CLI — not part of the generic harness-evals package.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from harness_env import (  # noqa: E402
    DEFAULT_PROFILE_PATH,
    HarnessEnvError,
    apply_env_to_process,
    format_shell_exports,
    format_show_output,
    load_and_resolve,
    load_profiles,
    validate_profile_name,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(_repo_root() / "src")
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = src
    return env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Harness environment profiles for harness-agent-conversation evals.",
    )

    env_common = argparse.ArgumentParser(add_help=False)
    env_common.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help=f"Path to environments YAML (default: {DEFAULT_PROFILE_PATH})",
    )
    env_action = argparse.ArgumentParser(add_help=False)
    env_action.add_argument(
        "--login",
        action="store_true",
        help="Fetch a fresh bearer token before export/show/run",
    )
    env_action.add_argument(
        "--save",
        action="store_true",
        help="Write refreshed token back to the profile file (requires --login)",
    )
    env_action.add_argument(
        "--show",
        action="store_true",
        help="Print resolved values with TOKEN masked (for export/login)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", parents=[env_common], help="List environment names in the profile file")

    export_parser = sub.add_parser(
        "export",
        parents=[env_common, env_action],
        help="Print shell export statements for an environment",
    )
    export_parser.add_argument("environment", help="Environment name (e.g. qa, prod0)")

    login_parser = sub.add_parser(
        "login",
        parents=[env_common, env_action],
        help="Fetch bearer token for an environment",
    )
    login_parser.add_argument("environment", help="Environment name (e.g. qa, prod0)")

    show_parser = sub.add_parser(
        "show",
        parents=[env_common, env_action],
        help="Print resolved env vars with secrets masked",
    )
    show_parser.add_argument("environment", help="Environment name (e.g. qa, prod0)")

    run_parser = sub.add_parser(
        "run",
        parents=[env_common, env_action],
        help="Load env vars and invoke harness-evals run",
    )
    run_parser.add_argument("environment", help="Environment name (e.g. qa, prod0)")
    run_parser.add_argument("config", help="Path to eval YAML config file")
    run_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to harness-evals run (prefix with -- if needed)",
    )

    return parser


def cmd_list(profiles: dict[str, dict]) -> int:
    for name in sorted(profiles):
        print(name)
    return 0


def cmd_export(
    env_name: str,
    profile_path: Path,
    *,
    do_login: bool,
    do_save: bool,
    show: bool,
) -> int:
    validate_profile_name(env_name)
    env = load_and_resolve(
        env_name,
        profile_path,
        login=do_login,
        save=do_save,
    )
    if show:
        print(format_show_output(env), file=sys.stderr)
    print(format_shell_exports(env))
    return 0


def cmd_login(
    env_name: str,
    profile_path: Path,
    *,
    do_save: bool,
    show: bool,
) -> int:
    validate_profile_name(env_name)
    env = load_and_resolve(
        env_name,
        profile_path,
        login=True,
        save=do_save,
    )
    if show:
        print(format_show_output(env), file=sys.stderr)
    print(env["TOKEN"])
    return 0


def cmd_show(env_name: str, profile_path: Path, *, do_login: bool) -> int:
    validate_profile_name(env_name)
    env = load_and_resolve(
        env_name,
        profile_path,
        login=do_login,
    )
    print(format_show_output(env))
    return 0


def cmd_run(
    env_name: str,
    config: str,
    profile_path: Path,
    extra_args: list[str],
    *,
    do_login: bool,
    do_save: bool,
) -> int:
    validate_profile_name(env_name)
    env = load_and_resolve(
        env_name,
        profile_path,
        login=do_login,
        save=do_save,
        auto_login=True,
    )
    apply_env_to_process(env)

    cmd = ["poetry", "run", "harness-evals", "run", config]
    if extra_args:
        if extra_args[0] == "--":
            extra_args = extra_args[1:]
        cmd.extend(extra_args)

    return subprocess.call(cmd, env=_subprocess_env(), cwd=_repo_root())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        if args.command == "list":
            profiles = load_profiles(args.profile)
            return cmd_list(profiles)
        if args.command == "export":
            return cmd_export(
                args.environment,
                args.profile,
                do_login=args.login,
                do_save=args.save,
                show=args.show,
            )
        if args.command == "login":
            return cmd_login(
                args.environment,
                args.profile,
                do_save=args.save,
                show=args.show,
            )
        if args.command == "show":
            return cmd_show(args.environment, args.profile, do_login=args.login)
        if args.command == "run":
            return cmd_run(
                args.environment,
                args.config,
                args.profile,
                args.extra_args,
                do_login=args.login,
                do_save=args.save,
            )
    except HarnessEnvError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
