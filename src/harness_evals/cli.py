"""harness-evals CLI — run, import, list-metrics, discover."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from harness_evals.errors import BaselineRegressionError, HarnessEvalsError

logger = logging.getLogger(__name__)

_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``harness-evals`` console script."""

    parser = argparse.ArgumentParser(prog="harness-evals", description="AI evaluation framework CLI")
    parser.add_argument("--verbose", action="store_true", help="Print full tracebacks on errors")
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = sub.add_parser("run", help="Run an eval from a YAML config")
    run_parser.add_argument("config", help="Path to eval YAML config file")
    run_parser.add_argument("--baseline", action="store_true", help="Enable baseline comparison")
    run_parser.add_argument("--update-baseline", action="store_true", help="Save current scores as new baseline")
    run_parser.add_argument("--fail-under", type=float, default=None, help="Exit non-zero if any metric mean < value")
    run_parser.add_argument("--validate", action="store_true", help="Parse and validate config without running")
    run_parser.add_argument(
        "--result-file",
        default=None,
        help="Override the JSON sink output path (timestamp suffix still applies when unique_per_run is set in the eval YAML)",
    )
    run_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stderr progress lines during eval runs",
    )
    run_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        help="Framework log level (overrides HARNESS_EVALS_LOG_LEVEL)",
    )
    run_parser.add_argument(
        "--golden-ids",
        default=None,
        help=(
            "Run only these dataset goldens (comma-separated ids; further restricts eval YAML "
            "golden_ids — errors if there is no overlap)"
        ),
    )
    run_parser.add_argument(
        "--modules",
        default=None,
        help=(
            "Run goldens whose tags.module matches (comma-separated, e.g. ci,ce,cd; further restricts "
            "eval YAML modules — errors if there is no overlap)"
        ),
    )
    run_parser.add_argument(
        "--golden-tags",
        default=None,
        help=(
            "Run goldens matching all tag filters (comma-separated key=value pairs, "
            "e.g. scenario_type=write,environment=qa; adds to eval YAML golden_tags and intersects "
            "shared keys — errors if there is no overlap)"
        ),
    )

    # --- import ---
    import_parser = sub.add_parser("import", help="Translate a platform eval definition to YAML")
    import_parser.add_argument("ref", help="Eval config resource ref (e.g. harness://evals/my-eval@2)")
    import_parser.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")

    # --- list-metrics ---
    sub.add_parser("list-metrics", help="List all available metrics")

    # --- discover ---
    discover_parser = sub.add_parser("discover", help="Discover eval configs in a directory")
    discover_parser.add_argument("path", nargs="?", default=".", help="Directory to search (default: .)")
    discover_parser.add_argument(
        "--glob",
        default=None,
        help="Custom glob pattern (default: **/*.eval.yaml for YAML configs, **/eval_*.py for Python eval files)",
    )

    # --- redteam ---
    redteam_parser = sub.add_parser("redteam", help="Security testing — run adversarial attack plugins")
    redteam_sub = redteam_parser.add_subparsers(dest="redteam_command")

    redteam_run = redteam_sub.add_parser("run", help="Run a red-team scan from a YAML config")
    redteam_run.add_argument("config", help="Path to red-team YAML config file")
    redteam_run.add_argument("--validate", action="store_true", help="Parse and validate config without running")
    redteam_run.add_argument(
        "--fail-on",
        choices=["any", "critical", "high", "medium", "never"],
        default="any",
        help="Minimum severity of a breach that exits non-zero (default: any)",
    )
    redteam_run.add_argument("--json", action="store_true", help="Emit the report as JSON instead of a table")
    redteam_run.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        help="Framework log level (overrides HARNESS_EVALS_LOG_LEVEL)",
    )

    redteam_sub.add_parser("list-packs", help="List available attack packs")

    redteam_plugins = redteam_sub.add_parser("list-plugins", help="List attack plugins and how each is graded")
    redteam_plugins.add_argument("--pack", default=None, help="Only show plugins in this pack")

    # --- init ---
    init_parser = sub.add_parser("init", help="Scaffold a new eval or red-team config")
    init_parser.add_argument(
        "--mode",
        choices=["eval", "redteam"],
        default=None,
        help="Skip the interactive prompt and scaffold this mode directly",
    )
    init_parser.add_argument("--name", default=None, help="Config name (default: prompted, or the directory name)")
    init_parser.add_argument("-o", "--output", default=None, help="Output file (default: <name>.<mode>.yaml)")

    # --- recommend ---
    recommend_parser = sub.add_parser("recommend", help="Recommend evals for a prompt, endpoint, or traces")
    recommend_parser.add_argument("--prompt", default=None, help="Path to a prompt file or prompt text")
    recommend_parser.add_argument("--endpoint", default=None, help="HTTP endpoint URL to evaluate")
    recommend_parser.add_argument("--traces", default=None, help="Path to a traces file (JSONL)")
    recommend_parser.add_argument(
        "--api-key", default=None, help="LLM provider API key (or set ANTHROPIC_API_KEY/OPENAI_API_KEY env var)"
    )
    recommend_parser.add_argument(
        "--provider", default="anthropic", choices=["anthropic", "openai"], help="LLM provider"
    )
    recommend_parser.add_argument("--model", default=None, help="Model name override")
    recommend_parser.add_argument("-o", "--output", default=".", help="Output directory for EvalConfig and goldens")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "recommend":
            return _cmd_recommend(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "redteam":
            if args.redteam_command is None:
                redteam_parser.print_help()
                return 0
            if args.redteam_command == "run":
                return _cmd_redteam_run(args)
            if args.redteam_command == "list-packs":
                return _cmd_redteam_list_packs()
            if args.redteam_command == "list-plugins":
                return _cmd_redteam_list_plugins(args)
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "import":
            return _cmd_import(args)
        if args.command == "list-metrics":
            return _cmd_list_metrics()
        if args.command == "discover":
            return _cmd_discover(args)
    except HarnessEvalsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 2
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 2

    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    from harness_evals._async_compat import _run_async
    from harness_evals.config.runner import build_llm
    from harness_evals.config.schema import ModelSpec
    from harness_evals.recommender.engine import recommend
    from harness_evals.recommender.output import default_model, print_recommendation, write_outputs
    from harness_evals.recommender.scenarios import load_scenario

    env_var = _PROVIDER_ENV_VARS.get(args.provider)
    api_key = args.api_key or (os.environ.get(env_var) if env_var else None)
    if not api_key:
        raise HarnessEvalsError(
            f"No API key provided. Pass --api-key or set {env_var or 'the provider API key'} env var."
        )

    scenario = load_scenario(
        prompt=getattr(args, "prompt", None),
        endpoint=getattr(args, "endpoint", None),
        traces=getattr(args, "traces", None),
    )

    # The CLI owns LLM construction — build_llm resolves the provider class and
    # wires the api_key into the ModelSpec params.
    model_name = args.model or default_model(args.provider)
    logger.info("Building %s LLM for recommendation (model=%s)", args.provider, model_name)
    model_spec = ModelSpec(
        provider=args.provider,
        name=model_name,
        params={"api_key": api_key},
    )
    llm = build_llm(model_spec)

    recommendation = _run_async(recommend(scenario=scenario, llm=llm))
    print_recommendation(recommendation)
    config_path, goldens_path = write_outputs(
        recommendation,
        output_dir=args.output,
        provider=args.provider,
        model=getattr(args, "model", None),
    )
    print(f"EvalConfig written to: {config_path}")
    print(f"Goldens written to:    {goldens_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from harness_evals.config.runner import (
        apply_json_result_file,
        build_baseline_store,
        gate_against_baseline,
        run_config,
        scores_to_baseline_dict,
    )
    from harness_evals.config.schema import load_config
    from harness_evals.datasets.filter import (
        intersect_golden_tag_filters,
        intersect_string_filters,
        parse_golden_ids,
        parse_golden_tags,
        parse_modules,
    )
    from harness_evals.logging_config import configure_logging

    configure_logging(args.log_level)
    _reject_redteam_config(args.config)
    cfg = load_config(args.config)

    if args.golden_ids:
        cli_golden_ids = parse_golden_ids(args.golden_ids)
        assert cli_golden_ids is not None
        cfg.golden_ids = intersect_string_filters(cfg.golden_ids, cli_golden_ids, field_name="golden_ids")
    if args.modules:
        cli_modules = parse_modules(args.modules)
        assert cli_modules is not None
        cfg.modules = intersect_string_filters(cfg.modules, cli_modules, field_name="modules")
    if args.golden_tags:
        cli_golden_tags = parse_golden_tags(args.golden_tags)
        assert cli_golden_tags is not None
        cfg.golden_tags = intersect_golden_tag_filters(cfg.golden_tags, cli_golden_tags)

    if args.validate:
        detail = f"{len(cfg.metrics)} metrics"
        if cfg.golden_ids:
            detail += f", golden_ids={','.join(cfg.golden_ids)}"
        if cfg.modules:
            detail += f", modules={','.join(cfg.modules)}"
        if cfg.golden_tags:
            detail += f", golden_tags={cfg.golden_tags}"
        print(f"Config valid: {cfg.name} ({detail})", file=sys.stderr)
        return 0

    if args.result_file:
        apply_json_result_file(cfg, args.result_file)

    baseline_spec = cfg.baseline if (args.baseline or args.update_baseline) else None

    scores = run_config(cfg, baseline=None, show_progress=not args.no_progress)

    exit_code = 0

    if baseline_spec:
        try:
            gate_against_baseline(scores, baseline_spec)
        except BaselineRegressionError as exc:
            print(f"Baseline regression: {exc}", file=sys.stderr)
            exit_code = 1

    if _any_metric_failed(scores):
        print("Some metrics failed their thresholds.", file=sys.stderr)
        exit_code = 1

    if args.fail_under is not None:
        fail_under_result = _check_fail_under(scores, args.fail_under)
        if fail_under_result:
            print(fail_under_result, file=sys.stderr)
            exit_code = 1

    if args.update_baseline and cfg.baseline:
        if exit_code != 0:
            # Never persist a failing run as the new baseline — doing so would
            # silently ratchet the baseline down to the regressed scores.
            print(
                "Skipping --update-baseline: run did not pass its gates; baseline left unchanged.",
                file=sys.stderr,
            )
        else:
            store = build_baseline_store(cfg.baseline)
            run_id = str(uuid.uuid4())[:8]
            store.save(run_id, scores_to_baseline_dict(scores))
            print(f"Baseline saved as run {run_id!r}", file=sys.stderr)

    return exit_code


def _reject_redteam_config(path: str) -> None:
    """Point the user at the right subcommand instead of a confusing schema error."""
    from harness_evals.config.redteam_schema import config_mode

    text = Path(path).read_text(encoding="utf-8")
    if config_mode(text) == "redteam":
        raise HarnessEvalsError(f"{path} is a red-team config (mode: redteam). Run `harness-evals redteam run {path}`.")


def _any_metric_failed(scores: list[list]) -> bool:
    for case_scores in scores:
        for score in case_scores:
            if not score.passed:
                return True
    return False


def _check_fail_under(scores: list[list], threshold: float) -> str | None:
    """Return an error message if any metric's mean score is below *threshold*."""

    from collections import defaultdict

    by_metric: dict[str, list[float]] = defaultdict(list)
    for case_scores in scores:
        for score in case_scores:
            by_metric[score.name].append(score.value)

    failures: list[str] = []
    for name, values in sorted(by_metric.items()):
        mean = sum(values) / len(values)
        if mean < threshold:
            failures.append(f"{name}={mean:.4f}")

    if failures:
        return f"Metrics below --fail-under {threshold}: {', '.join(failures)}"
    return None


def _cmd_import(args: argparse.Namespace) -> int:
    import yaml

    from harness_evals._async_compat import _run_async
    from harness_evals.plugins import eval_config_source
    from harness_evals.refs import resolve

    ref = resolve(args.ref)
    source_cls = eval_config_source(ref.source)
    source = source_cls()

    async def _fetch():
        async with source:
            return await source.fetch(ref)

    cfg = _run_async(_fetch())

    out_text = yaml.dump(_eval_config_to_dict(cfg), default_flow_style=False, sort_keys=False)
    if args.output:
        Path(args.output).write_text(out_text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(out_text)

    return 0


def _eval_config_to_dict(cfg) -> dict:
    """Serialize an EvalConfig back to a dict suitable for YAML output."""

    d: dict = {"name": cfg.name}
    d["dataset"] = f"{cfg.dataset.source}://{cfg.dataset.id}" + (
        f"@{cfg.dataset.version}" if cfg.dataset.version else ""
    )
    d["target"] = {"type": cfg.target.type, **cfg.target.params}
    d["metrics"] = []
    for m in cfg.metrics:
        if not m.params and m.threshold is None:
            d["metrics"].append(m.kind)
        else:
            entry: dict = {"kind": m.kind}
            if m.threshold is not None:
                entry["threshold"] = m.threshold
            if m.params:
                entry["params"] = m.params
            d["metrics"].append(entry)
    if cfg.sinks and cfg.sinks != []:
        d["sinks"] = []
        for s in cfg.sinks:
            if not s.params:
                d["sinks"].append(s.type)
            else:
                d["sinks"].append({"type": s.type, **s.params})
    return d


def _cmd_list_metrics() -> int:
    from harness_evals.catalog import catalog

    entries = catalog()
    entries.sort(key=lambda e: (e.category, e.kind))

    col_kind = max(len(e.kind) for e in entries) + 2
    col_cat = max(len(e.category) for e in entries) + 2
    col_dim = max(len(e.dimension.value) for e in entries) + 2

    header = f"{'KIND':<{col_kind}}{'CATEGORY':<{col_cat}}{'DIMENSION':<{col_dim}}{'THRESHOLD':>10}  {'LLM':>3}"
    print(header)
    print("-" * len(header))
    for e in entries:
        llm_flag = "yes" if e.requires_llm else ""
        print(
            f"{e.kind:<{col_kind}}{e.category:<{col_cat}}{e.dimension.value:<{col_dim}}{e.default_threshold:>10.2f}  {llm_flag:>3}"
        )

    print(f"\n{len(entries)} metrics available")
    return 0


_SEVERITY_ORDER = ("critical", "high", "medium", "low")

# Findings are for triage, not forensics — a wall of near-identical violations
# buries the distinct ones. --json carries the complete record.
_MAX_FINDING_LINES = 4


def _cmd_redteam_run(args: argparse.Namespace) -> int:
    import json

    from harness_evals.config.redteam_runner import run_redteam_config
    from harness_evals.config.redteam_schema import load_redteam_config
    from harness_evals.logging_config import configure_logging

    configure_logging(args.log_level)
    if args.log_level is None:
        # The dataset runner logs a full traceback per failing case. In a scan
        # that is dozens of near-identical stack traces burying the report,
        # which already names every failure. Opt back in with --log-level.
        logging.getLogger("harness_evals.core.runner").setLevel(logging.CRITICAL)

    cfg = load_redteam_config(args.config)

    if args.validate:
        from harness_evals.config.redteam_runner import resolve_plugins

        plugins = resolve_plugins(cfg)
        print(f"Config valid: {cfg.name} ({len(plugins)} plugin(s))", file=sys.stderr)
        return 0

    report = run_redteam_config(cfg)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_redteam_report(report)

    return _redteam_exit_code(report, args.fail_on)


def _redteam_exit_code(report, fail_on: str) -> int:
    """Exit non-zero when a breach meets the severity floor, or the scan broke.

    A scan that could not reach its target exits 2 rather than 0. Exiting
    clean would let CI go green on a scan that tested nothing, which is the
    failure mode a security gate exists to prevent.
    """
    from harness_evals.redteam.report import Verdict

    if report.by_verdict(Verdict.ERRORED):
        return 2

    if fail_on == "never":
        return 0
    breached = report.breached_plugins
    if fail_on != "any":
        floor = _SEVERITY_ORDER.index(fail_on)
        breached = [r for r in breached if _SEVERITY_ORDER.index(r.severity.value) <= floor]
    return 1 if breached else 0


def _print_redteam_report(report) -> None:
    """Render the per-plugin results table plus the ASR summary."""
    from harness_evals.redteam.report import Verdict

    _VERDICT_LABEL = {
        Verdict.BREACHED: "BREACHED",
        Verdict.DEFENDED: "defended",
        Verdict.DEFENDED_STRUCTURAL: "n/a (no surface)",
        Verdict.UNVERIFIED: "UNVERIFIED",
        Verdict.ERRORED: "ERRORED",
        Verdict.SKIPPED: "skipped",
    }

    results = sorted(
        report.results,
        key=lambda r: (r.verdict is not Verdict.BREACHED, _SEVERITY_ORDER.index(r.severity.value), r.plugin_id),
    )

    print(f"\nRed team: {report.name}")
    print("=" * 96)

    if not results:
        print("No plugins ran.")
        return

    col_id = max(max(len(r.plugin_id) for r in results), len("PLUGIN")) + 2
    col_verdict = max(len(v) for v in _VERDICT_LABEL.values()) + 2
    header = (
        f"{'PLUGIN':<{col_id}}{'SEVERITY':<10}{'VERDICT':<{col_verdict}}"
        f"{'ASR':>6}{'BREACHED':>10}   {'TRACE':<7}{'THREAT'}"
    )
    print(header)
    print("-" * 96)

    for result in results:
        threat = ", ".join((*result.owasp_agentic, *result.owasp_llm, *result.atlas)) or "—"
        asr = f"{result.attack_success_rate:.0%}" if result.tests else "—"
        counts = f"{result.breached}/{result.tests}"
        print(
            f"{result.plugin_id:<{col_id}}{result.severity.value:<10}"
            f"{_VERDICT_LABEL[result.verdict]:<{col_verdict}}"
            f"{asr:>6}{counts:>10}   "
            f"{'yes' if result.requires_trace else '':<7}{threat}"
        )

    print()
    _print_metric_breakdown(results, col_id)

    breaches = [r for r in results if r.verdict is Verdict.BREACHED]
    if breaches:
        print("Findings")
        print("-" * 96)
        for result in breaches:
            _print_breach_detail(result)

    _print_redteam_footer(report, results)


def _print_metric_breakdown(results, col_id: int) -> None:
    """Per-metric pass rates — every grader that ran, not just plugin verdicts."""
    rows: list[tuple[str, str, int, int, float]] = []
    for result in results:
        passed: dict[str, int] = {}
        total: dict[str, int] = {}
        value_sum: dict[str, float] = {}
        for case_scores in result.scores:
            for score in case_scores:
                total[score.name] = total.get(score.name, 0) + 1
                passed[score.name] = passed.get(score.name, 0) + (1 if score.passed else 0)
                value_sum[score.name] = value_sum.get(score.name, 0.0) + score.value
        for name in sorted(total):
            rows.append((result.plugin_id, name, passed[name], total[name], value_sum[name] / total[name]))

    if not rows:
        return

    col_metric = max(max(len(r[1]) for r in rows), len("METRIC")) + 4
    print("Metrics")
    print(f"{'PLUGIN':<{col_id}}{'METRIC':<{col_metric}}{'PASSED':>9}{'MEAN':>8}")
    print("-" * 96)
    for plugin_id, name, passed_n, total_n, mean in rows:
        flag = "" if passed_n == total_n else "  <-- failed"
        print(f"{plugin_id:<{col_id}}{name:<{col_metric}}{f'{passed_n}/{total_n}':>9}{mean:>8.2f}{flag}")
    print()


def _print_redteam_footer(report, results) -> None:
    """Aggregate totals plus the caveats that make a clean result honest."""
    from harness_evals.redteam.report import Verdict

    print("-" * 96)
    print(
        f"{report.total_tests} attacks · {report.total_breached} succeeded · "
        f"attack success rate {report.attack_success_rate:.1%}"
    )

    conclusive = sum(r.tests for r in results if r.conclusive)
    if conclusive != report.total_tests:
        print(
            f"informative attack success rate {report.informative_attack_success_rate:.1%} "
            f"({conclusive}/{report.total_tests} attacks conclusive)"
        )

    by_severity = {s.value: n for s, n in report.count_by_severity().items() if n}
    if by_severity:
        print("breached by severity: " + ", ".join(f"{sev}={n}" for sev, n in by_severity.items()))

    for result in report.by_verdict(Verdict.DEFENDED_STRUCTURAL):
        missing = ", ".join(sorted(c.value for c in result.missing_capabilities))
        print(f"not applicable: {result.plugin_id} — target has no {missing}")

    for result in report.by_verdict(Verdict.ERRORED):
        print(f"ERRORED: {result.plugin_id} — {result.reason or 'target unreachable'}")

    for result in report.by_verdict(Verdict.UNVERIFIED):
        unmet = ", ".join(sorted(p.value for p in result.unmet_preconditions))
        detail = result.reason or f"unmet precondition(s): {unmet}"
        print(f"UNVERIFIED: {result.plugin_id} — {detail}")

    for result in report.by_verdict(Verdict.SKIPPED):
        print(f"skipped: {result.plugin_id} — {result.reason or 'no reason given'}")

    print()
    if not report.passed:
        print("FAIL — at least one attack succeeded.")
        return

    inconclusive = report.by_verdict(Verdict.ERRORED) + report.by_verdict(Verdict.UNVERIFIED)
    if inconclusive:
        # Not "PASS": nothing was demonstrated about the plugins that could
        # not run, and saying otherwise is how a broken scan gets signed off.
        print(f"INCONCLUSIVE — no attack succeeded, but {len(inconclusive)} plugin(s) could not be verified.")
        return

    print("PASS — no attack succeeded.")


def _print_breach_detail(result) -> None:
    """Print the failing metric reasons for one breached plugin."""
    print(f"  {result.plugin_id} — {result.breached}/{result.tests} attacks succeeded:")

    # Attacks in the same family fail the same way, so collapse identical
    # reasons into one line with a count rather than printing seven copies.
    counts: dict[tuple[str, str], int] = {}
    for case_scores in result.scores:
        for score in case_scores:
            if not score.passed:
                key = (score.name, score.reason or "failed")
                counts[key] = counts.get(key, 0) + 1

    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    for (metric_name, reason), n in ordered[:_MAX_FINDING_LINES]:
        times = f" (x{n})" if n > 1 else ""
        print(f"    [{metric_name}]{times} {_truncate(reason)}")

    hidden = sum(n for _, n in ordered[_MAX_FINDING_LINES:])
    if hidden:
        print(f"    … and {hidden} more failure(s) across {len(ordered) - _MAX_FINDING_LINES} other reason(s)")
        print("    Run with --json for the full detail.")
    print()


def _truncate(text: str, limit: int = 180) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cmd_redteam_list_packs() -> int:
    from harness_evals.redteam.packs import packs

    for attack_pack in packs():
        total = len(attack_pack.plugin_ids)
        pending = len(attack_pack.unimplemented())
        print(f"{attack_pack.id}")
        print(f"  {attack_pack.name}")
        print(f"  {total - pending}/{total} plugins implemented")
        print(f"  {attack_pack.description}")
        print()
    return 0


def _cmd_redteam_list_plugins(args: argparse.Namespace) -> int:
    from harness_evals.plugins import registered_attack_plugins
    from harness_evals.redteam.packs import pack as lookup_pack

    if args.pack:
        try:
            attack_pack = lookup_pack(args.pack)
        except KeyError as err:
            raise HarnessEvalsError(err.args[0]) from None
        ids = list(attack_pack.plugin_ids)
    else:
        ids = sorted(registered_attack_plugins())

    if not ids:
        print("No attack plugins registered.")
        return 0

    registered = registered_attack_plugins()
    col_id = max(max(len(i) for i in ids), len("PLUGIN")) + 2

    header = f"{'PLUGIN':<{col_id}}{'SEVERITY':<10}{'STATUS':<14}{'TRACE':<7}{'METRICS'}"
    print(header)
    print("-" * max(len(header), 76))

    implemented = 0
    for plugin_id in ids:
        cls = registered.get(plugin_id)
        if cls is None:
            print(f"{plugin_id:<{col_id}}{'—':<10}{'pending':<14}{'':<7}—")
            continue
        implemented += 1
        plugin = cls()
        metric_names = ", ".join(_plugin_metric_names(plugin)) or "—"
        print(
            f"{plugin_id:<{col_id}}{plugin.severity.value:<10}{'implemented':<14}"
            f"{'yes' if plugin.requires_trace else '':<7}{metric_names}"
        )

    print(f"\n{implemented}/{len(ids)} plugin(s) implemented")
    return 0


def _plugin_metric_names(plugin) -> list[str]:
    """Metric names a plugin grades with, tolerating ones that need a judge."""
    try:
        return [m.name for m in plugin.metrics(llm=None)]
    except Exception:
        return ["(requires judge_llm)"]


_EVAL_TEMPLATE = """\
name: {name}
dataset: ./goldens.jsonl

target:
  type: http
  url: http://localhost:3000/chat
  output_path: $.response

metrics:
  - exact_match
  - {{kind: latency, params: {{max_ms: 10000}}}}

sinks: [stdout]
"""

_REDTEAM_TEMPLATE = """\
name: {name}
mode: redteam

target:
  type: http
  url: http://localhost:3000/chat
  output_path: $.response

redteam:
  packs: [owasp_agentic_top_10]

  # Capabilities the target actually has. Plugins attacking an absent
  # capability still run, but a clean result is reported as "no surface"
  # rather than counted as a defence.
  capabilities:
    - state_changing_tools

  # Harness conditions you have verified. Unmet preconditions downgrade a
  # clean result to UNVERIFIED instead of letting it read as a pass.
  preconditions:
    - tool_trace

sinks: [stdout]
"""


def _cmd_init(args: argparse.Namespace) -> int:
    mode = args.mode
    if mode is None:
        mode = _prompt_mode()
        if mode is None:
            print("Cancelled.", file=sys.stderr)
            return 1

    name = args.name or Path.cwd().name
    template = _EVAL_TEMPLATE if mode == "eval" else _REDTEAM_TEMPLATE
    suffix = "eval.yaml" if mode == "eval" else "redteam.yaml"
    out_path = Path(args.output) if args.output else Path(f"{name}.{suffix}")

    if out_path.exists():
        print(f"Refusing to overwrite existing file: {out_path}", file=sys.stderr)
        return 2

    out_path.write_text(template.format(name=name), encoding="utf-8")

    print(f"Wrote {out_path}")
    if mode == "eval":
        print(f"Next: edit the target and goldens, then run `harness-evals run {out_path}`")
    else:
        print("Next: check coverage with `harness-evals redteam list-plugins --pack owasp_agentic_top_10`")
        print(f"Then run `harness-evals redteam run {out_path}`")
    return 0


def _prompt_mode() -> str | None:
    """Ask which kind of testing to scaffold. Returns None if cancelled."""
    print("What would you like to set up?")
    print("  1) Evals            — measure quality: correctness, groundedness, cost, latency")
    print("  2) Security testing — attack the agent: OWASP Top 10 for Agentic Applications")

    try:
        answer = input("Choose [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if answer in {"1", "eval", "evals"}:
        return "eval"
    if answer in {"2", "security", "redteam"}:
        return "redteam"

    print(f"Unrecognised choice {answer!r}.", file=sys.stderr)
    return None


def _cmd_discover(args: argparse.Namespace) -> int:
    from harness_evals.config.schema import load_config

    root = Path(args.path)
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        return 2

    yaml_patterns = ["**/*.eval.yaml"]
    py_patterns = ["**/eval_*.py"]
    if args.glob:
        yaml_patterns = [args.glob]
        py_patterns = []

    found = 0

    def _is_hidden(path: Path, base: Path) -> bool:
        try:
            rel = path.relative_to(base)
        except ValueError:
            rel = path.resolve().relative_to(base.resolve())
        return any(part.startswith(".") for part in rel.parts)

    for pattern in yaml_patterns:
        for config_path in sorted(root.glob(pattern)):
            if _is_hidden(config_path, root):
                continue
            try:
                cfg = load_config(str(config_path))
                print(f"  {config_path}  name={cfg.name}  metrics={len(cfg.metrics)}")
                found += 1
            except Exception as exc:
                print(f"  {config_path}  ERROR: {exc}", file=sys.stderr)

    for pattern in py_patterns:
        for py_path in sorted(root.glob(pattern)):
            if _is_hidden(py_path, root):
                continue
            print(f"  {py_path}  (Python eval file)")
            found += 1

    print(f"\n{found} eval(s) discovered", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
