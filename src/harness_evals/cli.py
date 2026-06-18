"""harness-evals CLI — run, import, list-metrics, discover."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from harness_evals.errors import BaselineRegressionError, HarnessEvalsError


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
        "--glob", default=None,
        help="Custom glob pattern (default: **/*.eval.yaml for YAML configs, **/eval_*.py for Python eval files)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "import":
            return _cmd_import(args)
        if args.command == "list-metrics":
            return _cmd_list_metrics()
        if args.command == "discover":
            return _cmd_discover(args)
    except BaselineRegressionError as exc:
        print(f"Baseline regression: {exc}", file=sys.stderr)
        return 1
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


def _cmd_run(args: argparse.Namespace) -> int:
    from harness_evals.config.runner import (
        build_baseline_store,
        gate_against_baseline,
        run_config,
        scores_to_baseline_dict,
    )
    from harness_evals.config.schema import load_config

    cfg = load_config(args.config)

    if args.validate:
        print(f"Config valid: {cfg.name} ({len(cfg.metrics)} metrics)", file=sys.stderr)
        return 0

    baseline_spec = cfg.baseline if (args.baseline or args.update_baseline) else None

    scores = run_config(cfg, baseline=None)

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
        store = build_baseline_store(cfg.baseline)
        run_id = str(uuid.uuid4())[:8]
        store.save(run_id, scores_to_baseline_dict(scores))
        print(f"Baseline saved as run {run_id!r}", file=sys.stderr)

    return exit_code


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
    d["dataset"] = f"{cfg.dataset.source}://{cfg.dataset.id}" + (f"@{cfg.dataset.version}" if cfg.dataset.version else "")
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
        print(f"{e.kind:<{col_kind}}{e.category:<{col_cat}}{e.dimension.value:<{col_dim}}{e.default_threshold:>10.2f}  {llm_flag:>3}")

    print(f"\n{len(entries)} metrics available")
    return 0


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

    print(f"\n{found} eval(s) discovered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
