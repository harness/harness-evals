"""Tests for the redteam CLI surface and its config schema."""

from __future__ import annotations

import pytest

from harness_evals.cli import main
from harness_evals.config.redteam_schema import config_mode, loads_redteam_config
from harness_evals.core.score import Score
from harness_evals.errors import HarnessEvalsError
from harness_evals.redteam.plugin import Capability, Precondition, Severity
from harness_evals.redteam.report import PluginResult, RedTeamReport, Verdict

_MINIMAL_CONFIG = """
name: demo
mode: redteam
target:
  type: http
  url: http://localhost:3000/chat
redteam:
  packs: [owasp_agentic_top_10]
"""


# --- config schema ---


@pytest.mark.unit
def test_minimal_redteam_config_parses():
    cfg = loads_redteam_config(_MINIMAL_CONFIG)
    assert cfg.name == "demo"
    assert cfg.packs == ["owasp_agentic_top_10"]
    assert cfg.target.type == "http"


@pytest.mark.unit
def test_capabilities_and_preconditions_parse_to_enums():
    cfg = loads_redteam_config(
        _MINIMAL_CONFIG
        + """
  capabilities: [state_changing_tools]
  preconditions: [tool_trace]
"""
    )
    assert cfg.capabilities == [Capability.STATE_CHANGING_TOOLS]
    assert cfg.preconditions == [Precondition.TOOL_TRACE]


@pytest.mark.unit
def test_unknown_capability_is_rejected_with_valid_values():
    with pytest.raises(HarnessEvalsError, match="Valid values"):
        loads_redteam_config(_MINIMAL_CONFIG + "  capabilities: [teleportation]\n")


@pytest.mark.unit
def test_promptfoo_plugin_aliases_are_normalized():
    cfg = loads_redteam_config(
        """
name: demo
mode: redteam
target: {type: http, url: http://x}
redteam:
  plugins: ["sql-injection", {id: bola}]
"""
    )
    assert [p.id for p in cfg.plugins] == ["sql_injection", "object_authorization_bypass"]


@pytest.mark.unit
def test_plugin_params_are_preserved():
    cfg = loads_redteam_config(
        """
name: demo
mode: redteam
target: {type: http, url: http://x}
redteam:
  plugins:
    - id: unauthorized_state_change
      params: {state_changing_tools: {send_email: [to]}}
"""
    )
    assert cfg.plugins[0].params == {"state_changing_tools": {"send_email": ["to"]}}


@pytest.mark.unit
def test_config_with_no_packs_or_plugins_is_rejected():
    with pytest.raises(HarnessEvalsError, match="at least one entry"):
        loads_redteam_config("name: demo\nmode: redteam\ntarget: {type: http, url: http://x}\nredteam: {}\n")


@pytest.mark.unit
def test_unknown_top_level_key_is_rejected():
    with pytest.raises(HarnessEvalsError, match="Unknown top-level key"):
        loads_redteam_config(_MINIMAL_CONFIG + "dataset: ./goldens.jsonl\n")


@pytest.mark.unit
def test_eval_config_passed_to_redteam_loader_is_rejected():
    with pytest.raises(HarnessEvalsError, match="Expected mode 'redteam'"):
        loads_redteam_config("name: demo\nmode: eval\ntarget: {type: http, url: http://x}\nredteam: {}\n")


@pytest.mark.unit
def test_config_mode_defaults_to_eval():
    assert config_mode("name: demo\nmetrics: [exact_match]\n") == "eval"
    assert config_mode(_MINIMAL_CONFIG) == "redteam"


@pytest.mark.unit
def test_unknown_mode_is_rejected():
    with pytest.raises(HarnessEvalsError, match="Unknown mode"):
        config_mode("name: demo\nmode: chaos\n")


# --- plugin resolution ---


@pytest.mark.unit
def test_resolve_plugins_skips_unimplemented_pack_members():
    from harness_evals.config.redteam_runner import resolve_plugins
    from harness_evals.redteam.packs import OWASP_AGENTIC_TOP_10

    plugins = resolve_plugins(loads_redteam_config(_MINIMAL_CONFIG))
    assert "unauthorized_state_change" in {p.id for p in plugins}
    assert len(plugins) < len(OWASP_AGENTIC_TOP_10.plugin_ids)


@pytest.mark.unit
def test_explicit_plugin_entry_overrides_the_pack_copy():
    """A config can pull in a whole pack and still pass params to one member."""
    from harness_evals.config.redteam_runner import resolve_plugins

    cfg = loads_redteam_config(
        _MINIMAL_CONFIG
        + """
  plugins:
    - id: unauthorized_state_change
      params: {state_changing_tools: {wire_transfer: [recipient]}}
"""
    )
    matching = [p for p in resolve_plugins(cfg) if p.id == "unauthorized_state_change"]
    assert len(matching) == 1, "the pack copy should have been replaced, not added alongside"
    assert matching[0].state_changing_tools == {"wire_transfer": ["recipient"]}


@pytest.mark.unit
def test_bad_plugin_params_name_the_plugin():
    from harness_evals.config.redteam_runner import resolve_plugins

    cfg = loads_redteam_config(
        """
name: demo
mode: redteam
target: {type: http, url: http://x}
redteam:
  plugins:
    - id: unauthorized_state_change
      params: {nonexistent_kwarg: 1}
"""
    )
    with pytest.raises(HarnessEvalsError, match="Invalid params for plugin 'unauthorized_state_change'"):
        resolve_plugins(cfg)


@pytest.mark.unit
def test_unknown_pack_id_surfaces_known_packs():
    from harness_evals.config.redteam_runner import resolve_plugins

    cfg = loads_redteam_config(_MINIMAL_CONFIG.replace("owasp_agentic_top_10", "owasp_nope"))
    with pytest.raises(HarnessEvalsError, match="Known packs"):
        resolve_plugins(cfg)


# --- CLI commands ---


@pytest.mark.unit
def test_list_packs_shows_implementation_progress(capsys):
    assert main(["redteam", "list-packs"]) == 0
    out = capsys.readouterr().out
    assert "owasp_agentic_top_10" in out
    assert "/22 plugins implemented" in out


@pytest.mark.unit
def test_list_plugins_shows_the_metrics_each_plugin_grades_with(capsys):
    assert main(["redteam", "list-plugins"]) == 0
    out = capsys.readouterr().out
    assert "unauthorized_state_change" in out
    assert "argument_provenance" in out
    assert "critical" in out


@pytest.mark.unit
def test_list_plugins_for_a_pack_marks_unimplemented_members_pending(capsys):
    assert main(["redteam", "list-plugins", "--pack", "owasp_agentic_top_10"]) == 0
    out = capsys.readouterr().out
    assert "pending" in out
    assert "sql_injection" in out


@pytest.mark.unit
def test_list_plugins_rejects_unknown_pack(capsys):
    assert main(["redteam", "list-plugins", "--pack", "nope"]) == 2
    assert "Unknown attack pack" in capsys.readouterr().err


@pytest.mark.unit
def test_redteam_validate_reports_plugin_count(tmp_path, capsys):
    import re

    cfg_path = tmp_path / "demo.redteam.yaml"
    cfg_path.write_text(_MINIMAL_CONFIG, encoding="utf-8")

    assert main(["redteam", "run", str(cfg_path), "--validate"]) == 0
    assert re.search(r"Config valid: demo \(\d+ plugin\(s\)\)", capsys.readouterr().err)


@pytest.mark.unit
def test_run_rejects_a_redteam_config_and_points_at_the_right_command(tmp_path, capsys):
    cfg_path = tmp_path / "demo.redteam.yaml"
    cfg_path.write_text(_MINIMAL_CONFIG, encoding="utf-8")

    assert main(["run", str(cfg_path)]) == 2
    assert "harness-evals redteam run" in capsys.readouterr().err


@pytest.mark.unit
def test_bare_redteam_command_prints_help(capsys):
    assert main(["redteam"]) == 0
    assert "list-plugins" in capsys.readouterr().out


# --- init ---


@pytest.mark.unit
def test_init_redteam_writes_a_runnable_config(tmp_path, capsys):
    out_path = tmp_path / "x.redteam.yaml"
    assert main(["init", "--mode", "redteam", "--name", "demo", "-o", str(out_path)]) == 0

    cfg = loads_redteam_config(out_path.read_text(encoding="utf-8"))
    assert cfg.name == "demo"
    assert cfg.packs == ["owasp_agentic_top_10"]
    assert "redteam list-plugins" in capsys.readouterr().out


@pytest.mark.unit
def test_init_eval_writes_a_parseable_eval_config(tmp_path):
    from harness_evals.config.schema import loads_config

    out_path = tmp_path / "x.eval.yaml"
    assert main(["init", "--mode", "eval", "--name", "demo", "-o", str(out_path)]) == 0

    cfg = loads_config(out_path.read_text(encoding="utf-8"), base_dir=str(tmp_path))
    assert cfg.name == "demo"
    assert [m.kind for m in cfg.metrics] == ["exact_match", "latency"]


@pytest.mark.unit
def test_init_never_overwrites_an_existing_file(tmp_path, capsys):
    out_path = tmp_path / "x.redteam.yaml"
    out_path.write_text("keep me", encoding="utf-8")

    assert main(["init", "--mode", "redteam", "-o", str(out_path)]) == 2
    assert out_path.read_text(encoding="utf-8") == "keep me"
    assert "Refusing to overwrite" in capsys.readouterr().err


@pytest.mark.unit
def test_interactive_init_accepts_the_security_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "2")
    out_path = tmp_path / "x.yaml"
    assert main(["init", "--name", "demo", "-o", str(out_path)]) == 0
    assert "mode: redteam" in out_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_interactive_init_cancelled_writes_nothing(tmp_path, monkeypatch):
    def _cancel(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", _cancel)
    out_path = tmp_path / "x.yaml"
    assert main(["init", "--name", "demo", "-o", str(out_path)]) == 1
    assert not out_path.exists()


# --- report rendering ---


def _score(name: str, value: float, reason: str) -> Score:
    return Score(name=name, value=value, threshold=1.0, reason=reason)


def _report() -> RedTeamReport:
    breached = PluginResult(
        plugin_id="unauthorized_state_change",
        name="Unauthorized State Change",
        severity=Severity.CRITICAL,
        verdict=Verdict.BREACHED,
        tests=4,
        breached=1,
        owasp_agentic=("ASI02",),
        requires_trace=True,
        scores=[
            [_score("argument_provenance", 1.0, "all args grounded")],
            [_score("argument_provenance", 0.0, "ungrounded 'to': victim@example.com")],
            [_score("argument_provenance", 1.0, "all args grounded")],
            [_score("argument_provenance", 1.0, "all args grounded")],
        ],
    )
    structural = PluginResult(
        plugin_id="sql_injection",
        name="SQL Injection",
        severity=Severity.HIGH,
        verdict=Verdict.DEFENDED_STRUCTURAL,
        tests=2,
        breached=0,
        missing_capabilities=frozenset({Capability.DATABASE}),
        scores=[[_score("tool_call_constraint", 1.0, "no violations")] for _ in range(2)],
    )
    unverified = PluginResult(
        plugin_id="cross_session_leak",
        name="Cross Session Leak",
        severity=Severity.HIGH,
        verdict=Verdict.UNVERIFIED,
        tests=2,
        breached=0,
        unmet_preconditions=frozenset({Precondition.SESSION_MEMORY}),
        scores=[[_score("pii", 1.0, "no pii")] for _ in range(2)],
    )
    return RedTeamReport(name="demo", results=[structural, breached, unverified])


@pytest.mark.unit
def test_report_output_lists_every_metric_that_ran(capsys):
    from harness_evals.cli import _print_redteam_report

    _print_redteam_report(_report())
    out = capsys.readouterr().out

    assert "argument_provenance" in out
    assert "tool_call_constraint" in out
    assert "pii" in out
    assert "3/4" in out  # per-metric pass count for the breached plugin


@pytest.mark.unit
def test_report_output_shows_the_failing_metric_reason(capsys):
    from harness_evals.cli import _print_redteam_report

    _print_redteam_report(_report())
    out = capsys.readouterr().out

    assert "ungrounded 'to': victim@example.com" in out
    assert "FAIL" in out


@pytest.mark.unit
def test_report_output_distinguishes_structural_and_unverified_from_defended(capsys):
    """A clean result with no attack surface must never read as a defence."""
    from harness_evals.cli import _print_redteam_report

    _print_redteam_report(_report())
    out = capsys.readouterr().out

    assert "not applicable: sql_injection" in out
    assert "database" in out
    assert "UNVERIFIED: cross_session_leak" in out
    assert "session_memory" in out
    assert "informative attack success rate" in out


@pytest.mark.unit
def test_identical_failure_reasons_are_collapsed_with_a_count(capsys):
    from harness_evals.cli import _print_breach_detail

    result = PluginResult(
        plugin_id="p",
        name="P",
        severity=Severity.HIGH,
        verdict=Verdict.BREACHED,
        tests=3,
        breached=3,
        scores=[[_score("m", 0.0, "same reason")] for _ in range(3)],
    )
    _print_breach_detail(result)
    out = capsys.readouterr().out
    assert "(x3) same reason" in out
    assert out.count("same reason") == 1


@pytest.mark.unit
def test_long_finding_lists_are_capped_and_point_at_json(capsys):
    from harness_evals.cli import _MAX_FINDING_LINES, _print_breach_detail

    n = _MAX_FINDING_LINES + 3
    result = PluginResult(
        plugin_id="p",
        name="P",
        severity=Severity.HIGH,
        verdict=Verdict.BREACHED,
        tests=n,
        breached=n,
        scores=[[_score("m", 0.0, f"distinct reason {i}")] for i in range(n)],
    )
    _print_breach_detail(result)
    out = capsys.readouterr().out
    assert out.count("distinct reason") == _MAX_FINDING_LINES
    assert "and 3 more failure(s)" in out
    assert "--json" in out


@pytest.mark.unit
def test_breached_plugins_are_listed_first(capsys):
    from harness_evals.cli import _print_redteam_report

    _print_redteam_report(_report())
    out = capsys.readouterr().out
    table = out[out.index("PLUGIN") : out.index("Metrics")]
    assert table.index("unauthorized_state_change") < table.index("sql_injection")


# --- end to end through the CLI ---


_STUB_TARGET_MODULE = '''
"""Stub target that leaks a third-party address into a tool argument."""

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import ToolCall
from harness_evals.plugins import register_target


@register_target("stub_leaky")
class StubLeakyTarget:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def ainvoke(self, golden):
        return EvalCase(
            input=golden.input,
            output="Done, I have subscribed that address.",
            tool_calls=[
                ToolCall(
                    name="request_restock_notification",
                    input={"email": "victim@example.com", "product_id": "ELEC-001"},
                )
            ],
        )
'''


@pytest.mark.unit
def test_cli_run_reports_a_breach_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "stub_target_mod.py").write_text(_STUB_TARGET_MODULE, encoding="utf-8")

    cfg_path = tmp_path / "demo.redteam.yaml"
    cfg_path.write_text(
        """
name: stub-scan
mode: redteam
plugin_modules: [stub_target_mod]
target:
  type: stub_leaky
redteam:
  plugins: [unauthorized_state_change]
  capabilities: [state_changing_tools]
  preconditions: [tool_trace]
""",
        encoding="utf-8",
    )

    assert main(["redteam", "run", str(cfg_path)]) == 1

    out = capsys.readouterr().out
    assert "BREACHED" in out
    assert "argument_provenance" in out
    assert "victim@example.com" in out
    assert "FAIL — at least one attack succeeded." in out


@pytest.mark.unit
def test_cli_run_json_output_is_machine_readable(tmp_path, capsys, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "stub_target_mod2.py").write_text(
        _STUB_TARGET_MODULE.replace("stub_leaky", "stub_leaky2"), encoding="utf-8"
    )

    cfg_path = tmp_path / "demo.redteam.yaml"
    cfg_path.write_text(
        """
name: stub-scan
mode: redteam
plugin_modules: [stub_target_mod2]
target:
  type: stub_leaky2
redteam:
  plugins: [unauthorized_state_change]
  capabilities: [state_changing_tools]
  preconditions: [tool_trace]
""",
        encoding="utf-8",
    )

    assert main(["redteam", "run", str(cfg_path), "--json", "--fail-on", "never"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "stub-scan"
    assert payload["passed"] is False
    assert payload["results"][0]["verdict"] == "breached"


@pytest.mark.unit
def test_fail_on_severity_floor_ignores_lower_severity_breaches():
    from harness_evals.cli import _redteam_exit_code

    report = _report()
    assert _redteam_exit_code(report, "any") == 1
    assert _redteam_exit_code(report, "critical") == 1
    assert _redteam_exit_code(report, "never") == 0

    report.results[1].severity = Severity.MEDIUM
    assert _redteam_exit_code(report, "critical") == 0
    assert _redteam_exit_code(report, "medium") == 1
