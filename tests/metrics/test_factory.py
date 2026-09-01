"""Metric factory tests: catalog dispatch, code-loading guard, judge metadata."""

import logging
from unittest.mock import AsyncMock

import pytest

from harness_evals import EvalCase
from harness_evals.metrics import factory
from harness_evals.metrics.factory import build_metric, heuristic_options_schema, normalize_metric_config
from harness_evals.metrics.safety.role_violation import RoleViolationMetric


class TestBuildMetric:
    def test_heuristic_regex_from_catalog(self):
        m = build_metric("heuristic", {"kind": "regex"}, score_name="regex_match", threshold=1.0)
        assert m.name == "regex_match"

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_negated_heuristic_from_catalog(self, kind):
        metric = build_metric(
            "heuristic",
            {"kind": kind, "options": {"negate": True, "forbidden": "blocked"}},
            score_name="must_not_match_blocked",
            threshold=1.0,
        )

        assert metric.negate is True
        assert metric.forbidden == "blocked"
        assert metric.name == "must_not_match_blocked"

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_negated_heuristic_builds_when_threshold_omitted(self, kind):
        """``build_metric`` defaults threshold to 0.0, which must not reject negation.

        A negated metric at 0.0 would pass on every input, so it is clamped to
        1.0 rather than raising — otherwise every caller that omits ``threshold``
        could not use negation at all.
        """
        metric = build_metric("heuristic", {"kind": kind, "options": {"negate": True, "forbidden": "blocked"}})

        assert metric.threshold == 1.0
        assert metric.measure(EvalCase(input="q", output="blocked")).passed is False

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_negated_heuristic_allowed_as_composite_sub_metric(self, kind):
        """``_build_composite_metric`` does not forward a threshold to sub-metrics."""
        metric = build_metric(
            "composite",
            {
                "metrics": [
                    {
                        "ref": "must_not_say_blocked",
                        "type": "heuristic",
                        "config": {"kind": kind, "options": {"negate": True, "forbidden": "blocked"}},
                    }
                ]
            },
            threshold=1.0,
        )

        assert metric is not None

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_negated_composite_sub_metric_builds_without_warning(self, kind, caplog):
        """The no-threshold composite path is supported, so it must build quietly.

        ``_build_composite_metric`` never forwards a threshold to sub-metrics,
        and adding ``threshold`` to the sub-metric's ``options`` raises a
        conflict with a factory-supplied argument. A warning here would be
        impossible for the author to act on.
        """
        with caplog.at_level(logging.WARNING):
            build_metric(
                "composite",
                {
                    "metrics": [
                        {
                            "ref": "must_not_say_blocked",
                            "type": "heuristic",
                            "config": {"kind": kind, "options": {"negate": True, "forbidden": "blocked"}},
                        }
                    ]
                },
                score_name="c",
            )

        assert caplog.text == ""

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_threshold_in_options_is_rejected_so_warning_would_be_unactionable(self, kind):
        """Documents why the unspecified-threshold path must not warn."""
        with pytest.raises(TypeError, match="conflict with factory-supplied arguments: threshold"):
            build_metric(
                "heuristic",
                {"kind": kind, "options": {"negate": True, "forbidden": "blocked", "threshold": 1.0}},
            )

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_negated_heuristic_rejects_non_string_forbidden(self, kind):
        """An unquoted YAML scalar must fail config load, not silently always pass."""
        with pytest.raises(ValueError, match="forbidden must be a string"):
            build_metric(
                "heuristic",
                {"kind": kind, "options": {"negate": True, "forbidden": 404}},
                threshold=1.0,
            )

    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    @pytest.mark.parametrize("negate", ["false", "true", 1, 0])
    def test_negated_heuristic_rejects_non_bool_negate(self, kind, negate):
        """A Harness expression renders to a string, so "false" must not enable negation.

        Truthiness alone would turn negation *on* for the string "false" and
        silently invert the assertion, with no error for the author to see.
        """
        with pytest.raises(ValueError, match="negate must be a boolean"):
            build_metric(
                "heuristic",
                {"kind": kind, "options": {"negate": negate, "forbidden": "blocked"}},
                threshold=1.0,
            )

    def test_heuristic_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown heuristic kind"):
            build_metric("heuristic", {"kind": "nope"}, score_name="x")

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown metric type"):
            build_metric("wasm", {}, score_name="x")

    def test_code_metrics_rejected_when_disallowed(self):
        with pytest.raises(ValueError, match="not allowed"):
            build_metric("code", {"path": "mod:Cls"}, score_name="x", allow_code_loading=False)

    def test_nested_code_metrics_rejected_when_disallowed(self):
        with pytest.raises(ValueError, match="not allowed"):
            build_metric(
                "composite",
                {
                    "aggregation": "average",
                    "metrics": [{"ref": "unsafe", "type": "code", "config": {"path": "mod:Cls"}}],
                },
                score_name="composite",
                allow_code_loading=False,
            )

    def test_invalid_heuristic_options_fail_loudly(self):
        with pytest.raises(TypeError):
            build_metric("heuristic", {"kind": "regex", "options": {"not_an_option": True}})

    def test_heuristic_build_normalizes_legacy_latency_options(self):
        metric = build_metric("heuristic", {"kind": "latency", "options": {"max_value": 5000}})

        assert metric.max_ms == 5000

    def test_factory_arguments_cannot_be_overridden_by_options(self):
        with pytest.raises(TypeError, match="threshold"):
            build_metric("heuristic", {"kind": "regex", "options": {"threshold": 0.8}})
        with pytest.raises(TypeError, match="conflict with factory-supplied arguments: name"):
            build_metric("heuristic", {"kind": "regex", "options": {"name": "custom"}})

    def test_inherited_metric_options_are_accepted(self):
        metric = build_metric(
            "llm",
            {"kind": "turn_faithfulness", "options": {"allow_skips": True}, "metadata": {"llm": object()}},
        )
        assert metric.name == "turn_faithfulness"

    def test_geval_dimension_option_accepted(self):
        metric = build_metric(
            "llm",
            {
                "kind": "geval",
                "criteria": "Is it correct?",
                "options": {"dimension": "groundedness"},
                "metadata": {"llm": object()},
            },
            score_name="geval_groundedness",
        )
        assert metric.name == "geval_groundedness"

    def test_injected_llm_options_cannot_be_overridden(self):
        with pytest.raises(TypeError, match="criteria"):
            build_metric(
                "llm",
                {
                    "kind": "conversational_geval",
                    "criteria": "a",
                    "options": {"criteria": "b"},
                    "metadata": {"llm": object()},
                },
            )

    def test_role_violation_forwards_top_level_role_description(self):
        metric = build_metric(
            "llm",
            {
                "kind": "role_violation",
                "role_description": "Only provide account support.",
                "metadata": {"llm": object()},
            },
            threshold=0.85,
        )

        assert isinstance(metric, RoleViolationMetric)
        assert metric.role_description == "Only provide account support."
        assert metric.threshold == 0.85

    def test_null_options_and_config_handled_gracefully(self):
        m = build_metric(
            "heuristic",
            {"kind": "regex", "options": None},
            score_name="regex_match",
        )
        assert m.name == "regex_match"

    def test_weighted_average_with_zero_weights_fails_loudly(self):
        with pytest.raises(ValueError, match="zero total weight"):
            build_metric(
                "composite",
                {
                    "aggregation": "weighted_average",
                    "metrics": [{"ref": "regex", "type": "heuristic", "config": {"kind": "regex"}, "weight": 0}],
                },
                score_name="weighted",
            )

    def test_weighted_average_keeps_weights_aligned_after_skip(self):
        metric = build_metric(
            "composite",
            {
                "aggregation": "weighted_average",
                "metrics": [
                    {"ref": "skipped", "type": "heuristic", "config": {"kind": "regex"}, "weight": 0},
                    {"ref": "scored", "type": "heuristic", "config": {"kind": "regex"}, "weight": 1},
                ],
            },
            score_name="weighted",
        )
        metric.sub_metrics[0].a_measure = AsyncMock(return_value=None)
        score = metric.measure(EvalCase(input="x", output="x", expected="x"))
        assert score.value == 1.0

    def test_llm_build_injects_judge_metadata(self):
        m = build_metric(
            "llm",
            {
                "kind": "geval",
                "criteria": "Is it correct?",
                "metadata": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "k",
                    "base_url": "https://api.openai.com/v1",
                },
            },
            score_name="geval_correctness",
            threshold=0.8,
        )
        assert m.name == "geval_correctness"

    def test_llm_unknown_kind_falls_back_to_geval(self):
        m = build_metric(
            "llm",
            {"criteria": "rate it", "metadata": {"provider": "openai", "model": "gpt-4o", "api_key": "k"}},
            score_name="custom_judge",
        )
        assert m.name == "custom_judge"

    def test_llm_hardcoded_dimension_conflict_caught_at_runtime(self):
        with pytest.raises(TypeError, match="conflict with factory-supplied arguments: dimension"):
            build_metric(
                "llm",
                {
                    "kind": "code_safety",
                    "options": {"dimension": "safety"},
                    "metadata": {"llm": object()},
                },
            )


class TestHeuristicCompatibility:
    @pytest.mark.parametrize("kind", ["contains", "exact_match", "regex"])
    def test_negate_is_declared_in_schema_and_normalizes_without_warning(self, kind, caplog):
        config = {"kind": kind, "options": {"negate": True, "forbidden": "blocked"}}
        with caplog.at_level(logging.WARNING, logger="harness_evals.metrics.factory"):
            normalized = normalize_metric_config("heuristic", config)

        assert normalized == config
        assert heuristic_options_schema(kind)["properties"]["negate"] == {
            "type": "boolean",
            "default": False,
        }
        assert heuristic_options_schema(kind)["properties"]["forbidden"] == {
            "type": ["string", "null"],
            "default": None,
        }
        assert "not declared by the SDK" not in caplog.text

    @pytest.mark.parametrize(("kind", "option_names"), factory._LEGACY_HEURISTIC_OPTIONS.items())
    def test_legacy_heuristic_alias_targets_declared_option(self, kind, option_names):
        _, option_name = option_names
        schema = heuristic_options_schema(kind)

        assert schema is not None
        assert option_name in schema["properties"]

    @pytest.mark.parametrize("legacy_value", [0, 1, 99, -1, True, False, 0.05, "1000", None])
    def test_token_cost_legacy_value_requires_token_plausible_non_boolean_integer(self, caplog, legacy_value):
        config = {"options": {"max_value": legacy_value}}

        with caplog.at_level(logging.WARNING, logger="harness_evals.metrics.factory"):
            normalized = normalize_metric_config("heuristic", config, kind="token_cost")

        assert normalized == config
        assert "treated as monetary cost, not tokens" in caplog.text

    def test_token_cost_normalizes_token_plausible_legacy_value(self):
        normalized = normalize_metric_config("heuristic", {"options": {"max_value": 100}}, kind="token_cost")

        assert normalized == {"options": {"max_tokens": 100}}

    def test_token_cost_preserves_explicit_modern_option(self):
        normalized = normalize_metric_config(
            "heuristic",
            {"options": {"max_value": 1, "max_tokens": 10}},
            kind="token_cost",
        )

        assert normalized == {"options": {"max_tokens": 10}}

    def test_token_cost_rejects_ambiguous_legacy_value(self):
        with pytest.raises(TypeError, match="max_value"):
            build_metric("heuristic", {"kind": "token_cost", "options": {"max_value": 5}})

    def test_heuristic_options_schema_preserves_most_derived_default(self, monkeypatch):
        from harness_evals.metrics import factory

        class BaseMetric:
            def __init__(self, shared: str = "base") -> None:
                self.shared = shared

        class DerivedMetric(BaseMetric):
            def __init__(self, shared: str = "derived") -> None:
                super().__init__(shared)

        monkeypatch.setattr(factory, "_heuristic_registry", lambda: {"derived": DerivedMetric})

        schema = heuristic_options_schema("derived")

        assert schema == {
            "type": "object",
            "properties": {"shared": {"type": "string", "default": "derived"}},
        }

    def test_heuristic_options_schema_types_nullable_constructor_options(self):
        schema = heuristic_options_schema("trajectory_consistency")

        assert schema["properties"]["max_trajectory_length"] == {
            "type": ["integer", "null"],
            "default": None,
        }

    def test_heuristic_options_schema_preserves_nullable_collection_metadata(self):
        schema = heuristic_options_schema("json_diff")

        assert schema["properties"]["exclude_paths"] == {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "default": None,
        }

    @pytest.mark.parametrize(
        ("kind", "option_name", "default"),
        [
            ("latency", "max_ms", 5000),
            ("turn_latency", "max_ms_per_turn", 3000),
        ],
    )
    def test_heuristic_options_schema_prefers_float_annotations(self, kind, option_name, default):
        schema = heuristic_options_schema(kind)

        assert schema["properties"][option_name] == {"type": "number", "default": default}

    def test_heuristic_options_schema_falls_back_for_unsupported_and_unannotated_options(self, monkeypatch):
        class FallbackMetric:
            def __init__(self, unsupported: bytes = "fallback", unannotated=42) -> None:
                self.unsupported = unsupported
                self.unannotated = unannotated

        monkeypatch.setattr(factory, "_heuristic_registry", lambda: {"fallback": FallbackMetric})

        schema = heuristic_options_schema("fallback")

        assert schema["properties"]["unsupported"] == {"type": "string", "default": "fallback"}
        assert schema["properties"]["unannotated"] == {"type": "integer", "default": 42}

    def test_heuristic_options_schema_supports_declared_container_annotations(self):
        tool_arguments = heuristic_options_schema("tool_argument_match")
        schema_validation = heuristic_options_schema("schema_validation")
        trace_completeness = heuristic_options_schema("mcp_trace_completeness")

        assert tool_arguments["properties"]["ignore_keys"] == {
            "type": ["array", "null"],
            "uniqueItems": True,
            "items": {"type": "string"},
            "default": None,
        }
        assert schema_validation["properties"]["schema"]["type"] == ["object", "string"]
        assert trace_completeness["properties"]["expected_trace"]["type"] == "array"


class TestBuildLLMProviderMaxTokens:
    """build_llm_provider should let each provider's own default win when max_tokens is absent."""

    def test_bedrock_openai_no_max_tokens_uses_provider_default(self, monkeypatch):
        # When config has no max_tokens, BedrockOpenAILLM should receive its own 8192 default.
        import sys
        from unittest.mock import MagicMock

        mock_openai = MagicMock()
        captured = {}

        def fake_ctor(**kwargs):
            captured.update(kwargs)
            client = MagicMock()
            client.chat.completions.create = MagicMock()
            return client

        mock_openai.AsyncOpenAI.side_effect = fake_ctor
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer")

        from harness_evals.metrics.factory import build_llm_provider

        llm = build_llm_provider(
            {"metadata": {"provider": "openai", "bedrock": True, "model": "openai.gpt-oss-120b-1:0"}}
        )
        assert llm.max_tokens == 8192

    def test_bedrock_openai_explicit_max_tokens_is_honored(self, monkeypatch):
        # When config sets max_tokens explicitly, it overrides the provider default.
        import sys
        from unittest.mock import MagicMock

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.side_effect = lambda **kw: MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer")

        from harness_evals.metrics.factory import build_llm_provider

        llm = build_llm_provider(
            {
                "metadata": {
                    "provider": "openai",
                    "bedrock": True,
                    "model": "openai.gpt-oss-120b-1:0",
                    "max_tokens": 16384,
                }
            }
        )
        assert llm.max_tokens == 16384


def _capture_gateway_credentials(monkeypatch):
    """Patch the gateway HTTP client factory and record the key sent as ``x-api-key``."""
    import sys
    from unittest.mock import MagicMock

    import harness_evals.llm.harness_gateway as hg

    mock_openai = MagicMock()
    mock_openai.AsyncOpenAI.side_effect = lambda **kw: MagicMock()
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    keys: list[str] = []

    def fake_http_client(api_key, **_kwargs):
        keys.append(api_key)
        return MagicMock()

    monkeypatch.setattr(hg, "_make_x_api_key_http_client", fake_http_client)
    return keys


class TestBuildLLMProviderGatewayRouting:
    def test_promoted_gateway_prefers_harness_token_when_both_env_keys_exist(self, monkeypatch):
        keys = _capture_gateway_credentials(monkeypatch)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.harness")
        monkeypatch.setenv("LLM_GATEWAY_API_KEY", "pat.gateway")

        from harness_evals.metrics.factory import build_llm_provider

        build_llm_provider(
            {
                "metadata": {
                    "provider": "openai",
                    "use_llm_gateway": True,
                    "model": "gpt-4o",
                    "api_key": "sk-vendor",
                }
            }
        )

        assert keys == ["pat.harness"]

    def test_use_llm_gateway_with_anthropic_uses_harness_gateway(self, monkeypatch):
        keys = _capture_gateway_credentials(monkeypatch)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.acc.tok.secret")

        from harness_evals.llm.harness_gateway import HarnessGatewayOpenAILLM
        from harness_evals.metrics.factory import build_llm_provider

        llm = build_llm_provider(
            {
                "metadata": {
                    "provider": "anthropic",
                    "use_llm_gateway": True,
                    "model": "claude-sonnet-4-5",
                    "api_key": "sk-ant-test",
                }
            }
        )
        assert isinstance(llm, HarnessGatewayOpenAILLM)
        assert llm.model == "online/anthropic/claude-sonnet-4-5"
        # A promoted connector carries a vendor key; the gateway only accepts the PAT.
        assert keys == ["pat.acc.tok.secret"]

    def test_promoted_gateway_without_pat_fails_fast_instead_of_using_vendor_key(self, monkeypatch):
        monkeypatch.delenv("HARNESS_TOKEN", raising=False)
        monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")

        from harness_evals.metrics.factory import build_llm_provider

        with pytest.raises(ValueError, match="requires HARNESS_BASE_URL and HARNESS_TOKEN"):
            build_llm_provider(
                {
                    "metadata": {
                        "provider": "openai",
                        "use_llm_gateway": True,
                        "model": "gpt-4o",
                        "api_key": "sk-vendor",
                    }
                }
            )

    def test_explicit_gateway_provider_prefers_configured_pat(self, monkeypatch):
        keys = _capture_gateway_credentials(monkeypatch)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.env.tok.secret")

        from harness_evals.metrics.factory import build_llm_provider

        build_llm_provider(
            {
                "metadata": {
                    "provider": "harness_gateway",
                    "model": "online/openai/gpt-4o",
                    "api_key": "pat.configured.tok.secret",
                }
            }
        )
        assert keys == ["pat.configured.tok.secret"]

    def test_use_llm_gateway_with_openai_uses_harness_gateway(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.side_effect = lambda **kw: MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.acc.tok.secret")

        from harness_evals.llm.harness_gateway import HarnessGatewayOpenAILLM
        from harness_evals.metrics.factory import build_llm_provider

        llm = build_llm_provider({"metadata": {"provider": "openai", "use_llm_gateway": True, "model": "gpt-4o"}})
        assert isinstance(llm, HarnessGatewayOpenAILLM)

    def test_bedrock_openai_without_gateway_uses_bedrock_openai_llm(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer")

        from harness_evals.llm.bedrock import BedrockOpenAILLM
        from harness_evals.metrics.factory import build_llm_provider

        llm = build_llm_provider(
            {
                "metadata": {
                    "provider": "bedrock_openai",
                    "model": "openai.gpt-oss-120b-1:0",
                    "api_key": "bedrock-bearer",
                }
            }
        )
        assert isinstance(llm, BedrockOpenAILLM)


class TestBuildEmbeddingProviderGatewayRouting:
    def test_use_llm_gateway_returns_harness_gateway_embedding(self, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        captured: dict = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.side_effect = capture
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.acc.tok.secret")

        from harness_evals.llm.harness_gateway import HarnessGatewayOpenAIEmbedding
        from harness_evals.metrics.factory import build_embedding_provider

        embedding = build_embedding_provider({"use_llm_gateway": True, "provider": "openai"})
        assert isinstance(embedding, HarnessGatewayOpenAIEmbedding)
        assert captured["api_key"] == "unused"
        assert captured["http_client"] is not None
        assert captured["base_url"] == "https://qa.harness.io/prod1/llm-gw/v1"

    def test_use_llm_gateway_with_anthropic_returns_harness_gateway_embedding(self, monkeypatch):
        keys = _capture_gateway_credentials(monkeypatch)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.acc.tok.secret")

        from harness_evals.llm.harness_gateway import HarnessGatewayOpenAIEmbedding
        from harness_evals.metrics.factory import build_embedding_provider

        embedding = build_embedding_provider(
            {
                "use_llm_gateway": True,
                "provider": "anthropic",
                "embedding_model": "text-embedding-3-small",
                "api_key": "sk-ant-test",
            }
        )
        assert isinstance(embedding, HarnessGatewayOpenAIEmbedding)
        assert keys == ["pat.acc.tok.secret"]

    def test_explicit_gateway_provider_embedding_prefers_configured_pat(self, monkeypatch):
        keys = _capture_gateway_credentials(monkeypatch)
        monkeypatch.setenv("HARNESS_BASE_URL", "https://qa.harness.io/prod1")
        monkeypatch.setenv("HARNESS_TOKEN", "pat.env.tok.secret")

        from harness_evals.metrics.factory import build_embedding_provider

        build_embedding_provider({"provider": "harness_gateway", "api_key": "pat.configured.tok.secret"})
        assert keys == ["pat.configured.tok.secret"]
