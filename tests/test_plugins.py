"""Tests for plugin registration and lookup."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from harness_evals import EvalCase, Score, plugins
from harness_evals.catalog import catalog
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.errors import HarnessEvalsError, MissingAdapterError

FAMILY_HELPERS = [
    (plugins.DATASET_SOURCES, plugins.register_dataset_source, plugins.dataset_source),
    (plugins.PROMPT_SOURCES, plugins.register_prompt_source, plugins.prompt_source),
    (plugins.EVAL_CASE_SOURCES, plugins.register_eval_case_source, plugins.eval_case_source),
    (plugins.EVAL_CONFIG_SOURCES, plugins.register_eval_config_source, plugins.eval_config_source),
    (plugins.TARGETS, plugins.register_target, plugins.target),
    (plugins.METRICS, plugins.register_metric, plugins.metric),
    (plugins.BASELINE_STORES, plugins.register_baseline_store, plugins.baseline_store),
    (plugins.SINKS, plugins.register_sink, plugins.sink),
]


@pytest.fixture(autouse=True)
def restore_plugin_state() -> None:
    saved_registries = {family: registry.copy() for family, registry in plugins._REGISTRIES.items()}
    saved_entry_points = {family: discovered.copy() for family, discovered in plugins._ENTRY_POINTS.items()}
    saved_discovered = plugins._ENTRY_POINTS_DISCOVERED

    yield

    for family, registry in plugins._REGISTRIES.items():
        registry.clear()
        registry.update(saved_registries[family])
    for family, discovered in plugins._ENTRY_POINTS.items():
        discovered.clear()
        discovered.update(saved_entry_points[family])
    plugins._ENTRY_POINTS_DISCOVERED = saved_discovered


@pytest.mark.unit
@pytest.mark.parametrize(("family", "register", "lookup"), FAMILY_HELPERS)
def test_register_decorator_populates_each_family(
    family: str,
    register: Any,
    lookup: Any,
) -> None:
    class Adapter:
        pass

    decorated = register(f"{family}_adapter")(Adapter)

    assert decorated is Adapter
    assert lookup(f"{family}_adapter") is Adapter


@pytest.mark.unit
def test_double_registration_is_last_wins() -> None:
    class FirstAdapter:
        pass

    class SecondAdapter:
        pass

    plugins.register_dataset_source("duplicate")(FirstAdapter)
    with pytest.warns(UserWarning, match="duplicate"):
        plugins.register_dataset_source("duplicate")(SecondAdapter)

    assert plugins.dataset_source("duplicate") is SecondAdapter


@pytest.mark.unit
@pytest.mark.parametrize(("family", "_register", "lookup"), FAMILY_HELPERS)
def test_missing_adapter_error_names_family_and_install_hint(
    family: str,
    _register: Any,
    lookup: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    class FakeEntryPoint:
        name = "registered_elsewhere"

        def load(self) -> type:
            loaded.append("loaded")
            return type("ShouldNotLoad", (), {})

    monkeypatch.setattr(plugins, "entry_points", lambda group: [FakeEntryPoint()])
    plugins._ENTRY_POINTS_DISCOVERED = False

    with pytest.raises(MissingAdapterError) as exc_info:
        lookup("langfuse")

    err = exc_info.value
    assert err.source == "langfuse"
    assert err.family == family
    assert err.install_hint == "harness-evals[langfuse]"
    assert family in str(err)
    assert "harness-evals[langfuse]" in str(err)
    assert loaded == []


@pytest.mark.unit
def test_lookup_loads_lazy_entry_point_on_first_match(monkeypatch: pytest.MonkeyPatch) -> None:
    class LazyAdapter:
        pass

    class FakeEntryPoint:
        name = "lazy"

        def __init__(self) -> None:
            self.loads = 0

        def load(self) -> type:
            self.loads += 1
            return LazyAdapter

    fake_entry_point = FakeEntryPoint()
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda group: [fake_entry_point] if group == "harness_evals.dataset_sources" else [],
    )
    plugins._ENTRY_POINTS_DISCOVERED = False

    assert plugins.dataset_source("lazy") is LazyAdapter
    assert plugins.dataset_source("lazy") is LazyAdapter
    assert fake_entry_point.loads == 1


@pytest.mark.unit
def test_load_plugins_imports_module_and_triggers_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "sample_plugin.py"
    module_path.write_text(
        "\n".join(
            [
                "from harness_evals.plugins import register_dataset_source",
                "",
                "@register_dataset_source('sample')",
                "class SampleDatasetSource:",
                "    pass",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("sample_plugin", None)

    plugins.load_plugins(["sample_plugin"])

    assert plugins.dataset_source("sample").__name__ == "SampleDatasetSource"


@pytest.mark.unit
def test_load_plugins_unknown_module_raises_clear_error() -> None:
    with pytest.raises(HarnessEvalsError, match="Failed to load plugin module"):
        plugins.load_plugins(["definitely_missing_harness_evals_plugin"])


@pytest.mark.unit
def test_catalog_includes_registered_metric() -> None:
    @plugins.register_metric("custom_plugin_metric")
    class CustomPluginMetric(BaseMetric):
        """Custom plugin metric."""

        def __init__(self, threshold: float = 0.5) -> None:
            super().__init__(name="custom_plugin_metric", dimension=Dimension.CORRECTNESS, threshold=threshold)

        def measure(self, eval_case: EvalCase) -> Score:
            return Score(name=self.name, value=1.0, threshold=self.threshold)

    entry = next(item for item in catalog() if item.kind == "custom_plugin_metric")

    assert entry.metric_class is CustomPluginMetric
    assert entry.default_threshold == 0.5
    assert entry.description == "Custom plugin metric."


@pytest.mark.unit
def test_catalog_loads_metric_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    class LazyMetric(BaseMetric):
        """Lazy plugin metric."""

        def __init__(self, threshold: float = 0.25) -> None:
            super().__init__(name="lazy_plugin_metric", dimension=Dimension.CORRECTNESS, threshold=threshold)

        def measure(self, eval_case: EvalCase) -> Score:
            return Score(name=self.name, value=1.0, threshold=self.threshold)

    class FakeEntryPoint:
        name = "lazy_plugin_metric"

        def load(self) -> type[BaseMetric]:
            return LazyMetric

    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda group: [FakeEntryPoint()] if group == "harness_evals.metrics" else [],
    )
    plugins._ENTRY_POINTS_DISCOVERED = False

    entry = next(item for item in catalog() if item.kind == "lazy_plugin_metric")

    assert entry.metric_class is LazyMetric
    assert plugins.metric("lazy_plugin_metric") is LazyMetric
