"""Tests for importers base ABCs and plugin registration."""

from __future__ import annotations

import pytest

from harness_evals.importers.base import BaseEvalCaseSource, BaseEvalConfigSource
from harness_evals.refs import ResourceRef


@pytest.mark.unit
class TestBaseABCs:
    def test_base_eval_case_source_is_abstract(self):
        with pytest.raises(TypeError):
            BaseEvalCaseSource()  # type: ignore[abstract]

    def test_base_eval_config_source_is_abstract(self):
        with pytest.raises(TypeError):
            BaseEvalConfigSource()  # type: ignore[abstract]

    def test_concrete_eval_case_source_requires_fetch(self):
        class Minimal(BaseEvalCaseSource):
            name = "test"

            async def fetch(self, ref: ResourceRef) -> list:
                return []

        src = Minimal()
        assert src.name == "test"

    @pytest.mark.asyncio
    async def test_context_manager_protocol(self):
        class Minimal(BaseEvalCaseSource):
            name = "test"
            closed = False

            async def fetch(self, ref: ResourceRef) -> list:
                return []

            async def close(self) -> None:
                self.closed = True

        async with Minimal() as src:
            pass
        assert src.closed


@pytest.mark.unit
class TestPluginRegistration:
    def test_otel_registered(self):
        from harness_evals.importers.otel import OTELEvalCaseSource  # triggers @register
        from harness_evals.plugins import eval_case_source

        cls = eval_case_source("otel")
        assert cls is OTELEvalCaseSource

    def test_public_api_exports(self):
        import harness_evals

        assert hasattr(harness_evals, "BaseEvalCaseSource")
        assert hasattr(harness_evals, "BaseEvalConfigSource")
