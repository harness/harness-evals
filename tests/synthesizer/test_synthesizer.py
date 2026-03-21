"""Tests for the synthesizer module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.runner import evaluate
from harness_evals.datasets import Dataset, load_dataset, save_dataset
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.deterministic.exact_match import ExactMatchMetric
from harness_evals.synthesizer import Synthesizer
from harness_evals.synthesizer.base import BaseSynthesizer
from harness_evals.synthesizer.extraction import ExtractionSynthesizer
from harness_evals.synthesizer.qa import QASynthesizer
from harness_evals.synthesizer.structured import StructuredOutputSynthesizer, _extract_json
from harness_evals.synthesizer.summarization import SummarizationSynthesizer
from tests.conftest import MockLLM


class TextMockLLM(BaseLLM):
    """Mock that returns JSON text from ``generate()`` for testing
    synthesizers that bypass ``generate_json()``."""

    def __init__(self, text_response: dict) -> None:
        self._text_response = text_response

    async def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps(self._text_response)

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return self._text_response


SAMPLE_DOC = (
    "Kubernetes is an open-source container orchestration platform. "
    "It automates deploying, scaling, and managing containerized applications.\n\n"
    "A Pod is the smallest deployable unit in Kubernetes. "
    "Each Pod can contain one or more containers.\n\n"
    "A Deployment provides declarative updates for Pods and ReplicaSets. "
    "You describe a desired state and the Deployment controller changes "
    "the actual state to match."
)


# ──────────────────────────────────────────────────────────────────────
# BaseSynthesizer — chunking, distribution, dedup
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestChunkDocuments:
    def test_short_doc_returns_as_is(self):
        chunks = BaseSynthesizer._chunk_documents(["short doc"], max_chars=5000)
        assert len(chunks) == 1
        assert chunks[0] == (0, "short doc")

    def test_splits_long_doc_at_paragraph_boundaries(self):
        para_a = "A" * 100
        para_b = "B" * 100
        para_c = "C" * 100
        doc = f"{para_a}\n\n{para_b}\n\n{para_c}"
        chunks = BaseSynthesizer._chunk_documents([doc], max_chars=210)
        assert len(chunks) >= 2
        for doc_idx, text in chunks:
            assert doc_idx == 0
            assert len(text) <= 210

    def test_multiple_docs_preserve_index(self):
        chunks = BaseSynthesizer._chunk_documents(["doc0", "doc1", "doc2"], max_chars=5000)
        assert len(chunks) == 3
        assert [idx for idx, _ in chunks] == [0, 1, 2]

    def test_skips_empty_documents(self):
        chunks = BaseSynthesizer._chunk_documents(["", "  ", "real content"], max_chars=5000)
        assert len(chunks) == 1
        assert chunks[0][1] == "real content"


@pytest.mark.unit
class TestDistribute:
    def test_proportional_distribution(self):
        chunks = [(0, "A" * 100), (1, "B" * 300)]
        alloc = BaseSynthesizer._distribute(20, chunks)
        assert sum(alloc) == 20
        assert alloc[1] > alloc[0]

    def test_single_chunk_gets_all(self):
        chunks = [(0, "only chunk")]
        alloc = BaseSynthesizer._distribute(10, chunks)
        assert alloc == [10]

    def test_more_chunks_than_n(self):
        chunks = [(i, f"chunk{i}") for i in range(20)]
        alloc = BaseSynthesizer._distribute(5, chunks)
        assert sum(alloc) == 5
        assert all(a <= 1 for a in alloc)

    def test_n_equals_chunks(self):
        chunks = [(0, "aaa"), (1, "bbb"), (2, "ccc")]
        alloc = BaseSynthesizer._distribute(3, chunks)
        assert sum(alloc) == 3
        assert all(a == 1 for a in alloc)

    def test_empty_chunks(self):
        alloc = BaseSynthesizer._distribute(10, [])
        assert alloc == []

    def test_n_one_multiple_chunks_picks_longest(self):
        chunks = [(0, "short"), (1, "a much longer chunk of text"), (2, "mid")]
        alloc = BaseSynthesizer._distribute(1, chunks)
        assert sum(alloc) == 1
        assert alloc[1] == 1


@pytest.mark.unit
class TestDeduplicate:
    def test_removes_exact_dupes(self):
        goldens = [
            Golden(input="What is K8s?", expected="container orchestration"),
            Golden(input="What is K8s?", expected="different answer"),
            Golden(input="What is a Pod?", expected="smallest unit"),
        ]
        result = BaseSynthesizer._deduplicate(goldens)
        assert len(result) == 2
        assert result[0].input == "What is K8s?"
        assert result[1].input == "What is a Pod?"


# ──────────────────────────────────────────────────────────────────────
# BaseSynthesizer — generate orchestration
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBaseSynthesizerGenerate:
    async def test_generate_produces_goldens(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "What is K8s?", "answer": "Orchestration platform", "difficulty": "easy"},
                    {"question": "What is a Pod?", "answer": "Smallest unit", "difficulty": "easy"},
                ]
            }
        )
        synth = QASynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=2, difficulty="easy")
        assert len(result) == 2
        assert all(isinstance(g, Golden) for g in result)

    async def test_handles_llm_failure_gracefully(self):
        class FailingLLM(MockLLM):
            async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
                raise RuntimeError("LLM is down")

        synth = QASynthesizer(FailingLLM())
        result = await synth.generate([SAMPLE_DOC], n=5, difficulty="easy")
        assert result == []

    async def test_generate_sync_works(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "What is K8s?", "answer": "Platform", "difficulty": "easy"},
                ]
            }
        )
        synth = QASynthesizer(llm)
        result = synth.generate_sync([SAMPLE_DOC], n=1, difficulty="easy")
        assert len(result) >= 1

    async def test_invalid_difficulty_raises(self):
        llm = MockLLM()
        synth = QASynthesizer(llm)
        with pytest.raises(ValueError, match="Invalid difficulty"):
            await synth.generate([SAMPLE_DOC], n=1, difficulty="nightmare")


# ──────────────────────────────────────────────────────────────────────
# QASynthesizer
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestQASynthesizer:
    async def test_generates_qa_goldens(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "What is Kubernetes?", "answer": "Container orchestration", "difficulty": "easy"},
                ]
            }
        )
        synth = QASynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="easy")
        assert len(result) == 1
        g = result[0]
        assert g.input == "What is Kubernetes?"
        assert g.expected == "Container orchestration"

    async def test_difficulty_in_metadata(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "Q?", "answer": "A", "difficulty": "hard"},
                ]
            }
        )
        synth = QASynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="hard")
        assert result[0].metadata["difficulty"] == "hard"
        assert result[0].metadata["task_type"] == "qa"

    async def test_handles_empty_pairs(self):
        llm = MockLLM(default={"pairs": []})
        synth = QASynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=5, difficulty="easy")
        assert result == []

    async def test_mixed_difficulty(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "Q1?", "answer": "A1", "difficulty": "easy"},
                    {"question": "Q2?", "answer": "A2", "difficulty": "hard"},
                ]
            }
        )
        synth = QASynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=2, difficulty="mixed")
        assert len(result) == 2

    async def test_skips_pairs_with_empty_fields(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "", "answer": "A", "difficulty": "easy"},
                    {"question": "Q?", "answer": "", "difficulty": "easy"},
                    {"question": "Valid?", "answer": "Yes", "difficulty": "easy"},
                ]
            }
        )
        synth = QASynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=3, difficulty="easy")
        assert len(result) == 1
        assert result[0].input == "Valid?"


# ──────────────────────────────────────────────────────────────────────
# SummarizationSynthesizer
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSummarizationSynthesizer:
    async def test_generates_summarization_goldens(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {
                        "passage": "K8s automates deploying and scaling apps.",
                        "summary": "K8s handles deployment and scaling.",
                        "difficulty": "easy",
                    },
                ]
            }
        )
        synth = SummarizationSynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="easy")
        assert len(result) == 1
        assert result[0].metadata["task_type"] == "summarization"

    async def test_passage_as_input_summary_as_expected(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {
                        "passage": "Some passage to summarize.",
                        "summary": "The summary.",
                        "difficulty": "medium",
                    },
                ]
            }
        )
        synth = SummarizationSynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="medium")
        assert result[0].input == "Some passage to summarize."
        assert result[0].expected == "The summary."


# ──────────────────────────────────────────────────────────────────────
# ExtractionSynthesizer
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractionSynthesizer:
    async def test_generates_extraction_goldens(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {
                        "text": "K8s uses Pods and Deployments.",
                        "extracted": "Pods, Deployments",
                        "difficulty": "easy",
                    },
                ]
            }
        )
        synth = ExtractionSynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="easy")
        assert len(result) == 1
        assert result[0].metadata["task_type"] == "extraction"

    async def test_text_as_input_extracted_as_expected(self):
        llm = MockLLM(
            default={
                "pairs": [
                    {
                        "text": "The input text.",
                        "extracted": "Extracted info",
                        "difficulty": "medium",
                    },
                ]
            }
        )
        synth = ExtractionSynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="medium")
        assert result[0].input == "The input text."
        assert result[0].expected == "Extracted info"


# ──────────────────────────────────────────────────────────────────────
# StructuredOutputSynthesizer
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestStructuredOutputSynthesizer:
    async def test_generates_structured_goldens(self):
        llm = TextMockLLM(
            text_response={
                "pairs": [
                    {
                        "prompt": "Generate a K8s Pod spec for nginx",
                        "expected_output": {"apiVersion": "v1", "kind": "Pod"},
                        "difficulty": "medium",
                    },
                ]
            }
        )
        synth = StructuredOutputSynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="medium")
        assert len(result) == 1
        assert result[0].metadata["task_type"] == "structured_output"

    async def test_expected_is_dict(self):
        llm = TextMockLLM(
            text_response={
                "pairs": [
                    {
                        "prompt": "Generate config",
                        "expected_output": {"key": "value"},
                        "difficulty": "easy",
                    },
                ]
            }
        )
        synth = StructuredOutputSynthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, difficulty="easy")
        assert isinstance(result[0].expected, dict)
        assert result[0].expected == {"key": "value"}

    async def test_uses_generate_not_generate_json(self):
        """Verify StructuredOutputSynthesizer calls generate(), not generate_json()."""

        class TrackingLLM(BaseLLM):
            def __init__(self):
                self.generate_called = False
                self.generate_json_called = False

            async def generate(self, prompt: str, **kwargs) -> str:
                self.generate_called = True
                return json.dumps({"pairs": [{"prompt": "P", "expected_output": {"a": 1}, "difficulty": "easy"}]})

            async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
                self.generate_json_called = True
                return {}

        llm = TrackingLLM()
        synth = StructuredOutputSynthesizer(llm)
        await synth.generate([SAMPLE_DOC], n=1, difficulty="easy")
        assert llm.generate_called
        assert not llm.generate_json_called


# ──────────────────────────────────────────────────────────────────────
# _extract_json helper
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fences(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_trailing_text(self):
        text = '{"a": 1} here is some explanation'
        assert _extract_json(text) == {"a": 1}

    def test_preamble_and_json(self):
        text = 'Here is the output:\n{"key": "value"}'
        assert _extract_json(text) == {"key": "value"}

    def test_markdown_fences_with_trailing(self):
        text = '```json\n{"a": 1}\n```\nHope that helps!'
        assert _extract_json(text) == {"a": 1}

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError, match="No JSON object found"):
            _extract_json("no json here")

    def test_array_raises(self):
        with pytest.raises(json.JSONDecodeError, match="No JSON object found"):
            _extract_json("[1, 2, 3]")

    def test_nested_objects(self):
        text = '{"outer": {"inner": {"deep": true}}}'
        assert _extract_json(text) == {"outer": {"inner": {"deep": True}}}


# ──────────────────────────────────────────────────────────────────────
# Synthesizer facade
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSynthesizerFacade:
    async def test_dispatches_to_qa(self):
        llm = MockLLM(default={"pairs": [{"question": "Q?", "answer": "A", "difficulty": "easy"}]})
        synth = Synthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, task_type="qa", difficulty="easy")
        assert len(result) >= 1
        assert result[0].metadata["task_type"] == "qa"

    async def test_dispatches_to_summarization(self):
        llm = MockLLM(default={"pairs": [{"passage": "Text.", "summary": "Summary.", "difficulty": "easy"}]})
        synth = Synthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, task_type="summarization")
        assert len(result) >= 1
        assert result[0].metadata["task_type"] == "summarization"

    async def test_dispatches_to_extraction(self):
        llm = MockLLM(default={"pairs": [{"text": "Input.", "extracted": "Output.", "difficulty": "easy"}]})
        synth = Synthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, task_type="extraction")
        assert len(result) >= 1
        assert result[0].metadata["task_type"] == "extraction"

    async def test_dispatches_to_structured_output(self):
        llm = TextMockLLM(
            text_response={
                "pairs": [
                    {
                        "prompt": "Generate JSON",
                        "expected_output": {"a": 1},
                        "difficulty": "easy",
                    }
                ]
            }
        )
        synth = Synthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, task_type="structured_output")
        assert len(result) >= 1
        assert result[0].metadata["task_type"] == "structured_output"

    async def test_invalid_task_type_raises(self):
        llm = MockLLM()
        synth = Synthesizer(llm)
        with pytest.raises(ValueError, match="Unknown task_type"):
            await synth.generate([SAMPLE_DOC], n=1, task_type="nonexistent")

    async def test_invalid_difficulty_raises(self):
        llm = MockLLM()
        synth = Synthesizer(llm)
        with pytest.raises(ValueError, match="Invalid difficulty"):
            await synth.generate([SAMPLE_DOC], n=1, task_type="qa", difficulty="impossible")

    async def test_generate_sync_facade(self):
        llm = MockLLM(default={"pairs": [{"question": "Q?", "answer": "A", "difficulty": "easy"}]})
        synth = Synthesizer(llm)
        result = synth.generate_sync([SAMPLE_DOC], n=1, task_type="qa")
        assert len(result) >= 1

    async def test_returns_dataset_type(self):
        llm = MockLLM(default={"pairs": [{"question": "Q?", "answer": "A", "difficulty": "easy"}]})
        synth = Synthesizer(llm)
        result = await synth.generate([SAMPLE_DOC], n=1, task_type="qa")
        assert isinstance(result, list)
        assert all(isinstance(g, Golden) for g in result)


# ──────────────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestSynthesizerIntegration:
    async def test_synthesize_save_load_roundtrip(self, tmp_path: Path):
        """Generate goldens, save to JSONL, load back, verify match."""
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "What is K8s?", "answer": "Orchestration", "difficulty": "easy"},
                    {"question": "What is a Pod?", "answer": "Smallest unit", "difficulty": "medium"},
                ]
            }
        )
        synth = Synthesizer(llm)
        generated: Dataset = await synth.generate([SAMPLE_DOC], n=2, task_type="qa")
        assert len(generated) >= 1

        path = str(tmp_path / "generated.jsonl")
        save_dataset(generated, path)
        loaded = load_dataset(path)

        assert len(loaded) == len(generated)
        for orig, loaded_g in zip(generated, loaded, strict=True):
            assert orig.input == loaded_g.input
            assert orig.expected == loaded_g.expected

    async def test_synthesize_then_evaluate(self):
        """Generate goldens, create EvalCases, run evaluate, verify scores."""
        llm = MockLLM(
            default={
                "pairs": [
                    {"question": "What is K8s?", "answer": "Orchestration", "difficulty": "easy"},
                ]
            }
        )
        synth = Synthesizer(llm)
        goldens: Dataset = await synth.generate([SAMPLE_DOC], n=1, task_type="qa")
        assert len(goldens) >= 1

        golden = goldens[0]
        ec = EvalCase.from_golden(golden, output=golden.expected)
        scores = evaluate(ec, metrics=[ExactMatchMetric()])

        assert len(scores) == 1
        assert scores[0].passed
        assert scores[0].value == 1.0
