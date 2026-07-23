"""Tests for ROUGEMetric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.similarity.rouge import ROUGEMetric


class TestROUGEMetric:
    @pytest.mark.unit
    def test_rouge_perfect_match(self):
        ec = EvalCase(input="q", output="the cat sat on the mat", expected="the cat sat on the mat")
        score = ROUGEMetric(threshold=0.5).measure(ec)
        assert score.passed
        assert score.value == 1.0

    @pytest.mark.unit
    def test_rouge_partial_match(self):
        ec = EvalCase(input="q", output="the cat sat", expected="the cat sat on the mat")
        score = ROUGEMetric(threshold=0.3, variant="rouge-1").measure(ec)
        assert 0.0 < score.value < 1.0

    @pytest.mark.unit
    def test_rouge_no_overlap(self):
        ec = EvalCase(input="q", output="dogs run fast", expected="the cat sat on the mat")
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert score.value == 0.0
        assert not score.passed

    @pytest.mark.unit
    def test_rouge_no_expected(self):
        ec = EvalCase(input="q", output="hello world", expected=None)
        score = ROUGEMetric().measure(ec)
        assert score.value == 0.0
        assert "expected" in score.reason.lower()

    @pytest.mark.unit
    def test_rouge_empty_expected(self):
        ec = EvalCase(input="q", output="hello world", expected="")
        score = ROUGEMetric().measure(ec)
        assert score.value == 0.0

    @pytest.mark.unit
    def test_rouge_metadata_fields(self):
        ec = EvalCase(input="q", output="the cat sat on the mat", expected="the cat sat on the mat")
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert score.metadata is not None
        assert score.metadata["variant"] == "rouge-1"
        assert "precision" in score.metadata
        assert "recall" in score.metadata
        assert "fmeasure" in score.metadata
        assert score.metadata["precision"] == 1.0
        assert score.metadata["recall"] == 1.0
        assert score.metadata["fmeasure"] == 1.0

    @pytest.mark.unit
    def test_rouge_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Invalid variant"):
            ROUGEMetric(variant="rouge-3")

    @pytest.mark.unit
    def test_rouge2_bigram_overlap(self):
        ec = EvalCase(input="q", output="the cat sat on a mat", expected="the cat sat on the mat")
        score = ROUGEMetric(variant="rouge-2").measure(ec)
        assert 0.0 < score.value < 1.0

    @pytest.mark.unit
    def test_all_variants_produce_scores(self):
        ec = EvalCase(input="q", output="the big cat sat", expected="the cat sat on the mat quietly")
        scores = {}
        for v in ["rouge-1", "rouge-2", "rouge-l"]:
            scores[v] = ROUGEMetric(variant=v).measure(ec).value
        assert all(0.0 < s < 1.0 for s in scores.values())

    @pytest.mark.unit
    def test_rouge_l_perfect(self):
        ec = EvalCase(input="q", output="hello world foo bar", expected="hello world foo bar")
        score = ROUGEMetric(variant="rouge-l").measure(ec)
        assert score.value == 1.0

    @pytest.mark.unit
    def test_rouge_l_partial_subsequence(self):
        ec = EvalCase(input="q", output="the cat on mat", expected="the cat sat on the mat")
        score = ROUGEMetric(variant="rouge-l").measure(ec)
        assert 0.0 < score.value < 1.0

    @pytest.mark.unit
    def test_rouge_1_recall_vs_precision(self):
        ec = EvalCase(input="q", output="the cat sat on the mat is soft", expected="the cat sat on the mat")
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert score.metadata["recall"] == 1.0
        assert score.metadata["precision"] < 1.0

    @pytest.mark.unit
    def test_rouge_threshold_pass(self):
        ec = EvalCase(input="q", output="the cat sat on the mat", expected="the cat sat on the mat")
        score = ROUGEMetric(threshold=0.9, variant="rouge-1").measure(ec)
        assert score.passed

    @pytest.mark.unit
    def test_rouge_threshold_fail(self):
        ec = EvalCase(input="q", output="dogs run fast", expected="the cat sat on the mat")
        score = ROUGEMetric(threshold=0.5, variant="rouge-1").measure(ec)
        assert not score.passed

    @pytest.mark.unit
    def test_rouge_empty_output(self):
        ec = EvalCase(input="q", output="", expected="the cat sat on the mat")
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert score.value == 0.0

    @pytest.mark.unit
    def test_rouge_single_word_match(self):
        ec = EvalCase(input="q", output="cat", expected="cat")
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert score.value == 1.0

    @pytest.mark.unit
    def test_rouge_2_no_bigrams_short_text(self):
        ec = EvalCase(input="q", output="hello", expected="hello")
        score = ROUGEMetric(variant="rouge-2").measure(ec)
        assert score.value == 0.0

    @pytest.mark.unit
    def test_rouge_l_no_common_subsequence(self):
        ec = EvalCase(input="q", output="x y z", expected="a b c")
        score = ROUGEMetric(variant="rouge-l").measure(ec)
        assert score.value == 0.0

    @pytest.mark.unit
    def test_rouge_score_name(self):
        ec = EvalCase(input="q", output="hello", expected="hello")
        score = ROUGEMetric().measure(ec)
        assert score.name == "rouge"

    @pytest.mark.unit
    def test_rouge_default_variant_is_rouge_l(self):
        metric = ROUGEMetric()
        assert metric.variant == "rouge-l"

    @pytest.mark.unit
    def test_rouge_default_threshold(self):
        metric = ROUGEMetric()
        assert metric.threshold == 0.5

    @pytest.mark.unit
    def test_rouge_long_text(self):
        ref = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day"
        hyp = "the quick brown fox leaps over a lazy dog by the river bank on a sunny afternoon"
        ec = EvalCase(input="q", output=hyp, expected=ref)
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert 0.5 < score.value < 1.0

    @pytest.mark.unit
    def test_rouge_2_perfect(self):
        ec = EvalCase(input="q", output="the cat sat", expected="the cat sat")
        score = ROUGEMetric(variant="rouge-2").measure(ec)
        assert score.value == 1.0

    @pytest.mark.unit
    def test_rouge_reason_contains_variant(self):
        ec = EvalCase(input="q", output="hello world", expected="hello world")
        score = ROUGEMetric(variant="rouge-2").measure(ec)
        assert "rouge-2" in score.reason

    @pytest.mark.unit
    def test_rouge_dict_input_stringified(self):
        ec = EvalCase(input="q", output="{'key': 'value'}", expected="{'key': 'value'}")
        score = ROUGEMetric(variant="rouge-1").measure(ec)
        assert score.value == 1.0

    @pytest.mark.unit
    def test_rouge_symmetric_for_identical(self):
        text = "the cat sat on the mat"
        ec1 = EvalCase(input="q", output=text, expected=text)
        score = ROUGEMetric(variant="rouge-1").measure(ec1)
        assert score.metadata["precision"] == score.metadata["recall"]

    @pytest.mark.unit
    def test_rouge_variants_differ_for_partial_match(self):
        ec = EvalCase(input="q", output="the cat jumped over", expected="the cat sat on the mat by the window")
        scores = {v: ROUGEMetric(variant=v).measure(ec).value for v in ["rouge-1", "rouge-2", "rouge-l"]}
        unique_scores = set(round(s, 6) for s in scores.values())
        assert len(unique_scores) >= 2
