"""Tests for dataset loading and saving."""

import json
from pathlib import Path

import pytest

from harness_evals import Golden, load_dataset, save_dataset


@pytest.mark.unit
class TestLoadDataset:
    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"input": "q1", "expected": "a1"}\n'
            '{"input": "q2", "expected": "a2"}\n'
        )
        dataset = load_dataset(str(f))
        assert len(dataset) == 2
        assert dataset[0].input == "q1"
        assert dataset[1].expected == "a2"

    def test_load_json_array(self, tmp_path):
        f = tmp_path / "data.json"
        data = [
            {"input": "q1", "expected": "a1"},
            {"input": "q2", "expected": "a2"},
        ]
        f.write_text(json.dumps(data))
        dataset = load_dataset(str(f), format="json")
        assert len(dataset) == 2
        assert dataset[0].input == "q1"

    def test_load_jsonl_skips_empty_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "q1"}\n\n\n{"input": "q2"}\n')
        dataset = load_dataset(str(f))
        assert len(dataset) == 2

    def test_load_jsonl_skips_malformed(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "q1"}\nnot json\n{"input": "q2"}\n')
        dataset = load_dataset(str(f))
        assert len(dataset) == 2

    def test_load_backward_compat_aliases(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "q", "expected_output": "a"}\n')
        dataset = load_dataset(str(f))
        assert dataset[0].expected == "a"

    def test_load_unsupported_format(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_dataset(str(f), format="csv")

    def test_load_json_not_array(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"input": "single"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_dataset(str(f), format="json")

    def test_load_with_context_and_tags(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "q", "expected": "a", "context": ["c1"], "tags": {"env": "ci"}}\n')
        dataset = load_dataset(str(f))
        assert dataset[0].context == ["c1"]
        assert dataset[0].tags == {"env": "ci"}


@pytest.mark.unit
class TestSaveDataset:
    def test_save_jsonl(self, tmp_path):
        dataset = [
            Golden(input="q1", expected="a1"),
            Golden(input="q2", expected="a2"),
        ]
        path = str(tmp_path / "out.jsonl")
        save_dataset(dataset, path)

        loaded = load_dataset(path)
        assert len(loaded) == 2
        assert loaded[0].input == "q1"
        assert loaded[1].expected == "a2"

    def test_save_json(self, tmp_path):
        dataset = [Golden(input="q1", expected="a1")]
        path = str(tmp_path / "out.json")
        save_dataset(dataset, path, format="json")

        loaded = load_dataset(path, format="json")
        assert len(loaded) == 1
        assert loaded[0].input == "q1"

    def test_save_creates_parent_dirs(self, tmp_path):
        dataset = [Golden(input="q")]
        path = str(tmp_path / "sub" / "dir" / "out.jsonl")
        save_dataset(dataset, path)
        assert Path(path).exists()

    def test_save_omits_none(self, tmp_path):
        dataset = [Golden(input="q")]
        path = str(tmp_path / "out.jsonl")
        save_dataset(dataset, path)

        line = Path(path).read_text().strip()
        obj = json.loads(line)
        assert obj == {"input": "q"}
        assert "expected" not in obj

    def test_roundtrip(self, tmp_path):
        original = [
            Golden(input="q1", expected="a1", context=["c1", "c2"], tags={"env": "prod"}),
            Golden(input="q2"),
        ]
        path = str(tmp_path / "roundtrip.jsonl")
        save_dataset(original, path)
        loaded = load_dataset(path)

        assert len(loaded) == 2
        assert loaded[0].input == "q1"
        assert loaded[0].context == ["c1", "c2"]
        assert loaded[0].tags == {"env": "prod"}
        assert loaded[1].expected is None
