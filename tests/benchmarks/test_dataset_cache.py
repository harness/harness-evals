"""Tests for dataset caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_evals.benchmarks.dataset_cache import (
    get_cache_path,
    load_cached,
    save_to_cache,
)


@pytest.mark.unit
class TestDatasetCache:
    def test_get_cache_path(self, tmp_path: Path):
        path = get_cache_path("mmlu", "test", cache_dir=tmp_path)
        assert path == tmp_path / "mmlu" / "test.jsonl"

    def test_save_and_load(self, tmp_path: Path):
        items = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 2+2?", "answer": "4"},
        ]
        save_to_cache(items, "test_bench", "test", cache_dir=tmp_path)

        loaded = load_cached("test_bench", "test", cache_dir=tmp_path)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["question"] == "What is 1+1?"
        assert loaded[1]["answer"] == "4"

    def test_load_cached_missing(self, tmp_path: Path):
        result = load_cached("nonexistent", "test", cache_dir=tmp_path)
        assert result is None

    def test_save_creates_directories(self, tmp_path: Path):
        items = [{"key": "value"}]
        path = save_to_cache(items, "new_bench", "validation", cache_dir=tmp_path)
        assert path.exists()
        assert path.parent.name == "new_bench"

    def test_roundtrip_preserves_unicode(self, tmp_path: Path):
        items = [{"text": "日本語テスト", "emoji": "🎉"}]
        save_to_cache(items, "unicode", "test", cache_dir=tmp_path)
        loaded = load_cached("unicode", "test", cache_dir=tmp_path)
        assert loaded[0]["text"] == "日本語テスト"
        assert loaded[0]["emoji"] == "🎉"


@pytest.mark.unit
async def test_fetch_hf_dataset_offline_no_cache(tmp_path: Path):
    """Offline mode should raise when cache doesn't exist."""
    from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset

    with pytest.raises(FileNotFoundError, match="not found in cache"):
        await fetch_hf_dataset("test/dataset", "test", cache_dir=tmp_path, offline=True)


@pytest.mark.unit
async def test_fetch_hf_dataset_uses_cache(tmp_path: Path):
    """Should return cached data without making HTTP requests."""
    from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset

    items = [{"q": "test", "a": "answer"}]
    save_to_cache(items, "test__dataset", "test", cache_dir=tmp_path)

    result = await fetch_hf_dataset("test/dataset", "test", cache_dir=tmp_path, offline=True)
    assert result == items


@pytest.mark.unit
async def test_fetch_github_json_uses_cache(tmp_path: Path):
    from harness_evals.benchmarks.dataset_cache import fetch_github_json

    items = [{"id": "1", "target_label": "positive"}]
    save_to_cache(items, "github_test", "default", cache_dir=tmp_path)

    result = await fetch_github_json(
        "https://example.com/data.json",
        "github_test",
        cache_dir=tmp_path,
        offline=True,
    )
    assert result == items


@pytest.mark.unit
async def test_fetch_github_json_offline_no_cache(tmp_path: Path):
    from harness_evals.benchmarks.dataset_cache import fetch_github_json

    with pytest.raises(FileNotFoundError, match="not found in cache"):
        await fetch_github_json(
            "https://example.com/missing.json",
            "missing",
            cache_dir=tmp_path,
            offline=True,
        )
