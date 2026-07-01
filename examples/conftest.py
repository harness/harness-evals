"""Pytest configuration for integration examples."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integrations" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
