.PHONY: format lint test test-unit check install install-dev clean

install:
	pip install -e "."

install-dev:
	pip install -e ".[all,dev]"
	pre-commit install

format:
	ruff format src/ tests/ examples/
	ruff check --fix src/ tests/ examples/

lint:
	ruff check src/ tests/ examples/
	ruff format --check src/ tests/ examples/

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m unit

check: lint test

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
