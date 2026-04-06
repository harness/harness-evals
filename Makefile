# Registry — defaults to public PyPI; override to publish elsewhere:
#   make publish                                                                     → public PyPI (default)
#   PYPI_REGISTRY=https://test.pypi.org/legacy/ PYPI_TOKEN=<token> make publish    → Test PyPI
#   PYPI_REGISTRY=https://harness0.harness.io/pkg/.../python PYPI_TOKEN=<token> make publish  → Harness HAR
PYPI_REGISTRY ?= https://upload.pypi.org/legacy/
PYPI_TOKEN    ?= $(HARNESS_PYPI_TOKEN)

# Build wheel and sdist
build:
	@echo "Building harness-evals..."
	pip3 install --upgrade build
	python3 -m build .
	@echo "Built artifacts in dist/"

# Publish to registry (public PyPI by default)
# Usage: PYPI_TOKEN=<token> make publish
publish: build
	@if [ -z "$(PYPI_TOKEN)" ]; then \
		echo "ERROR: PYPI_TOKEN is not set."; \
		echo "Usage: PYPI_TOKEN=<token> make publish"; \
		exit 1; \
	fi
	pip3 install --upgrade twine
	python3 -m twine upload \
		--repository-url "$(PYPI_REGISTRY)" \
		--username "token" \
		--password "$(PYPI_TOKEN)" \
		dist/*
	@echo "Published successfully to $(PYPI_REGISTRY)"

# Full release: bump patch + build + publish
# Usage: HARNESS_PYPI_TOKEN=<token> make release
release: bump-patch publish
	@echo "Released version: $$(make -s version)"

# Print current version
version:
	@python3 -c "import re; t=open('pyproject.toml').read(); print(re.search(r'version = \"([^\"]+)\"', t).group(1))"

# Increment patch version in pyproject.toml (e.g. 0.1.0 → 0.1.1)
bump-patch:
	@python3 -c "\
import re, pathlib; \
f = pathlib.Path('pyproject.toml'); t = f.read_text(); \
m = re.search(r'(version = \")(\d+\.\d+\.)(\d+)(\")', t); \
new = m.group(1) + m.group(2) + str(int(m.group(3)) + 1) + m.group(4); \
f.write_text(t.replace(m.group(0), new)); \
print('Bumped to', m.group(2) + str(int(m.group(3)) + 1))"

bump-minor:
	@python3 -c "\
import re, pathlib; \
f = pathlib.Path('pyproject.toml'); t = f.read_text(); \
m = re.search(r'(version = \")(\d+\.)(\d+)(\.\d+)(\")', t); \
new = m.group(1) + m.group(2) + str(int(m.group(3)) + 1) + '.0' + m.group(5); \
f.write_text(t.replace(m.group(0), new)); \
print('Bumped to', m.group(2) + str(int(m.group(3)) + 1) + '.0')"

# Install for development
install-dev:
	pip3 install -e ".[all,dev]"

# Run tests
test:
	pytest tests/ -v

# Lint
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

# Auto-fix lint
lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Clean build artifacts
clean:
	rm -rf dist/ *.egg-info src/*.egg-info build/

.PHONY: build publish release version bump-patch bump-minor install-dev test lint lint-fix clean
