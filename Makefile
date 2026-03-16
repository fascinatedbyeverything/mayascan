PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
NPM ?= npm
PY_LINT_PATHS = mayascan/__init__.py mayascan/_optional.py mayascan/detect.py mayascan/visualize.py mayascan/models/unet.py mayascan/models/dinov2.py tests/test_api.py tests/test_dinov2.py tests/test_integration.py scripts/type_coverage.py

.PHONY: install-dev install-foundation lint typecheck type-coverage test-collect test test-cov web-install web-check check-python check

install-dev:
	$(PIP) install -e ".[dev]"

install-foundation:
	$(PIP) install -e ".[dev,foundation]"

lint:
	$(PYTHON) -m ruff check $(PY_LINT_PATHS)

typecheck:
	$(PYTHON) -m mypy

type-coverage:
	$(PYTHON) scripts/type_coverage.py mayascan

test-collect:
	pytest --collect-only -q

test:
	pytest -q

test-cov:
	pytest --cov=mayascan --cov-report=term-missing --cov-report=xml

web-install:
	cd web && $(NPM) ci

web-check:
	cd web && $(NPM) run check

check-python: lint typecheck type-coverage test-collect test

check: check-python web-check
