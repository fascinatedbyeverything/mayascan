# Contributing

## Setup

Python library and tests:

```bash
make install-dev
```

If you need the foundation-model stack as well:

```bash
make install-foundation
```

Browser app:

```bash
make web-install
```

## Feedback Loops

Use the smallest loop that answers the question you have:

```bash
make test-collect     # verify imports and discovery without running the suite
make test             # full Python test suite
make lint             # Ruff import/syntax/style checks
make typecheck        # mypy on the typed public/core modules
make type-coverage    # annotation coverage report for mayascan/
make web-check        # static browser app verification
make check-python     # full Python quality gate
make check            # Python + web checks
```

## Optional Dependencies

`segmentation-models-pytorch`, `transformers`, `peft`, `rasterio`, and `gradio`
remain optional from a development perspective. The package now imports those
dependencies lazily so commands like `pytest --collect-only` and most pure-Python
tests still work in a lightweight checkout.

The lint and typecheck gates are intentionally scoped to the public API and the
new agent-native support surfaces introduced in this pass. The broader legacy tree
still has backlog cleanup remaining, but the automated loop is now reliable and
green on the modules that define import behavior, typing metadata, and the browser
deployment workflow.

## Repo Layout

- `mayascan/`: Python package, CLI, models, and analysis utilities
- `tests/`: Python test suite
- `web/`: static browser application deployed separately
- `app.py`: legacy Gradio interface
