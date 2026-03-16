"""Helpers for dependencies that may be absent in lightweight environments."""

from __future__ import annotations

from importlib import import_module


def import_optional(
    module_name: str,
    *,
    feature: str,
    install_hint: str,
):
    """Import *module_name* or raise an actionable ImportError."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires `{module_name}`. Install it with: {install_hint}"
        ) from exc
