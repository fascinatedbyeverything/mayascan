"""Pytest configuration — redirect temp files to external drive when available."""

import os
from pathlib import Path


def pytest_configure(config):
    """Use external drive for temp files if available."""
    ext = Path("/Volumes/macos4tb/.tmp/pytest")
    if ext.parent.exists():
        ext.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(ext.parent))
