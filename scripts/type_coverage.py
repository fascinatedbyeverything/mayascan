#!/usr/bin/env python3
"""Report annotation coverage for Python functions in a package tree."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def iter_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Return every argument node that should carry an annotation."""
    args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)
    return args


def count_annotated_functions(root: Path) -> tuple[int, int, int]:
    """Return file count, function count, and fully annotated function count."""
    file_count = 0
    function_count = 0
    fully_annotated = 0

    for path in sorted(root.rglob("*.py")):
        file_count += 1
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_count += 1
            args = iter_annotations(node)
            if node.returns is not None and all(arg.annotation is not None for arg in args):
                fully_annotated += 1

    return file_count, function_count, fully_annotated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report annotation coverage for a Python source tree.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="mayascan",
        help="Package or source directory to inspect.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    files, functions, annotated = count_annotated_functions(root)
    percent = (annotated / functions * 100.0) if functions else 0.0

    print(f"files: {files}")
    print(f"functions: {functions}")
    print(f"fully_annotated: {annotated}")
    print(f"coverage_pct: {percent:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
