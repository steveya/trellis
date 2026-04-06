"""Compatibility audit for deprecated lattice builder usage in tests."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPRECATED_BUILDERS = {"build_rate_lattice", "build_spot_lattice"}
NON_LEGACY_LATTICE_TESTS = (
    "tests/test_models/test_generalized_methods.py",
    "tests/test_verification/test_numerical_calibration.py",
    "tests/test_verification/test_literature_benchmarks.py",
)
TREE_LATTICE_SUITE = REPO_ROOT / "tests/test_models/test_trees/test_lattice.py"


def _builder_calls(module: ast.AST) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in DEPRECATED_BUILDERS:
            calls.append((int(node.lineno), node.func.id))
    return calls


def _has_legacy_compat_marker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if not isinstance(target, ast.Attribute) or target.attr != "legacy_compat":
            continue
        value = target.value
        if isinstance(value, ast.Attribute) and value.attr == "mark":
            return True
    return False


def test_non_legacy_lattice_suites_do_not_call_deprecated_builders():
    offenders: list[str] = []
    for rel_path in NON_LEGACY_LATTICE_TESTS:
        module = ast.parse((REPO_ROOT / rel_path).read_text())
        offenders.extend(
            f"{rel_path}:{lineno}:{name}"
            for lineno, name in _builder_calls(module)
        )

    assert offenders == []


def test_tree_lattice_suite_limits_deprecated_builders_to_legacy_compat_tests():
    module = ast.parse(TREE_LATTICE_SUITE.read_text())
    offenders: list[str] = []
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = {name for _, name in _builder_calls(node)}
        if calls and not _has_legacy_compat_marker(node):
            offenders.append(f"{node.name}:{','.join(sorted(calls))}")

    assert offenders == []
