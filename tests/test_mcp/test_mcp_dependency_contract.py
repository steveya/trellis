"""Packaging contract for the supported MCP transport dependency."""

from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _mcp_requirement(extra: str) -> Requirement:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["optional-dependencies"][extra]
    matches = [
        Requirement(dependency)
        for dependency in dependencies
        if Requirement(dependency).name == "mcp"
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("extra", ["mcp", "dev"])
def test_mcp_extras_exclude_unmigrated_major_version(extra: str) -> None:
    requirement = _mcp_requirement(extra)

    assert Version("1.27") in requirement.specifier
    assert Version("1.99.99") in requirement.specifier
    assert Version("2.0.0") not in requirement.specifier
