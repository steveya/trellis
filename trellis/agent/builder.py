"""Code generation with test-fix loop."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path


TRELLIS_ROOT = Path(__file__).parent.parent


class WriteTargetEscapeError(RuntimeError):
    """Raised when a write target resolves outside the expected root."""


def validate_write_target(resolved_path: Path, root: Path, label: str) -> None:
    """Fail closed if *resolved_path* would land outside *root*.

    A relative path like ``../../etc/foo.py`` joined onto ``root`` then
    ``.resolve()``-d would escape the package tree.  ``mkdir(parents=True)``
    on the result would silently create the directory structure.  This guard
    stops that before any I/O.  Refs: QUA-382.
    """
    canonical = resolved_path.resolve()
    root_canonical = root.resolve()
    try:
        canonical.relative_to(root_canonical)
    except ValueError:
        raise WriteTargetEscapeError(
            f"QUA-382: {label} write target resolves outside the expected "
            f"root ({root_canonical}): {canonical}"
        )


# Underscored alias for callers that imported the private name.
_validate_write_target = validate_write_target


def write_module(relative_path: str, content: str) -> Path:
    """Write a module file into the trellis package.

    Fails closed if the resolved target would escape ``TRELLIS_ROOT``
    (e.g. a relative path containing ``../``).  Refs: QUA-382.
    """
    target = TRELLIS_ROOT / relative_path
    validate_write_target(target, TRELLIS_ROOT, "write_module")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def run_tests(test_path: str | None = None, max_retries: int = 3) -> dict:
    """Run pytest and return results. Returns dict with 'success' and 'output'."""
    project_root = TRELLIS_ROOT.parent
    cmd = [sys.executable, "-m", "pytest", "-x", "-q"]
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append(str(project_root / "tests"))

    for attempt in range(1, max_retries + 1):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            return {
                "success": True,
                "attempt": attempt,
                "output": result.stdout,
            }

    return {
        "success": False,
        "attempt": max_retries,
        "output": result.stdout + "\n" + result.stderr,
    }


def dynamic_import(file_path: Path, module_name: str):
    """Dynamically import a module from a file path.

    Returns the module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_agent_package():
    """Ensure trellis/instruments/_agent/ exists as a Python package."""
    agent_dir = TRELLIS_ROOT / "instruments" / "_agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    init_file = agent_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
    return agent_dir
