"""AST-based library inspection for the agent."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = PROJECT_ROOT / "tests"
LESSONS_ROOT = PACKAGE_ROOT / "agent" / "knowledge" / "lessons"
TRACES_ROOT = PACKAGE_ROOT / "agent" / "knowledge" / "traces"


def get_package_tree(package_name: str = "trellis") -> dict[str, Any]:
    """Walk the trellis package and return a tree of modules and their public symbols."""
    pkg = importlib.import_module(package_name)
    pkg_path = Path(pkg.__file__).parent
    tree: dict[str, Any] = {}

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=[str(pkg_path)],
        prefix=package_name + ".",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            tree[modname] = {"error": "import failed"}
            continue

        symbols: dict[str, str] = {}
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isclass(obj):
                sig = _class_signature(obj)
                symbols[name] = sig
            elif inspect.isfunction(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (ValueError, TypeError):
                    sig = "(…)"
                symbols[name] = f"def {name}{sig}"

        tree[modname] = {"is_package": ispkg, "symbols": symbols}

    return tree


def read_module_source(module_path: str) -> str:
    """Read the source code of a module by dotted path."""
    mod = importlib.import_module(module_path)
    return inspect.getsource(mod)


def list_module_exports(module_path: str) -> list[dict[str, str]]:
    """Return public classes/functions exported by a module.

    The output is JSON-friendly for tool responses.
    """
    mod = importlib.import_module(module_path)
    exports: list[dict[str, str]] = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isclass(obj):
            if getattr(obj, "__module__", "") != module_path:
                continue
            exports.append({
                "name": name,
                "kind": "class",
                "signature": _class_signature(obj),
            })
        elif inspect.isfunction(obj):
            if getattr(obj, "__module__", "") != module_path:
                continue
            try:
                signature = str(inspect.signature(obj))
            except (ValueError, TypeError):
                signature = "(…)"
            exports.append({
                "name": name,
                "kind": "function",
                "signature": f"def {name}{signature}",
            })
    return sorted(exports, key=lambda item: (item["kind"], item["name"]))


def find_symbol(symbol: str) -> list[dict[str, str]]:
    """Find a symbol across the registry-backed package surface."""
    from trellis.agent.knowledge.import_registry import find_symbol_modules

    matches: list[dict[str, str]] = []
    for module_path in find_symbol_modules(symbol):
        try:
            mod = importlib.import_module(module_path)
            obj = getattr(mod, symbol)
        except Exception:
            continue

        if inspect.isclass(obj):
            signature = _class_signature(obj)
            kind = "class"
        elif inspect.isfunction(obj):
            try:
                signature = f"def {symbol}{inspect.signature(obj)}"
            except (ValueError, TypeError):
                signature = f"def {symbol}(…)"
            kind = "function"
        else:
            signature = type(obj).__name__
            kind = "other"

        matches.append({
            "module": module_path,
            "symbol": symbol,
            "kind": kind,
            "signature": signature,
        })
    return matches


def search_package(pattern: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the live trellis package source tree."""
    return _search_paths(pattern, [PACKAGE_ROOT], limit=limit, include_suffixes={".py"})


def search_tests(pattern: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search tests for a pattern."""
    return _search_paths(pattern, [TESTS_ROOT], limit=limit, include_suffixes={".py"})


def search_lessons(pattern: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search lessons and trace artifacts for a pattern."""
    return _search_paths(
        pattern,
        [LESSONS_ROOT, TRACES_ROOT],
        limit=limit,
        include_suffixes={".py", ".md", ".yaml", ".yml", ".json"},
    )


def _class_signature(cls) -> str:
    """Build a compact printable signature summary for a class."""
    bases = ", ".join(b.__name__ for b in cls.__bases__ if b is not object)
    try:
        init_sig = str(inspect.signature(cls.__init__))
    except (ValueError, TypeError):
        init_sig = "(…)"
    base_str = f"({bases})" if bases else ""
    methods = [
        name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    return f"class {cls.__name__}{base_str}.__init__{init_sig} methods=[{', '.join(methods)}]"


def _search_paths(
    pattern: str,
    roots: list[Path],
    *,
    limit: int,
    include_suffixes: set[str],
) -> list[dict[str, Any]]:
    """Search files under ``roots`` and return ranked snippet matches."""
    matcher = _build_matcher(pattern)
    results: list[dict[str, Any]] = []
    lowered = pattern.lower()
    simple_symbol = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern))

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in include_suffixes:
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                lines = path.read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if matcher(line):
                    results.append({
                        "path": str(path),
                        "line": line_number,
                        "snippet": line.strip(),
                    })
    results.sort(
        key=lambda item: _search_result_rank(
            item,
            lowered=lowered,
            simple_symbol=simple_symbol,
        )
    )
    return results[:limit]


def _search_result_rank(
    item: dict[str, Any],
    *,
    lowered: str,
    simple_symbol: bool,
) -> tuple[int, str, int]:
    """Rank search hits so likely definitions beat incidental references."""
    path = item["path"]
    snippet = item["snippet"].strip()
    snippet_lower = snippet.lower()

    priority = 50
    if simple_symbol:
        if snippet.startswith(f"def {lowered}") or snippet.startswith(f"class {lowered}"):
            priority = 0
        elif snippet_lower.startswith(f"def {lowered}") or snippet_lower.startswith(f"class {lowered}"):
            priority = 0
        elif f"def {lowered}" in snippet_lower or f"class {lowered}" in snippet_lower:
            priority = 5
        elif lowered in Path(path).stem.lower():
            priority = 10
    if "/models/" in path:
        priority -= 2
    elif "/core/" in path:
        priority += 2
    elif "/agent/" in path:
        priority += 4
    return (priority, path, item["line"])


def _build_matcher(pattern: str):
    """Compile a regex matcher, falling back to case-insensitive substring search."""
    try:
        regex = re.compile(pattern, flags=re.IGNORECASE)
    except re.error:
        lowered = pattern.lower()
        return lambda text: lowered in text.lower()
    return lambda text: bool(regex.search(text))
