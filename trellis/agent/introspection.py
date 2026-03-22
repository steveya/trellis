"""AST-based library inspection for the agent."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any


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


def _class_signature(cls) -> str:
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
