"""Deterministic inventory of route-helper authority and adapter delegation."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import yaml


_ROUTES_PATH = Path("trellis/agent/knowledge/canonical/routes.yaml")
_BINDINGS_PATH = Path(
    "trellis/agent/knowledge/canonical/backend_bindings.yaml"
)
_ADAPTER_ROOT = Path("trellis/instruments/_agent")


@dataclass(frozen=True, order=True)
class HelperAuthorityReference:
    """One required route-helper declaration under an explicit condition."""

    route_id: str
    condition: str
    module: str
    symbol: str
    required: bool = True

    @property
    def identity(self) -> tuple[str, str, str, str, bool]:
        """Return the source-independent identity used for drift comparison."""
        return (
            self.route_id,
            self.condition,
            self.module,
            self.symbol,
            self.required,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True, order=True)
class AdapterDelegationCall:
    """One imported pricing or authoritative call in a checked-in adapter."""

    path: str
    line: int
    local_name: str
    module: str
    symbol: str
    is_price_call: bool
    matches_required_authority: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class HelperAuthorityReport:
    """Machine-readable helper-authority inventory for one repository state."""

    schema_version: int
    promoted_route_count: int
    route_authority: tuple[HelperAuthorityReference, ...]
    binding_authority: tuple[HelperAuthorityReference, ...]
    route_only_authority: tuple[HelperAuthorityReference, ...]
    binding_only_authority: tuple[HelperAuthorityReference, ...]
    adapter_calls: tuple[AdapterDelegationCall, ...]

    @property
    def has_route_binding_drift(self) -> bool:
        """Return whether canonical routes and exact bindings disagree."""
        return bool(self.route_only_authority or self.binding_only_authority)

    @property
    def summary(self) -> dict[str, int]:
        """Return stable low-cardinality counts for comparisons over time."""
        route_ids = {item.route_id for item in self.route_authority}
        binding_ids = {item.route_id for item in self.binding_authority}
        price_calls = tuple(item for item in self.adapter_calls if item.is_price_call)
        price_call_paths = {item.path for item in price_calls}
        authority_calls = tuple(
            item for item in self.adapter_calls if item.matches_required_authority
        )
        authority_call_paths = {item.path for item in authority_calls}
        return {
            "promoted_route_count": self.promoted_route_count,
            "route_authority_route_count": len(route_ids),
            "route_authority_reference_count": len(self.route_authority),
            "binding_authority_route_count": len(binding_ids),
            "binding_authority_reference_count": len(self.binding_authority),
            "route_only_reference_count": len(self.route_only_authority),
            "binding_only_reference_count": len(self.binding_only_authority),
            "adapter_price_call_file_count": len(price_call_paths),
            "adapter_price_call_count": len(price_calls),
            "adapter_authority_call_file_count": len(authority_call_paths),
            "adapter_authority_call_count": len(authority_calls),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON payload emitted by the audit CLI."""
        return {
            "schema_version": self.schema_version,
            "summary": self.summary,
            "route_authority": [item.to_dict() for item in self.route_authority],
            "binding_authority": [
                item.to_dict() for item in self.binding_authority
            ],
            "drift": {
                "route_only": [
                    item.to_dict() for item in self.route_only_authority
                ],
                "binding_only": [
                    item.to_dict() for item in self.binding_only_authority
                ],
            },
            "adapter_calls": [
                item.to_dict() for item in self.adapter_calls
            ],
        }


def build_helper_authority_report(root: str | Path) -> HelperAuthorityReport:
    """Build a deterministic helper-authority report from repository files."""
    repo_root = Path(root).resolve()
    routes = _load_manifest(repo_root / _ROUTES_PATH, key="routes")
    bindings = _load_manifest(repo_root / _BINDINGS_PATH, key="bindings")

    promoted_routes = tuple(
        entry for entry in routes if str(entry.get("status") or "") == "promoted"
    )
    promoted_route_ids = {
        str(entry.get("id") or "").strip() for entry in promoted_routes
    }
    promoted_bindings = tuple(
        entry
        for entry in bindings
        if str(entry.get("route_id") or "").strip() in promoted_route_ids
    )
    route_authority = _collect_authority(
        promoted_routes,
        route_id_key="id",
    )
    binding_authority = _collect_authority(
        promoted_bindings,
        route_id_key="route_id",
    )
    route_only, binding_only = _authority_drift(
        route_authority,
        binding_authority,
    )
    authority_symbols = {
        item.symbol for item in route_authority + binding_authority
    }
    adapter_calls = _scan_adapter_calls(
        repo_root,
        authority_symbols=authority_symbols,
    )
    return HelperAuthorityReport(
        schema_version=1,
        promoted_route_count=len(promoted_routes),
        route_authority=route_authority,
        binding_authority=binding_authority,
        route_only_authority=route_only,
        binding_only_authority=binding_only,
        adapter_calls=adapter_calls,
    )


def render_helper_authority_report(report: HelperAuthorityReport) -> str:
    """Render the inventory as deterministic human-readable text."""
    summary = report.summary
    lines = [
        "Helper authority audit",
        f"schema_version={report.schema_version}",
        f"promoted_routes={summary['promoted_route_count']}",
        f"route_authority_routes={summary['route_authority_route_count']}",
        f"route_authority_references={summary['route_authority_reference_count']}",
        f"binding_authority_routes={summary['binding_authority_route_count']}",
        f"binding_authority_references={summary['binding_authority_reference_count']}",
        f"route_only_references={summary['route_only_reference_count']}",
        f"binding_only_references={summary['binding_only_reference_count']}",
        f"adapter_price_call_files={summary['adapter_price_call_file_count']}",
        f"adapter_price_calls={summary['adapter_price_call_count']}",
        f"adapter_authority_call_files={summary['adapter_authority_call_file_count']}",
        f"adapter_authority_calls={summary['adapter_authority_call_count']}",
    ]
    _append_authority_section(lines, "Route authority", report.route_authority)
    _append_authority_section(lines, "Binding authority", report.binding_authority)
    _append_authority_section(
        lines,
        "Route-only authority drift",
        report.route_only_authority,
    )
    _append_authority_section(
        lines,
        "Binding-only authority drift",
        report.binding_only_authority,
    )
    lines.append("")
    lines.append("Adapter imported pricing and authority calls")
    if not report.adapter_calls:
        lines.append("- none")
    else:
        for item in report.adapter_calls:
            marker = "authority" if item.matches_required_authority else "price-call"
            lines.append(
                f"- [{marker}] {item.path}:{item.line} "
                f"{item.module}.{item.symbol} as {item.local_name}"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``scripts/audit_helper_authority.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root containing canonical knowledge and adapters.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the versioned machine-readable report.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Return exit code 1 when route and binding authority differ.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the helper-authority audit CLI."""
    args = build_parser().parse_args(argv)
    report = build_helper_authority_report(args.root)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_helper_authority_report(report), end="")
    if args.fail_on_drift and report.has_route_binding_drift:
        return 1
    return 0


def _load_manifest(path: Path, *, key: str) -> tuple[Mapping[str, object], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Helper-authority audit requires {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Helper-authority manifest {path} must be a mapping")
    entries = payload.get(key)
    if not isinstance(entries, list):
        raise ValueError(f"Helper-authority manifest {path} requires list {key!r}")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError(f"Helper-authority manifest {path} has non-mapping entries")
    return tuple(entries)


def _collect_authority(
    entries: Iterable[Mapping[str, object]],
    *,
    route_id_key: str,
) -> tuple[HelperAuthorityReference, ...]:
    authority: list[HelperAuthorityReference] = []
    for entry in entries:
        route_id = str(entry.get(route_id_key) or "").strip()
        if not route_id:
            raise ValueError(f"Helper-authority entry requires {route_id_key!r}")
        authority.extend(
            _authority_from_primitives(
                route_id,
                "base",
                entry.get("primitives"),
            )
        )
        conditional = entry.get("conditional_primitives") or ()
        if not isinstance(conditional, (list, tuple)):
            raise ValueError(
                f"Helper-authority route {route_id!r} conditional_primitives must be a list"
            )
        for block in conditional:
            if not isinstance(block, Mapping):
                raise ValueError(
                    f"Helper-authority route {route_id!r} has invalid conditional block"
                )
            condition = json.dumps(
                block.get("when"),
                sort_keys=True,
                separators=(",", ":"),
            )
            authority.extend(
                _authority_from_primitives(
                    route_id,
                    condition,
                    block.get("primitives"),
                )
            )
    return tuple(sorted(authority, key=_authority_sort_key))


def _authority_from_primitives(
    route_id: str,
    condition: str,
    raw_primitives: object,
) -> list[HelperAuthorityReference]:
    primitives = raw_primitives or ()
    if not isinstance(primitives, (list, tuple)):
        raise ValueError(
            f"Helper-authority route {route_id!r} primitives must be a list"
        )
    authority: list[HelperAuthorityReference] = []
    for primitive in primitives:
        if not isinstance(primitive, Mapping):
            raise ValueError(
                f"Helper-authority route {route_id!r} has invalid primitive"
            )
        if str(primitive.get("role") or "") != "route_helper":
            continue
        required = bool(primitive.get("required", True))
        if not required:
            continue
        module = str(primitive.get("module") or "").strip()
        symbol = str(primitive.get("symbol") or "").strip()
        if not module or not symbol:
            raise ValueError(
                f"Required route helper for {route_id!r} requires module and symbol"
            )
        authority.append(
            HelperAuthorityReference(
                route_id=route_id,
                condition=condition,
                module=module,
                symbol=symbol,
                required=required,
            )
        )
    return authority


def _authority_drift(
    routes: tuple[HelperAuthorityReference, ...],
    bindings: tuple[HelperAuthorityReference, ...],
) -> tuple[
    tuple[HelperAuthorityReference, ...],
    tuple[HelperAuthorityReference, ...],
]:
    route_counts = Counter(item.identity for item in routes)
    binding_counts = Counter(item.identity for item in bindings)
    route_examples = {item.identity: item for item in routes}
    binding_examples = {item.identity: item for item in bindings}
    route_only = tuple(
        sorted(
            [
                route_examples[identity]
                for identity, count in (route_counts - binding_counts).items()
                for _ in range(count)
            ],
            key=_authority_sort_key,
        )
    )
    binding_only = tuple(
        sorted(
            [
                binding_examples[identity]
                for identity, count in (binding_counts - route_counts).items()
                for _ in range(count)
            ],
            key=_authority_sort_key,
        )
    )
    return route_only, binding_only


def _authority_sort_key(
    item: HelperAuthorityReference,
) -> tuple[str, int, str, str, str, bool]:
    return (
        item.route_id,
        0 if item.condition == "base" else 1,
        item.condition,
        item.module,
        item.symbol,
        item.required,
    )


def _scan_adapter_calls(
    repo_root: Path,
    *,
    authority_symbols: set[str],
) -> tuple[AdapterDelegationCall, ...]:
    adapter_root = repo_root / _ADAPTER_ROOT
    if not adapter_root.is_dir():
        return ()
    calls: list[AdapterDelegationCall] = []
    for path in sorted(adapter_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_names, imported_modules = _import_index(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            resolved = _resolve_imported_call(
                node.func,
                imported_names=imported_names,
                imported_modules=imported_modules,
            )
            if resolved is None:
                continue
            local_name, module, symbol = resolved
            is_price_call = symbol.startswith("price_")
            matches_required_authority = symbol in authority_symbols
            if not is_price_call and not matches_required_authority:
                continue
            calls.append(
                AdapterDelegationCall(
                    path=path.relative_to(repo_root).as_posix(),
                    line=int(node.lineno),
                    local_name=local_name,
                    module=module,
                    symbol=symbol,
                    is_price_call=is_price_call,
                    matches_required_authority=matches_required_authority,
                )
            )
    return tuple(sorted(calls))


def _import_index(
    tree: ast.AST,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    imported_names: dict[str, tuple[str, str]] = {}
    imported_modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_names[alias.asname or alias.name] = (
                    node.module,
                    alias.name,
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                imported_modules[local_name] = alias.name
    return imported_names, imported_modules


def _resolve_imported_call(
    function: ast.expr,
    *,
    imported_names: Mapping[str, tuple[str, str]],
    imported_modules: Mapping[str, str],
) -> tuple[str, str, str] | None:
    if isinstance(function, ast.Name):
        imported = imported_names.get(function.id)
        if imported is None:
            return None
        module, symbol = imported
        return function.id, module, symbol

    dotted = _dotted_name(function)
    if dotted is None or "." not in dotted:
        return None
    matching_imports = [
        local_name
        for local_name in imported_modules
        if dotted.startswith(f"{local_name}.")
    ]
    if not matching_imports:
        return None
    local_root = max(matching_imports, key=len)
    remainder = dotted[len(local_root) + 1 :]
    imported_module = imported_modules[local_root]
    parts = remainder.rsplit(".", 1)
    if len(parts) == 1:
        module = imported_module
        symbol = parts[0]
    else:
        module = f"{imported_module}.{parts[0]}"
        symbol = parts[1]
    return dotted, module, symbol


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _append_authority_section(
    lines: list[str],
    title: str,
    authority: tuple[HelperAuthorityReference, ...],
) -> None:
    lines.append("")
    lines.append(title)
    if not authority:
        lines.append("- none")
        return
    grouped: dict[str, list[HelperAuthorityReference]] = defaultdict(list)
    for item in authority:
        grouped[item.route_id].append(item)
    for route_id in sorted(grouped):
        lines.append(f"- {route_id}")
        for item in grouped[route_id]:
            lines.append(
                f"  [{item.condition}] {item.module}.{item.symbol}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
