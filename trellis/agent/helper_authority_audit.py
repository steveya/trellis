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
_ADAPTER_PACKAGE = ".".join(_ADAPTER_ROOT.parts)
_MAX_SOURCE_COMPONENT = 2**31 - 1


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


@dataclass(frozen=True, order=True)
class AdapterIndirectAuthorityUse:
    """One non-call reference to imported authority in a checked adapter."""

    path: str
    line: int
    local_name: str
    module: str
    symbol: str
    use_kind: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class _ImportBinding:
    """One source-ordered import binding in a lexical scope."""

    position: tuple[int, int, int]
    module: str
    symbol: str
    conditional: bool


@dataclass(frozen=True)
class _ImportScope:
    """Imported bindings and lexical-name ownership for one Python scope."""

    parent: _ImportScope | None
    kind: str
    definition_position: tuple[int, int, int]
    imports: Mapping[str, tuple[_ImportBinding, ...]]
    local_names: frozenset[str]
    global_names: frozenset[str]
    nonlocal_names: frozenset[str]


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
    adapter_indirect_authority_uses: tuple[AdapterIndirectAuthorityUse, ...]

    @property
    def has_route_binding_drift(self) -> bool:
        """Return whether canonical routes and exact bindings disagree."""
        return bool(self.route_only_authority or self.binding_only_authority)

    @property
    def has_adapter_authority(self) -> bool:
        """Return whether checked adapters call or reference required authority."""
        return any(
            item.matches_required_authority for item in self.adapter_calls
        ) or bool(self.adapter_indirect_authority_uses)

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
        indirect_authority_paths = {
            item.path for item in self.adapter_indirect_authority_uses
        }
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
            "adapter_indirect_authority_use_file_count": len(
                indirect_authority_paths
            ),
            "adapter_indirect_authority_use_count": len(
                self.adapter_indirect_authority_uses
            ),
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
            "adapter_indirect_authority_uses": [
                item.to_dict()
                for item in self.adapter_indirect_authority_uses
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
    authority_targets = {
        (item.module, item.symbol)
        for item in route_authority + binding_authority
    }
    adapter_calls = _scan_adapter_calls(
        repo_root,
        authority_targets=authority_targets,
    )
    adapter_indirect_authority_uses = _scan_indirect_authority_uses(
        repo_root,
        authority_targets=authority_targets,
    )
    return HelperAuthorityReport(
        schema_version=2,
        promoted_route_count=len(promoted_routes),
        route_authority=route_authority,
        binding_authority=binding_authority,
        route_only_authority=route_only,
        binding_only_authority=binding_only,
        adapter_calls=adapter_calls,
        adapter_indirect_authority_uses=adapter_indirect_authority_uses,
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
        "adapter_indirect_authority_use_files="
        f"{summary['adapter_indirect_authority_use_file_count']}",
        "adapter_indirect_authority_uses="
        f"{summary['adapter_indirect_authority_use_count']}",
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
    lines.append("")
    lines.append("Adapter indirect authority references")
    if not report.adapter_indirect_authority_uses:
        lines.append("- none")
    else:
        for item in report.adapter_indirect_authority_uses:
            lines.append(
                f"- [{item.use_kind}] {item.path}:{item.line} "
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
    parser.add_argument(
        "--fail-on-adapter-authority",
        action="store_true",
        help=(
            "Return exit code 1 when a checked adapter calls or indirectly "
            "references a symbol that is required authority on a promoted "
            "route or exact binding."
        ),
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
    if (args.fail_on_drift and report.has_route_binding_drift) or (
        args.fail_on_adapter_authority and report.has_adapter_authority
    ):
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
    authority_targets: set[tuple[str, str]],
) -> tuple[AdapterDelegationCall, ...]:
    adapter_root = repo_root / _ADAPTER_ROOT
    if not adapter_root.is_dir():
        return ()
    calls: list[AdapterDelegationCall] = []
    for path in sorted(adapter_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scope_by_node = _index_import_scopes(
            tree,
            package=_ADAPTER_PACKAGE,
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            resolved_references = _resolve_imported_references(
                node.func,
                scope=scope_by_node[id(node.func)],
            )
            for local_name, module, symbol in resolved_references:
                is_price_call = symbol.startswith("price_")
                matches_required_authority = (
                    module,
                    symbol,
                ) in authority_targets
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


def _scan_indirect_authority_uses(
    repo_root: Path,
    *,
    authority_targets: set[tuple[str, str]],
) -> tuple[AdapterIndirectAuthorityUse, ...]:
    """Find authority symbols used as values rather than direct call targets."""
    adapter_root = repo_root / _ADAPTER_ROOT
    if not adapter_root.is_dir():
        return ()
    uses: list[AdapterIndirectAuthorityUse] = []
    for path in sorted(adapter_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scope_by_node = _index_import_scopes(
            tree,
            package=_ADAPTER_PACKAGE,
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not any(alias.name == "*" for alias in node.names):
                continue
            module = _normalize_import_from_module(
                node,
                package=_ADAPTER_PACKAGE,
            )
            if module is None or not _is_authority_namespace(
                module,
                authority_targets,
            ):
                continue
            uses.append(
                AdapterIndirectAuthorityUse(
                    path=path.relative_to(repo_root).as_posix(),
                    line=int(node.lineno),
                    local_name="*",
                    module=module,
                    symbol="*",
                    use_kind="wildcard_import",
                )
            )
        direct_call_targets = {
            id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        nested_attribute_nodes = {
            id(child)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            for child in ast.walk(node.value)
            if isinstance(child, (ast.Name, ast.Attribute))
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Name, ast.Attribute)):
                continue
            if (
                not isinstance(node.ctx, ast.Load)
                or id(node) in direct_call_targets
            ):
                continue
            resolved_references = _resolve_imported_references(
                node,
                scope=scope_by_node[id(node)],
            )
            for local_name, resolved_module, resolved_symbol in resolved_references:
                module = resolved_module
                symbol = resolved_symbol
                if symbol:
                    if (module, symbol) in authority_targets:
                        use_kind = "indirect_reference"
                    elif _is_authority_namespace(
                        f"{module}.{symbol}",
                        authority_targets,
                    ):
                        if id(node) in nested_attribute_nodes:
                            continue
                        module = f"{module}.{symbol}"
                        symbol = "*"
                        use_kind = "indirect_module_reference"
                    else:
                        continue
                elif _is_authority_namespace(module, authority_targets):
                    if id(node) in nested_attribute_nodes:
                        continue
                    symbol = "*"
                    use_kind = "indirect_module_reference"
                else:
                    continue
                uses.append(
                    AdapterIndirectAuthorityUse(
                        path=path.relative_to(repo_root).as_posix(),
                        line=int(node.lineno),
                        local_name=local_name,
                        module=module,
                        symbol=symbol,
                        use_kind=use_kind,
                    )
                )
    return tuple(sorted(uses))


class _ScopeBindingCollector(ast.NodeVisitor):
    """Collect bindings in one scope without entering child scopes."""

    def __init__(self, *, package: str) -> None:
        self.package = package
        self.imports: dict[str, list[_ImportBinding]] = defaultdict(list)
        self.local_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self._conditional_depth = 0
        self._binding_sequence = 0

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                local_name = alias.asname
                module = alias.name
            else:
                local_name = alias.name.split(".", 1)[0]
                module = local_name
            self._record_import(
                local_name=local_name,
                module=module,
                symbol="",
                node=alias,
            )
            self.local_names.add(local_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        from_module = _normalize_import_from_module(
            node,
            package=self.package,
        )
        if not from_module:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            self._record_import(
                local_name=local_name,
                module=from_module,
                symbol=alias.name,
                node=alias,
            )
            self.local_names.add(local_name)

    def visit_If(self, node: ast.If) -> None:
        self._visit_conditional(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_conditional(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_conditional(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_conditional(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_conditional(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_conditional(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_conditional(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_conditional(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._visit_conditional(node)

    def _visit_conditional(self, node: ast.AST) -> None:
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def _record_import(
        self,
        *,
        local_name: str,
        module: str,
        symbol: str,
        node: ast.AST,
    ) -> None:
        self._binding_sequence += 1
        self.imports[local_name].append(
            _ImportBinding(
                position=(
                    int(getattr(node, "lineno", 0)),
                    int(getattr(node, "col_offset", 0)),
                    self._binding_sequence,
                ),
                module=module,
                symbol=symbol,
                conditional=self._conditional_depth > 0,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.local_names.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.local_names.add(node.name)
        self.generic_visit(node)


def _make_import_scope(
    node: ast.AST,
    *,
    parent: _ImportScope | None,
    kind: str,
    package: str,
) -> _ImportScope:
    collector = _ScopeBindingCollector(package=package)
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for statement in node.body:
            collector.visit(statement)
    elif isinstance(node, ast.Lambda):
        collector.visit(node.body)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        collector.local_names.update(_argument_names(node.args))
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for generator in node.generators:
            collector.local_names.update(_stored_names(generator.target))

    collector.local_names.difference_update(collector.global_names)
    collector.local_names.difference_update(collector.nonlocal_names)
    imports = {
        local_name: tuple(sorted(bindings, key=lambda item: item.position))
        for local_name, bindings in collector.imports.items()
    }
    return _ImportScope(
        parent=parent,
        kind=kind,
        definition_position=(
            int(getattr(node, "lineno", 0)),
            int(getattr(node, "col_offset", 0)),
            _MAX_SOURCE_COMPONENT,
        ),
        imports=imports,
        local_names=frozenset(collector.local_names),
        global_names=frozenset(collector.global_names),
        nonlocal_names=frozenset(collector.nonlocal_names),
    )


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _stored_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, (ast.Store, ast.Del))
    }


class _ImportScopeIndexer(ast.NodeVisitor):
    """Associate each AST node with the lexical import scope it executes in."""

    def __init__(self, tree: ast.Module, *, package: str) -> None:
        self.package = package
        self.current = _make_import_scope(
            tree,
            parent=None,
            kind="module",
            package=package,
        )
        self.scope_by_node: dict[int, _ImportScope] = {}

    def index(self, tree: ast.Module) -> dict[int, _ImportScope]:
        self.visit(tree)
        return self.scope_by_node

    def generic_visit(self, node: ast.AST) -> None:
        self.scope_by_node[id(node)] = self.current
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self.scope_by_node[id(node)] = self.current
        self._visit_argument_expressions(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns is not None:
            self.visit(node.returns)
        child = _make_import_scope(
            node,
            parent=self.current,
            kind="function",
            package=self.package,
        )
        self._visit_body_in_scope(node.body, child)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_by_node[id(node)] = self.current
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        child = _make_import_scope(
            node,
            parent=self.current,
            kind="class",
            package=self.package,
        )
        self._visit_body_in_scope(node.body, child)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scope_by_node[id(node)] = self.current
        self._visit_argument_expressions(node.args)
        child = _make_import_scope(
            node,
            parent=self.current,
            kind="lambda",
            package=self.package,
        )
        previous = self.current
        self.current = child
        self.visit(node.body)
        self.current = previous

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, (node.elt,))

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        result_nodes: tuple[ast.AST, ...],
    ) -> None:
        self.scope_by_node[id(node)] = self.current
        first, *remaining = node.generators
        self.visit(first.iter)
        child = _make_import_scope(
            node,
            parent=self.current,
            kind=(
                "generator"
                if isinstance(node, ast.GeneratorExp)
                else "comprehension"
            ),
            package=self.package,
        )
        previous = self.current
        self.current = child
        self.visit(first.target)
        for condition in first.ifs:
            self.visit(condition)
        for generator in remaining:
            self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for result_node in result_nodes:
            self.visit(result_node)
        self.current = previous

    def _visit_argument_expressions(self, arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if arguments.vararg and arguments.vararg.annotation is not None:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg and arguments.kwarg.annotation is not None:
            self.visit(arguments.kwarg.annotation)
        for default in arguments.defaults:
            self.visit(default)
        for default in arguments.kw_defaults:
            if default is not None:
                self.visit(default)

    def _visit_body_in_scope(
        self,
        body: list[ast.stmt],
        scope: _ImportScope,
    ) -> None:
        previous = self.current
        self.current = scope
        for statement in body:
            self.visit(statement)
        self.current = previous


def _index_import_scopes(
    tree: ast.AST,
    *,
    package: str,
) -> dict[int, _ImportScope]:
    if not isinstance(tree, ast.Module):
        raise TypeError("Import scope indexing requires a module AST")
    return _ImportScopeIndexer(tree, package=package).index(tree)


def _normalize_import_from_module(
    node: ast.ImportFrom,
    *,
    package: str,
) -> str | None:
    """Return an absolute module path for one ``from`` import."""
    if node.level == 0:
        return node.module
    package_parts = package.split(".")
    parent_count = node.level - 1
    if parent_count >= len(package_parts):
        return None
    resolved_parts = package_parts[: len(package_parts) - parent_count]
    if node.module:
        resolved_parts.extend(node.module.split("."))
    return ".".join(resolved_parts)


def _resolve_imported_references(
    reference: ast.expr,
    *,
    scope: _ImportScope,
) -> tuple[tuple[str, str, str], ...]:
    if isinstance(reference, ast.Name):
        return tuple(
            (reference.id, binding.module, binding.symbol)
            for binding in _lookup_import_bindings(
                scope,
                reference.id,
                reference=reference,
            )
        )

    dotted = _dotted_name(reference)
    if dotted is None or "." not in dotted:
        return ()
    parts = dotted.split(".")
    local_root = parts[0]
    remainder = parts[1:]
    resolved: set[tuple[str, str, str]] = set()
    for binding in _lookup_import_bindings(
        scope,
        local_root,
        reference=reference,
    ):
        imported_module = binding.module
        imported_symbol = binding.symbol
        base = imported_module
        if imported_symbol:
            base = f"{base}.{imported_symbol}"
        module = base
        if len(remainder) > 1:
            module = f"{base}.{'.'.join(remainder[:-1])}"
        resolved.add((dotted, module, remainder[-1]))
    return tuple(sorted(resolved))


def _lookup_import_bindings(
    scope: _ImportScope,
    local_name: str,
    *,
    reference: ast.expr,
) -> tuple[_ImportBinding, ...]:
    reference_position = (
        int(getattr(reference, "lineno", 0)),
        int(getattr(reference, "col_offset", 0)),
        _MAX_SOURCE_COMPONENT,
    )
    possible_since: tuple[int, int, int] | None = None
    current: _ImportScope | None = scope
    while current is not None:
        if local_name in current.global_names:
            if current.kind != "module":
                possible_since = current.definition_position
            current = _module_scope(current)
        bindings = current.imports.get(local_name)
        if bindings:
            if possible_since is None:
                active_bindings = _active_import_bindings(
                    bindings,
                    reference_position=reference_position,
                )
            else:
                active_bindings = _possible_deferred_import_bindings(
                    bindings,
                    possible_since=possible_since,
                )
            if active_bindings:
                return active_bindings
        if local_name in current.local_names:
            return ()
        parent = current.parent
        if current.kind in {
            "function",
            "lambda",
            "comprehension",
            "generator",
        }:
            while parent is not None and parent.kind == "class":
                parent = parent.parent
        if current.kind in {"function", "lambda", "generator"}:
            possible_since = current.definition_position
        current = parent
    return ()


def _active_import_bindings(
    bindings: tuple[_ImportBinding, ...],
    *,
    reference_position: tuple[int, int, int],
) -> tuple[_ImportBinding, ...]:
    """Return imports that can be active at one source position."""
    eligible = tuple(
        binding for binding in bindings if binding.position <= reference_position
    )
    if not eligible:
        return ()
    unconditional = tuple(
        binding for binding in eligible if not binding.conditional
    )
    if not unconditional:
        return eligible
    latest_unconditional = unconditional[-1]
    return (latest_unconditional,) + tuple(
        binding
        for binding in eligible
        if binding.conditional and binding.position > latest_unconditional.position
    )


def _possible_deferred_import_bindings(
    bindings: tuple[_ImportBinding, ...],
    *,
    possible_since: tuple[int, int, int],
) -> tuple[_ImportBinding, ...]:
    """Return enclosing imports possible across deferred execution times."""
    candidates = _active_import_bindings(
        bindings,
        reference_position=possible_since,
    ) + tuple(
        binding for binding in bindings if binding.position > possible_since
    )
    unique: dict[tuple[str, str], _ImportBinding] = {}
    for binding in candidates:
        unique.setdefault((binding.module, binding.symbol), binding)
    return tuple(unique.values())


def _module_scope(scope: _ImportScope) -> _ImportScope:
    current = scope
    while current.parent is not None:
        current = current.parent
    return current


def _is_authority_namespace(
    module: str,
    authority_targets: set[tuple[str, str]],
) -> bool:
    return any(
        target_module == module or target_module.startswith(f"{module}.")
        for target_module, _ in authority_targets
    )


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
