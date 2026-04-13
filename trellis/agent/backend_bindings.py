"""Canonical backend-binding catalog independent from the route registry.

The binding catalog is now the authoritative source for exact runtime binding
facts: helper/kernel primitives, conditional binding-family overrides, and the
stable binding identity derived from those surfaces. The route registry remains
responsible for matching, aliases, admissibility, and temporary transition
guidance, while discovered routes can still be overlaid into the catalog for
analysis mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.knowledge.schema import ProductIR


@dataclass(frozen=True)
class ConditionalBindingPrimitives:
    """Conditional primitive override carried by a backend-binding spec."""

    when: dict[str, Any] | str
    primitives: tuple[PrimitiveRef, ...] = ()


@dataclass(frozen=True)
class ConditionalBindingFamily:
    """Conditional family override carried by a backend-binding spec."""

    when: dict[str, Any] | str
    route_family: str


@dataclass(frozen=True)
class BackendBindingSpec:
    """Catalog entry for one route-backed exact binding surface."""

    route_id: str
    engine_family: str
    route_family: str
    aliases: tuple[str, ...] = ()
    compatibility_alias_policy: str = "operator_visible"
    primitives: tuple[PrimitiveRef, ...] = ()
    conditional_primitives: tuple[ConditionalBindingPrimitives, ...] = ()
    conditional_route_family: tuple[ConditionalBindingFamily, ...] = ()


@dataclass(frozen=True)
class ResolvedBackendBindingSpec:
    """Resolved exact-backend facts for one product/binding combination."""

    route_id: str
    engine_family: str
    route_family: str
    aliases: tuple[str, ...] = ()
    compatibility_alias_policy: str = "operator_visible"
    binding_id: str = ""
    primitives: tuple[PrimitiveRef, ...] = ()
    primitive_refs: tuple[str, ...] = ()
    helper_refs: tuple[str, ...] = ()
    pricing_kernel_refs: tuple[str, ...] = ()
    schedule_builder_refs: tuple[str, ...] = ()
    cashflow_engine_refs: tuple[str, ...] = ()
    market_binding_refs: tuple[str, ...] = ()
    exact_target_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendBindingCatalog:
    """Loaded binding catalog for canonical or analysis route authority."""

    bindings: tuple[BackendBindingSpec, ...]
    _route_index: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)


_CATALOG_CACHE: dict[tuple, BackendBindingCatalog] = {}
_CANONICAL_PATH = Path(__file__).resolve().parent / "knowledge" / "canonical" / "backend_bindings.yaml"


def _cache_key(*, include_discovered: bool) -> tuple:
    canonical_mtime = _CANONICAL_PATH.stat().st_mtime_ns if _CANONICAL_PATH.exists() else 0
    discovered_key: tuple[object, ...] = ()
    if include_discovered:
        from trellis.agent import route_registry as route_registry_module

        discovered_key = tuple(route_registry_module._cache_key(include_discovered=True))
    return (include_discovered, canonical_mtime, get_repo_revision(), discovered_key)


def load_backend_binding_catalog(
    *,
    include_discovered: bool = False,
    registry=None,
) -> BackendBindingCatalog:
    """Load the canonical backend-binding catalog with optional discovered overlays."""
    key = None
    if registry is None:
        key = _cache_key(include_discovered=include_discovered)
        cached = _CATALOG_CACHE.get(key)
        if cached is not None:
            return cached

    bindings = list(_load_canonical_bindings())
    if registry is None and include_discovered:
        from trellis.agent.route_registry import load_route_registry

        registry = load_route_registry(include_discovered=True)
    if registry is not None:
        bindings = _overlay_registry_bindings(bindings, registry)

    route_index: dict[str, list[int]] = {}
    for idx, binding in enumerate(bindings):
        route_index.setdefault(binding.route_id, []).append(idx)
        for alias in binding.aliases:
            route_index.setdefault(alias, []).append(idx)
    catalog = BackendBindingCatalog(
        bindings=tuple(bindings),
        _route_index={name: tuple(indexes) for name, indexes in route_index.items()},
    )
    if key is not None:
        _CATALOG_CACHE[key] = catalog
    return catalog


def clear_backend_binding_catalog_cache() -> None:
    """Clear the backend-binding catalog cache."""
    _CATALOG_CACHE.clear()


def find_backend_binding_by_route_id(
    route_id: str,
    catalog: BackendBindingCatalog | None = None,
) -> BackendBindingSpec | None:
    """Look up a binding catalog entry by route id or alias."""
    if not str(route_id or "").strip():
        return None
    if catalog is None:
        catalog = load_backend_binding_catalog()
    matches = catalog._route_index.get(str(route_id).strip(), ())
    if not matches:
        return None
    return catalog.bindings[matches[0]]


def resolve_backend_binding_spec(
    binding: BackendBindingSpec,
    *,
    product_ir: ProductIR | None = None,
    primitive_plan=None,
) -> ResolvedBackendBindingSpec:
    """Resolve one binding spec for the current product traits."""
    route_family = _resolve_binding_route_family(binding, product_ir)
    primitives = _resolve_binding_primitives(binding, product_ir)
    primitive_refs = tuple(
        dict.fromkeys(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitives
            if str(getattr(primitive, "module", "") or "").strip()
            and str(getattr(primitive, "symbol", "") or "").strip()
        )
    )
    helper_refs = _primitive_refs_for_role(primitives, "route_helper")
    pricing_kernel_refs = _primitive_refs_for_role(primitives, "pricing_kernel")
    schedule_builder_refs = _primitive_refs_for_role(primitives, "schedule_builder")
    cashflow_engine_refs = _primitive_refs_for_role(primitives, "cashflow_engine")
    market_binding_refs = _primitive_refs_for_role(primitives, "market_binding")
    exact_target_refs = _exact_target_refs_for(primitives)
    binding_id = _binding_id_for(
        primitives=primitives,
        exact_target_refs=exact_target_refs,
        engine_family=binding.engine_family,
        route_family=route_family,
        primitive_plan=primitive_plan,
    )
    return ResolvedBackendBindingSpec(
        route_id=binding.route_id,
        engine_family=binding.engine_family,
        route_family=route_family,
        aliases=binding.aliases,
        compatibility_alias_policy=binding.compatibility_alias_policy,
        binding_id=binding_id,
        primitives=primitives,
        primitive_refs=primitive_refs,
        helper_refs=helper_refs,
        pricing_kernel_refs=pricing_kernel_refs,
        schedule_builder_refs=schedule_builder_refs,
        cashflow_engine_refs=cashflow_engine_refs,
        market_binding_refs=market_binding_refs,
        exact_target_refs=exact_target_refs,
    )


def resolve_backend_binding_by_route_id(
    route_id: str,
    *,
    product_ir: ProductIR | None = None,
    primitive_plan=None,
    catalog: BackendBindingCatalog | None = None,
) -> ResolvedBackendBindingSpec | None:
    """Resolve one binding surface by route id or alias."""
    binding = find_backend_binding_by_route_id(route_id, catalog)
    if binding is None:
        return None
    return resolve_backend_binding_spec(
        binding,
        product_ir=product_ir,
        primitive_plan=primitive_plan,
    )


def _load_canonical_bindings() -> tuple[BackendBindingSpec, ...]:
    raw = _load_yaml(_CANONICAL_PATH, default={})
    entries = raw.get("bindings") if isinstance(raw, dict) else ()
    if not isinstance(entries, list):
        return ()
    return tuple(_binding_from_raw(entry) for entry in entries if isinstance(entry, dict) and entry.get("route_id"))


def _overlay_registry_bindings(
    bindings: list[BackendBindingSpec],
    registry,
) -> list[BackendBindingSpec]:
    existing = {binding.route_id for binding in bindings}
    aliases = {
        alias
        for binding in bindings
        for alias in binding.aliases
    }
    overlaid = list(bindings)
    for route in getattr(registry, "routes", ()) or ():
        route_id = str(getattr(route, "id", "") or "").strip()
        if not route_id or route_id in existing or route_id in aliases:
            continue
        overlaid.append(_binding_from_route(route))
    return overlaid


def _binding_from_raw(raw: dict[str, Any]) -> BackendBindingSpec:
    return BackendBindingSpec(
        route_id=str(raw.get("route_id") or "").strip(),
        engine_family=str(raw.get("engine_family") or "").strip(),
        route_family=str(raw.get("route_family") or "").strip(),
        aliases=_str_tuple(raw.get("aliases")),
        compatibility_alias_policy=str(raw.get("compatibility_alias_policy") or "operator_visible").strip() or "operator_visible",
        primitives=_parse_primitives(raw.get("primitives")),
        conditional_primitives=_parse_conditional_primitives(raw.get("conditional_primitives")),
        conditional_route_family=_parse_conditional_route_family(raw.get("conditional_route_family")),
    )


def _binding_from_route(route) -> BackendBindingSpec:
    return BackendBindingSpec(
        route_id=str(route.id),
        engine_family=str(route.engine_family),
        route_family=str(route.route_family),
        aliases=tuple(getattr(route, "aliases", ()) or ()),
        compatibility_alias_policy=str(
            getattr(route, "compatibility_alias_policy", None) or "operator_visible"
        ),
        primitives=tuple(getattr(route, "primitives", ()) or ()),
        conditional_primitives=tuple(
            ConditionalBindingPrimitives(
                when=getattr(block, "when", "default"),
                primitives=tuple(getattr(block, "primitives", ()) or ()),
            )
            for block in (getattr(route, "conditional_primitives", ()) or ())
        ),
        conditional_route_family=tuple(
            ConditionalBindingFamily(
                when=getattr(block, "when", "default"),
                route_family=str(getattr(block, "route_family", "") or ""),
            )
            for block in (getattr(route, "conditional_route_family", ()) or ())
        ),
    )


def _load_yaml(path: Path, *, default):
    if not path.exists():
        return default
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return default if data is None else data


def _str_tuple(values) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, (str, bytes)):
        return (str(values),)
    return tuple(str(value) for value in values if str(value).strip())


def _parse_primitives(raw: Any) -> tuple[PrimitiveRef, ...]:
    if not raw:
        return ()
    return tuple(_parse_primitive(item) for item in raw if isinstance(item, dict))


def _parse_primitive(raw: dict[str, Any]) -> PrimitiveRef:
    return PrimitiveRef(
        module=str(raw["module"]),
        symbol=str(raw["symbol"]),
        role=str(raw["role"]),
        required=raw.get("required", True),
        excluded=raw.get("excluded", False),
    )


def _parse_conditional_primitives(raw: Any) -> tuple[ConditionalBindingPrimitives, ...]:
    if not raw:
        return ()
    rows: list[ConditionalBindingPrimitives] = []
    for entry in raw:
        if not isinstance(entry, dict) or "when" not in entry:
            continue
        rows.append(
            ConditionalBindingPrimitives(
                when=entry.get("when", "default"),
                primitives=_parse_primitives(entry.get("primitives")),
            )
        )
    return tuple(rows)


def _parse_conditional_route_family(raw: Any) -> tuple[ConditionalBindingFamily, ...]:
    if not raw:
        return ()
    rows: list[ConditionalBindingFamily] = []
    for entry in raw:
        if not isinstance(entry, dict) or "when" not in entry or "route_family" not in entry:
            continue
        rows.append(
            ConditionalBindingFamily(
                when=entry.get("when", "default"),
                route_family=str(entry.get("route_family") or "").strip(),
            )
        )
    return tuple(rows)


def _resolve_binding_primitives(
    binding: BackendBindingSpec,
    product_ir: ProductIR | None,
) -> tuple[PrimitiveRef, ...]:
    if not binding.conditional_primitives:
        return binding.primitives

    exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
    payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
    model_family = getattr(product_ir, "model_family", "generic") if product_ir is not None else "generic"

    explicit_default: tuple[PrimitiveRef, ...] | None = None
    for cond in binding.conditional_primitives:
        if isinstance(cond.when, str) and cond.when == "default":
            explicit_default = cond.primitives if cond.primitives else binding.primitives
            continue
        if isinstance(cond.when, dict) and _matches_condition(
            cond.when,
            payoff_family,
            exercise,
            model_family,
            product_ir,
        ):
            return cond.primitives if cond.primitives else binding.primitives
    return explicit_default if explicit_default is not None else binding.primitives


def _resolve_binding_route_family(
    binding: BackendBindingSpec,
    product_ir: ProductIR | None,
) -> str:
    if not binding.conditional_route_family:
        return binding.route_family

    exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
    payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
    model_family = getattr(product_ir, "model_family", "generic") if product_ir is not None else "generic"

    explicit_default = None
    for cond in binding.conditional_route_family:
        if isinstance(cond.when, str) and cond.when == "default":
            explicit_default = cond.route_family
            continue
        if isinstance(cond.when, dict) and _matches_condition(
            cond.when,
            payoff_family,
            exercise,
            model_family,
            product_ir,
        ):
            return cond.route_family
    return explicit_default if explicit_default is not None else binding.route_family


def _primitive_refs_for_role(
    primitives: tuple[PrimitiveRef, ...],
    role: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitives
            if str(getattr(primitive, "role", "") or "").strip() == role
            and str(getattr(primitive, "module", "") or "").strip()
            and str(getattr(primitive, "symbol", "") or "").strip()
        )
    )


def _exact_target_refs_for(
    primitives: tuple[PrimitiveRef, ...],
) -> tuple[str, ...]:
    prioritized_roles = (
        "route_helper",
        "pricing_kernel",
        "solver",
        "payoff_kernel",
        "engine",
        "market_binding",
        "cashflow_engine",
    )
    refs_by_role = {
        role: _primitive_refs_for_role(primitives, role)
        for role in prioritized_roles
    }
    for role in prioritized_roles:
        refs = refs_by_role[role]
        if refs:
            return refs
    return ()


def _binding_id_for(
    *,
    primitives: tuple[PrimitiveRef, ...],
    exact_target_refs: tuple[str, ...],
    engine_family: str,
    route_family: str,
    primitive_plan=None,
) -> str:
    prioritized_roles = (
        "route_helper",
        "pricing_kernel",
        "solver",
        "payoff_kernel",
        "engine",
        "market_binding",
        "cashflow_engine",
    )
    source_primitives = tuple(getattr(primitive_plan, "primitives", ()) or ()) or primitives
    if source_primitives:
        for role in prioritized_roles:
            for primitive in source_primitives:
                if str(getattr(primitive, "role", "") or "").strip() != role:
                    continue
                module = str(getattr(primitive, "module", "") or "").strip()
                symbol = str(getattr(primitive, "symbol", "") or "").strip()
                if module and symbol:
                    return f"{module}.{symbol}"
    if exact_target_refs:
        return str(exact_target_refs[0]).strip()
    engine = str(engine_family or "").strip()
    family = str(route_family or "").strip()
    if engine and family:
        return f"{engine}:{family}:fallback"
    if engine:
        return f"{engine}:fallback"
    return "unbound:fallback"


def _matches_condition(
    when: dict[str, Any],
    payoff_family: str,
    exercise_style: str,
    model_family: str,
    product_ir: ProductIR | None,
) -> bool:
    payoff_families = _expanded_payoff_families(payoff_family, product_ir)
    for key, expected in when.items():
        if key == "payoff_family":
            if isinstance(expected, list):
                if not any(candidate in payoff_families for candidate in expected):
                    return False
            elif expected not in payoff_families:
                return False
        elif key == "exercise_style":
            if isinstance(expected, list):
                if exercise_style not in expected:
                    return False
            elif exercise_style != expected:
                return False
        elif key == "model_family":
            if isinstance(expected, list):
                if model_family not in expected:
                    return False
            elif model_family != expected:
                return False
        elif key == "schedule_dependence":
            if product_ir is not None and product_ir.schedule_dependence != expected:
                return False
        else:
            return False
    return True


def _expanded_payoff_families(
    payoff_family: str,
    product_ir: ProductIR | None,
) -> frozenset[str]:
    families = {str(payoff_family or "")}
    instrument = str(getattr(product_ir, "instrument", "") or "").strip().lower()
    if instrument == "puttable_bond" or payoff_family == "puttable_fixed_income":
        families.update({"puttable_fixed_income", "callable_fixed_income", "callable_bond", "bond"})
    elif instrument == "callable_bond" or payoff_family == "callable_fixed_income":
        families.update({"callable_fixed_income", "callable_bond", "bond"})
    families.discard("")
    return frozenset(families)
