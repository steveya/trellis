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
from trellis.agent.contract_pattern import (
    ContractPattern,
    ContractPatternParseError,
    parse_contract_pattern,
)
from trellis.agent.contract_pattern_eval import evaluate_pattern
from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.knowledge.schema import ProductIR


@dataclass(frozen=True)
class ConditionalBindingPrimitives:
    """Conditional primitive override carried by a backend-binding spec.

    Mirrors :class:`trellis.agent.route_registry.ConditionalPrimitive`'s two-
    form dispatch contract (QUA-919 / QUA-921):

    * **Legacy string-tag filter** (``when`` is either a trait mapping or the
      string sentinel ``"default"``).  Dispatch goes through
      :func:`_matches_condition`.
    * **DSL contract pattern** (``contract_pattern`` is non-``None``).
      Dispatch goes through
      :func:`trellis.agent.contract_pattern_eval.evaluate_pattern`.  The
      accompanying ``when`` field is an empty dict so the legacy tag-key
      scan treats the clause as "no legacy conditions" and the evaluator
      takes over.

    Declaring both a non-empty ``when`` mapping and a non-``None``
    ``contract_pattern`` is disallowed — the YAML parser and this
    ``__post_init__`` both reject it so the dispatch fork stays unambiguous.
    The ``"default"`` sentinel is always legacy form and must not carry a
    ``contract_pattern`` (fall-through markers are not structural matches).
    """

    when: dict[str, Any] | str
    primitives: tuple[PrimitiveRef, ...] = ()
    contract_pattern: ContractPattern | None = None

    def __post_init__(self) -> None:
        if self.contract_pattern is None:
            return
        if isinstance(self.when, str):
            raise ValueError(
                "ConditionalBindingPrimitives cannot combine a string 'when' "
                f"sentinel ({self.when!r}) with a contract_pattern; DSL clauses "
                "must use an empty-dict 'when' placeholder."
            )
        if isinstance(self.when, dict) and self.when:
            raise ValueError(
                "ConditionalBindingPrimitives cannot populate both 'when' "
                f"trait keys ({sorted(self.when.keys())}) and a "
                "'contract_pattern'; choose one form."
            )


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
    method: str | None = None,
) -> ResolvedBackendBindingSpec:
    """Resolve one binding spec for the current product traits.

    The optional ``method`` keyword lets conditional-primitive dispatch
    discriminate by the pricing method (e.g. ``analytical`` vs
    ``monte_carlo``) when ProductIR traits alone cannot separate two
    branches — the quanto family collapse in QUA-912 is the first use.
    """
    route_family = _resolve_binding_route_family(binding, product_ir, method=method)
    primitives = _resolve_binding_primitives(binding, product_ir, method=method)
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
    method: str | None = None,
) -> ResolvedBackendBindingSpec | None:
    """Resolve one binding surface by route id or alias.

    ``method`` (QUA-915) threads the pricing-plan method through to the
    binding overlay so method-keyed ``conditional_primitives`` clauses
    dispatch consistently.
    """
    binding = find_backend_binding_by_route_id(route_id, catalog)
    if binding is None:
        return None
    return resolve_backend_binding_spec(
        binding,
        product_ir=product_ir,
        primitive_plan=primitive_plan,
        method=method,
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
                # Preserve DSL contract_pattern when the upstream
                # route-registry block was already DSL form (QUA-919 +
                # QUA-921 share the same two-form contract across registries).
                contract_pattern=getattr(block, "contract_pattern", None),
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
    return tuple(
        _parse_primitive(item)
        for item in raw
        if isinstance(item, dict)
        and item.get("module")
        and item.get("symbol")
        and item.get("role")
    )


def _parse_primitive(raw: dict[str, Any]) -> PrimitiveRef:
    return PrimitiveRef(
        module=str(raw["module"]),
        symbol=str(raw["symbol"]),
        role=str(raw["role"]),
        required=raw.get("required", True),
        excluded=raw.get("excluded", False),
    )


def _parse_conditional_primitives(raw: Any) -> tuple[ConditionalBindingPrimitives, ...]:
    """Parse a ``conditional_primitives`` list into structured dataclass entries.

    Each entry's ``when:`` block is inspected up-front to decide which form it
    uses:

    * A mapping containing the ``contract_pattern`` key is DSL form; the
      accompanying payload is parsed via
      :func:`trellis.agent.contract_pattern.parse_contract_pattern` and stored
      on :attr:`ConditionalBindingPrimitives.contract_pattern`.  The
      dispatch-time ``when`` field is left as an empty dict so legacy
      trait-key scans treat the clause as "no legacy conditions" and hand off
      to the evaluator.
    * Any other mapping is legacy trait-filter form and stored verbatim on
      ``when``.
    * The string sentinel ``"default"`` is preserved as-is for the catch-all
      branch.

    Mixing ``contract_pattern`` with legacy trait keys in the same ``when:``
    block is rejected (mirrors the route registry parser) to keep the
    dispatch fork unambiguous.
    """
    if not raw:
        return ()
    rows: list[ConditionalBindingPrimitives] = []
    for entry in raw:
        if not isinstance(entry, dict) or "when" not in entry:
            continue
        raw_when = entry.get("when", "default")
        when_value, contract_pattern = _parse_binding_when_clause(raw_when)
        rows.append(
            ConditionalBindingPrimitives(
                when=when_value,
                primitives=_parse_primitives(entry.get("primitives")),
                contract_pattern=contract_pattern,
            )
        )
    return tuple(rows)


def _parse_binding_when_clause(
    raw_when: Any,
) -> tuple[dict[str, Any] | str, ContractPattern | None]:
    """Classify a raw ``when`` payload as legacy trait-filter or DSL form.

    Returns ``(when_value, contract_pattern)``.  For legacy form
    ``contract_pattern`` is ``None`` and ``when_value`` is the original mapping
    or sentinel string.  For DSL form ``contract_pattern`` is a parsed
    :class:`ContractPattern` and ``when_value`` is an empty dict so the
    legacy-dispatch short-circuit in :func:`_resolve_binding_primitives`
    naturally hands off to the evaluator.

    Mirrors :func:`trellis.agent.route_registry._parse_when_clause` so
    ``routes.yaml`` and ``backend_bindings.yaml`` share a common contract.
    """
    if isinstance(raw_when, str):
        # ``"default"`` sentinel or any other raw string is legacy form.
        return raw_when, None
    if not isinstance(raw_when, dict):
        # Preserve historical tolerance: anything non-dict, non-string falls
        # through to the legacy path unchanged so downstream dispatch decides.
        return raw_when, None

    if "contract_pattern" not in raw_when:
        return raw_when, None

    extra_keys = sorted(key for key in raw_when.keys() if key != "contract_pattern")
    if extra_keys:
        raise ValueError(
            "when-clause cannot mix 'contract_pattern' with legacy trait keys "
            f"{extra_keys}; use either a contract_pattern or legacy tag filters "
            "in a single clause, not both."
        )

    try:
        pattern = parse_contract_pattern(raw_when["contract_pattern"])
    except ContractPatternParseError as exc:
        raise ValueError(
            f"invalid contract_pattern in when-clause: {exc}"
        ) from exc

    return {}, pattern


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
    *,
    method: str | None = None,
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
        if _conditional_binding_primitive_matches(
            cond,
            payoff_family,
            exercise,
            model_family,
            product_ir,
            method=method,
        ):
            return cond.primitives if cond.primitives else binding.primitives
    return explicit_default if explicit_default is not None else binding.primitives


def _conditional_binding_primitive_matches(
    cond: ConditionalBindingPrimitives,
    payoff_family: str,
    exercise_style: str,
    model_family: str,
    product_ir: ProductIR | None,
    *,
    method: str | None = None,
) -> bool:
    """Dispatch a single :class:`ConditionalBindingPrimitives` against a ProductIR.

    Routes the decision through the DSL pattern evaluator when the clause
    carries a :class:`ContractPattern`, or through the legacy string-tag
    filter :func:`_matches_condition` otherwise.  Mirrors
    :func:`trellis.agent.route_registry._conditional_primitive_matches` so the
    binding catalog and route registry dispatch forks stay structurally in
    step.

    The ``"default"`` sentinel is handled by the caller (it needs to
    distinguish catch-all fall-through from a conditional match miss) and
    never reaches this helper.
    """
    if cond.contract_pattern is not None:
        # DSL path: ``None`` product_ir cannot satisfy a structural pattern,
        # matching the route-registry symmetry.  The legacy path also fails
        # every specific clause on ``None`` via its stringy default fallbacks.
        if product_ir is None:
            return False
        return evaluate_pattern(cond.contract_pattern, product_ir).ok

    if isinstance(cond.when, dict):
        return _matches_condition(
            cond.when,
            payoff_family,
            exercise_style,
            model_family,
            product_ir,
            method=method,
        )
    return False


def _resolve_binding_route_family(
    binding: BackendBindingSpec,
    product_ir: ProductIR | None,
    *,
    method: str | None = None,
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
            method=method,
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
    *,
    method: str | None = None,
) -> bool:
    payoff_families = _expanded_payoff_families(payoff_family, product_ir)
    payoff_traits = {
        str(item).strip().lower()
        for item in getattr(product_ir, "payoff_traits", ()) or ()
    }
    instrument = str(getattr(product_ir, "instrument", "") or "").strip().lower()
    for key, expected in when.items():
        if key == "payoff_family":
            if isinstance(expected, list):
                if not any(candidate in payoff_families for candidate in expected):
                    return False
            elif expected not in payoff_families:
                return False
        elif key == "instrument":
            expected_instruments = (
                {str(item).strip().lower() for item in expected}
                if isinstance(expected, list)
                else {str(expected).strip().lower()}
            )
            if instrument not in expected_instruments:
                return False
        elif key == "payoff_traits":
            expected_traits = (
                {str(item).strip().lower() for item in expected}
                if isinstance(expected, list)
                else {str(expected).strip().lower()}
            )
            if not expected_traits.issubset(payoff_traits):
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
        elif key == "methods":
            # Method-keyed dispatch for routes that collapse an analytical /
            # monte_carlo pair whose payoff / exercise / model-family tuple is
            # otherwise identical.  Mirror the route-registry semantics from
            # ``route_registry._matches_condition``: unknown method fails
            # closed so the ``when: default`` branch wins when no method was
            # threaded through.
            if method is None:
                return False
            normalized = str(method).strip()
            if isinstance(expected, list):
                if normalized not in expected:
                    return False
            elif normalized != expected:
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
