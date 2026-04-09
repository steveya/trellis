"""Declarative route registry: loads canonical + discovered routes from YAML.

Replaces the hard-coded ``_candidate_routes`` / ``_route_components`` /
``_route_engine_family`` / ``_route_family`` functions that were formerly in
``codegen_guardrails.py`` with a data-driven registry that supports
two tiers:

* **Tier 1 — Canonical seed** routes in ``knowledge/canonical/routes.yaml``
* **Tier 2 — Discovered** routes in ``knowledge/routes/entries/*.yaml``

The registry is cached per (routes_yaml_mtime, discovered_dir_mtime,
repo_revision) and validated against the live import registry at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from trellis.agent.codegen_guardrails import PrimitiveRef
from trellis.agent.family_lowering_ir import EventAwareMonteCarloIR, EventAwarePDEIR
from trellis.agent.knowledge.import_registry import (
    get_repo_revision,
    is_valid_import,
    module_exists,
)
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.sensitivity_support import normalize_requested_outputs
from trellis.core.types import DslMeasure


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketDataAccessSpec:
    """Required and optional market-state access patterns for a route."""

    required: dict[str, tuple[str, ...]] = field(default_factory=dict)
    optional: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterBindingSpec:
    """Required and optional parameter bindings for a route."""

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConditionalRouteFamily:
    """A conditional override for route_family based on ProductIR traits."""

    when: dict[str, Any]
    route_family: str


@dataclass(frozen=True)
class ConditionalPrimitive:
    """A conditional block of primitives/adapters/notes applied when traits match."""

    when: dict[str, Any] | str  # dict of trait conditions, or "default"
    primitives: tuple[PrimitiveRef, ...] = ()
    adapters: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DynamicNote:
    """A note whose content is resolved at runtime from a Python function."""

    source: str       # module path, e.g. "trellis.agent.early_exercise_policy"
    function: str     # function name
    template: str     # f-string with {result} placeholder


@dataclass(frozen=True)
class RouteAdmissibilitySpec:
    """Typed tranche-1 capability contract for one route."""

    supported_control_styles: tuple[str, ...] = ("identity",)
    event_support: str = "none"  # "none" | "automatic"
    phase_sensitivity: str = "default_phase_order_only"  # "default_phase_order_only" | "ordered_same_day"
    multicurrency_support: str = "single_currency_only"  # "single_currency_only" | "native_payout_with_fx" | "reporting_currency_supported"
    supported_outputs: tuple[str, ...] = ("price",)
    supports_sensitivity_outputs: bool = True
    supported_state_tags: tuple[str, ...] = ()
    supported_process_families: tuple[str, ...] = ()
    supported_path_requirement_kinds: tuple[str, ...] = ()
    supported_operator_families: tuple[str, ...] = ()
    supported_event_transform_kinds: tuple[str, ...] = ()
    supports_calibration: bool = False


@dataclass(frozen=True)
class RouteAdmissibilityDecision:
    """Deterministic route-admissibility result."""

    ok: bool
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteSpec:
    """A single route declaration — canonical or discovered."""

    id: str
    engine_family: str
    route_family: str
    status: str                                     # "candidate" | "validated" | "promoted"
    confidence: float
    match_methods: tuple[str, ...]
    match_instruments: tuple[str, ...] | None       # None = any
    exclude_instruments: tuple[str, ...]
    match_exercise: tuple[str, ...] | None          # None = any
    exclude_exercise: tuple[str, ...]
    match_payoff_family: tuple[str, ...] | None
    match_payoff_traits: tuple[str, ...] | None
    match_required_market_data: tuple[str, ...] | None   # pricing_plan required data
    exclude_required_market_data: tuple[str, ...] | None
    primitives: tuple[PrimitiveRef, ...]
    conditional_primitives: tuple[ConditionalPrimitive, ...]
    conditional_route_family: tuple[ConditionalRouteFamily, ...] | None
    adapters: tuple[str, ...]
    notes: tuple[str, ...]
    dynamic_notes: tuple[DynamicNote, ...] = ()
    market_data_access: MarketDataAccessSpec = MarketDataAccessSpec()
    parameter_bindings: ParameterBindingSpec = ParameterBindingSpec()
    admissibility: RouteAdmissibilitySpec = RouteAdmissibilitySpec()
    score_hints: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    discovered_from: str | None = None
    reuse_module_paths: tuple[str, ...] = ()
    successful_builds: int = 0


@dataclass(frozen=True)
class RouteRegistry:
    """Merged canonical + discovered route registry."""

    routes: tuple[RouteSpec, ...]
    _method_index: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class RouteBindingAuthority:
    """Structured route-binding authority packet for build, replay, and review."""

    route_id: str
    route_family: str
    engine_family: str
    authority_kind: str
    exact_backend_fit: bool
    exact_target_refs: tuple[str, ...] = ()
    approved_modules: tuple[str, ...] = ()
    primitive_refs: tuple[str, ...] = ()
    helper_refs: tuple[str, ...] = ()
    validation_bundle_id: str = ""
    validation_check_ids: tuple[str, ...] = ()
    admissibility: RouteAdmissibilitySpec = RouteAdmissibilitySpec()
    admissibility_failures: tuple[str, ...] = ()
    canary_task_ids: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Freeze mutable provenance metadata for stable comparisons and traces."""
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance or {})))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_REGISTRY_CACHE: dict[tuple, RouteRegistry] = {}

_CANONICAL_PATH = (
    Path(__file__).resolve().parent / "knowledge" / "canonical" / "routes.yaml"
)
_DISCOVERED_DIR = (
    Path(__file__).resolve().parent / "knowledge" / "routes" / "entries"
)


def _cache_key() -> tuple:
    canonical_mtime = _CANONICAL_PATH.stat().st_mtime if _CANONICAL_PATH.exists() else 0
    discovered_mtime = 0.0
    if _DISCOVERED_DIR.exists():
        for entry in _DISCOVERED_DIR.iterdir():
            if entry.suffix in (".yaml", ".yml"):
                discovered_mtime = max(discovered_mtime, entry.stat().st_mtime)
    return (canonical_mtime, discovered_mtime, get_repo_revision())


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _parse_primitive(raw: dict) -> PrimitiveRef:
    return PrimitiveRef(
        module=raw["module"],
        symbol=raw["symbol"],
        role=raw["role"],
        required=raw.get("required", True),
        excluded=raw.get("excluded", False),
    )


def _parse_market_data_access(raw: dict | None) -> MarketDataAccessSpec:
    if not raw:
        return MarketDataAccessSpec()
    required = {
        k: tuple(v) if isinstance(v, list) else (v,)
        for k, v in raw.get("required", {}).items()
    }
    optional = {
        k: tuple(v) if isinstance(v, list) else (v,)
        for k, v in raw.get("optional", {}).items()
    }
    return MarketDataAccessSpec(required=required, optional=optional)


def _parse_parameter_bindings(raw: dict | None) -> ParameterBindingSpec:
    if not raw:
        return ParameterBindingSpec()
    return ParameterBindingSpec(
        required=tuple(raw.get("required", ())),
        optional=tuple(raw.get("optional", ())),
    )


def _parse_conditional_route_family(raw: list | None) -> tuple[ConditionalRouteFamily, ...] | None:
    if not raw:
        return None
    result = []
    for entry in raw:
        if isinstance(entry, dict) and "when" in entry and "route_family" in entry:
            when = entry["when"]
            # "default" string or a dict of conditions
            result.append(ConditionalRouteFamily(
                when=when if isinstance(when, dict) else str(when),
                route_family=entry["route_family"],
            ))
    return tuple(result) if result else None


def _parse_conditional_primitives(raw: list | None) -> tuple[ConditionalPrimitive, ...]:
    if not raw:
        return ()
    result = []
    for entry in raw:
        when = entry.get("when", "default")
        primitives = tuple(_parse_primitive(p) for p in entry.get("primitives", ()))
        adapters = tuple(entry.get("adapters", ()))
        notes = tuple(entry.get("notes", ()))
        result.append(ConditionalPrimitive(
            when=when,
            primitives=primitives,
            adapters=adapters,
            notes=notes,
        ))
    return tuple(result)


def _parse_dynamic_notes(raw: list | None) -> tuple[DynamicNote, ...]:
    if not raw:
        return ()
    return tuple(
        DynamicNote(source=entry["source"], function=entry["function"], template=entry["template"])
        for entry in raw
    )


_KNOWN_MEASURE_OUTPUTS = tuple(measure.value for measure in DslMeasure)


def _default_admissibility_for_route(route_id: str, engine_family: str) -> RouteAdmissibilitySpec:
    """Return conservative tranche-1 admissibility defaults."""
    defaults = {
        "analytical": RouteAdmissibilitySpec(
            supported_control_styles=("identity", "holder_max"),
            event_support="none",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("terminal_markov", "recombining_safe", "schedule_state"),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "pde_solver": RouteAdmissibilitySpec(
            supported_control_styles=("identity", "holder_max"),
            event_support="none",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("terminal_markov", "recombining_safe"),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=("black_scholes_1d",),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "lattice": RouteAdmissibilitySpec(
            supported_control_styles=("identity", "holder_max", "issuer_min"),
            event_support="none",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl", "exercise_boundary"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("terminal_markov", "recombining_safe", "schedule_state"),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=True,
        ),
        "monte_carlo": RouteAdmissibilitySpec(
            supported_control_styles=("identity",),
            event_support="automatic",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl"),
            supports_sensitivity_outputs=True,
            supported_state_tags=(
                "pathwise_only",
                "remaining_pool",
                "locked_cashflow_state",
                "terminal_markov",
                "recombining_safe",
                "schedule_state",
            ),
            supported_process_families=(),
            supported_path_requirement_kinds=("terminal_only", "full_path", "event_snapshots", "event_replay", "reducer_state"),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "exercise": RouteAdmissibilitySpec(
            supported_control_styles=("holder_max",),
            event_support="none",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl", "exercise_boundary"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("pathwise_only", "terminal_markov", "schedule_state"),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "qmc": RouteAdmissibilitySpec(
            supported_control_styles=("identity",),
            event_support="automatic",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("pathwise_only", "terminal_markov", "schedule_state"),
            supported_process_families=(),
            supported_path_requirement_kinds=("terminal_only", "full_path", "event_snapshots", "event_replay", "reducer_state"),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "fft_pricing": RouteAdmissibilitySpec(
            supported_control_styles=("identity",),
            event_support="none",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("terminal_markov",),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "copula": RouteAdmissibilitySpec(
            supported_control_styles=("identity",),
            event_support="automatic",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("pathwise_only", "schedule_state"),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
        "waterfall": RouteAdmissibilitySpec(
            supported_control_styles=("identity",),
            event_support="none",
            phase_sensitivity="default_phase_order_only",
            multicurrency_support="single_currency_only",
            supported_outputs=("price", "scenario_pnl", "cashflow_projection"),
            supports_sensitivity_outputs=True,
            supported_state_tags=("schedule_state",),
            supported_process_families=(),
            supported_path_requirement_kinds=(),
            supported_operator_families=(),
            supported_event_transform_kinds=(),
            supports_calibration=False,
        ),
    }
    spec = defaults.get(engine_family, RouteAdmissibilitySpec())
    if route_id in {"quanto_adjustment_analytical", "correlated_gbm_monte_carlo", "analytical_garman_kohlhagen"}:
        return RouteAdmissibilitySpec(
            supported_control_styles=spec.supported_control_styles,
            event_support=spec.event_support,
            phase_sensitivity=spec.phase_sensitivity,
            multicurrency_support="native_payout_with_fx",
            supported_outputs=spec.supported_outputs,
            supports_sensitivity_outputs=spec.supports_sensitivity_outputs,
            supported_state_tags=spec.supported_state_tags,
            supported_process_families=spec.supported_process_families,
            supported_path_requirement_kinds=spec.supported_path_requirement_kinds,
            supported_operator_families=spec.supported_operator_families,
            supported_event_transform_kinds=spec.supported_event_transform_kinds,
            supports_calibration=spec.supports_calibration,
        )
    return spec


def _parse_admissibility(raw: dict | None, *, route_id: str, engine_family: str) -> RouteAdmissibilitySpec:
    """Parse one route admissibility block with conservative defaults."""
    default = _default_admissibility_for_route(route_id, engine_family)
    if not raw:
        return default
    return RouteAdmissibilitySpec(
        supported_control_styles=_str_tuple(raw.get("control_styles")) or default.supported_control_styles,
        event_support=str(raw.get("event_support", default.event_support)).strip() or default.event_support,
        phase_sensitivity=str(raw.get("phase_sensitivity", default.phase_sensitivity)).strip() or default.phase_sensitivity,
        multicurrency_support=str(raw.get("multicurrency_support", default.multicurrency_support)).strip() or default.multicurrency_support,
        supported_outputs=_str_tuple(raw.get("supported_outputs")) or default.supported_outputs,
        supports_sensitivity_outputs=bool(raw.get("supports_sensitivity_outputs", default.supports_sensitivity_outputs)),
        supported_state_tags=_str_tuple(raw.get("supported_state_tags")) or default.supported_state_tags,
        supported_process_families=_str_tuple(raw.get("supported_process_families")) or default.supported_process_families,
        supported_path_requirement_kinds=_str_tuple(raw.get("supported_path_requirement_kinds")) or default.supported_path_requirement_kinds,
        supported_operator_families=_str_tuple(raw.get("supported_operator_families")) or default.supported_operator_families,
        supported_event_transform_kinds=_str_tuple(raw.get("supported_event_transform_kinds")) or default.supported_event_transform_kinds,
        supports_calibration=bool(raw.get("supports_calibration", default.supports_calibration)),
    )


def _str_tuple(val) -> tuple[str, ...]:
    if val is None:
        return ()
    if isinstance(val, str):
        return (val,)
    return tuple(val)


def _optional_str_tuple(val) -> tuple[str, ...] | None:
    if val is None:
        return None
    return _str_tuple(val)


def _parse_route(raw: dict) -> RouteSpec:
    match = raw.get("match", {})
    route_id = raw["id"]
    engine_family = raw["engine_family"]
    return RouteSpec(
        id=route_id,
        engine_family=engine_family,
        route_family=raw.get("route_family", raw["engine_family"]),
        status=raw.get("status", "promoted"),
        confidence=float(raw.get("confidence", 1.0)),
        match_methods=_str_tuple(match.get("methods")),
        match_instruments=_optional_str_tuple(match.get("instruments")),
        exclude_instruments=_str_tuple(match.get("exclude_instruments")),
        match_exercise=_optional_str_tuple(match.get("exercise")),
        exclude_exercise=_str_tuple(match.get("exclude_exercise")),
        match_payoff_family=_optional_str_tuple(match.get("payoff_family")),
        match_payoff_traits=_optional_str_tuple(match.get("payoff_traits")),
        match_required_market_data=_optional_str_tuple(match.get("required_market_data")),
        exclude_required_market_data=_optional_str_tuple(match.get("exclude_required_market_data")),
        primitives=tuple(_parse_primitive(p) for p in raw.get("primitives", ())),
        conditional_primitives=_parse_conditional_primitives(raw.get("conditional_primitives")),
        conditional_route_family=_parse_conditional_route_family(raw.get("conditional_route_family")),
        adapters=tuple(raw.get("adapters", ())),
        notes=tuple(raw.get("notes", ())),
        dynamic_notes=_parse_dynamic_notes(raw.get("dynamic_notes")),
        market_data_access=_parse_market_data_access(raw.get("market_data_access")),
        parameter_bindings=_parse_parameter_bindings(raw.get("parameter_bindings")),
        admissibility=_parse_admissibility(raw.get("admissibility"), route_id=route_id, engine_family=engine_family),
        score_hints=dict(raw.get("score_hints", {})),
        aliases=_str_tuple(raw.get("aliases")),
        discovered_from=raw.get("discovered_from"),
        reuse_module_paths=_str_tuple(raw.get("reuse_module_paths")),
        successful_builds=int(raw.get("successful_builds", 0)),
    )


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_route_registry() -> RouteRegistry:
    """Load canonical + discovered routes, merged and indexed.

    Results are cached per (routes.yaml mtime, discovered dir mtime, repo rev).
    """
    key = _cache_key()
    cached = _REGISTRY_CACHE.get(key)
    if cached is not None:
        return cached

    routes: list[RouteSpec] = []

    # Tier 1: canonical
    if _CANONICAL_PATH.exists():
        with open(_CANONICAL_PATH) as fh:
            data = yaml.safe_load(fh)
        for raw in data.get("routes", ()):
            routes.append(_parse_route(raw))

    # Tier 2: discovered
    if _DISCOVERED_DIR.exists():
        for entry in sorted(_DISCOVERED_DIR.iterdir()):
            if entry.suffix not in (".yaml", ".yml"):
                continue
            with open(entry) as fh:
                raw = yaml.safe_load(fh)
            if raw:
                routes.append(_parse_route(raw))

    # Build method index
    method_index: dict[str, list[int]] = {}
    for idx, route in enumerate(routes):
        for method in route.match_methods:
            method_index.setdefault(method, []).append(idx)
        if not route.match_methods:
            method_index.setdefault("", []).append(idx)
    frozen_index = {k: tuple(v) for k, v in method_index.items()}

    registry = RouteRegistry(routes=tuple(routes), _method_index=frozen_index)
    _REGISTRY_CACHE[key] = registry
    return registry


def clear_route_registry_cache() -> None:
    """Clear the route registry cache."""
    _REGISTRY_CACHE.clear()


# ---------------------------------------------------------------------------
# Registry validation (catches YAML ↔ repo drift)
# ---------------------------------------------------------------------------

def validate_registry(registry: RouteRegistry | None = None) -> tuple[str, ...]:
    """Check every primitive in the registry against the import registry.

    Returns a tuple of error strings.  Empty means all primitives are valid.
    """
    if registry is None:
        registry = load_route_registry()
    errors: list[str] = []
    for route in registry.routes:
        for prim in route.primitives:
            if not module_exists(prim.module):
                errors.append(f"Route '{route.id}': module '{prim.module}' does not exist")
            elif prim.required and not is_valid_import(prim.module, prim.symbol):
                errors.append(f"Route '{route.id}': symbol '{prim.module}.{prim.symbol}' not exported")
        for cond in route.conditional_primitives:
            for prim in cond.primitives:
                if not module_exists(prim.module):
                    errors.append(f"Route '{route.id}' (conditional): module '{prim.module}' does not exist")
                elif prim.required and not is_valid_import(prim.module, prim.symbol):
                    errors.append(f"Route '{route.id}' (conditional): symbol '{prim.module}.{prim.symbol}' not exported")
    return tuple(errors)


# ---------------------------------------------------------------------------
# Route matching
# ---------------------------------------------------------------------------

def match_candidate_routes(
    registry: RouteRegistry,
    method: str,
    product_ir: ProductIR | None,
    *,
    pricing_plan=None,
    promoted_only: bool = True,
) -> tuple[RouteSpec, ...]:
    """Filter routes by match conditions.

    When ``promoted_only`` is True (default for live builds), only routes with
    ``status == "promoted"`` are returned.  Set to False for gap analysis.
    """
    method = normalize_method(method) if method else ""
    instrument = getattr(product_ir, "instrument", None) if product_ir is not None else None
    exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
    payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
    payoff_traits = set(getattr(product_ir, "payoff_traits", ())) if product_ir is not None else set()
    required_market_data = set()
    if pricing_plan is not None:
        required_market_data = set(getattr(pricing_plan, "required_market_data", ()) or ())

    # Collect candidate indices from method index
    candidate_indices = set()
    for idx in registry._method_index.get(method, ()):
        candidate_indices.add(idx)
    for idx in registry._method_index.get("", ()):
        candidate_indices.add(idx)

    matches: list[RouteSpec] = []
    for idx in sorted(candidate_indices):
        route = registry.routes[idx]

        # Status gate
        if promoted_only and route.status != "promoted":
            continue

        # Method match
        if route.match_methods and method not in route.match_methods:
            continue

        # Instrument/payoff family/payoff traits — OR match
        # The old _candidate_routes checks instrument OR payoff_family OR
        # payoff_traits with OR semantics.  Any positive match qualifies;
        # exclusions are always AND (hard veto).
        if route.exclude_instruments and instrument in route.exclude_instruments:
            continue

        has_positive_filter = (
            route.match_instruments is not None
            or route.match_payoff_family is not None
            or route.match_payoff_traits is not None
        )
        if has_positive_filter:
            instrument_ok = (
                route.match_instruments is not None
                and instrument in route.match_instruments
            )
            payoff_family_ok = (
                route.match_payoff_family is not None
                and payoff_family in route.match_payoff_family
            )
            payoff_traits_ok = (
                route.match_payoff_traits is not None
                and bool(payoff_traits.intersection(route.match_payoff_traits))
            )
            if not (instrument_ok or payoff_family_ok or payoff_traits_ok):
                continue

        # Exercise include/exclude
        if route.match_exercise is not None and exercise not in route.match_exercise:
            continue
        if route.exclude_exercise and exercise in route.exclude_exercise:
            continue

        # Required market data match
        if route.match_required_market_data is not None:
            if not all(md in required_market_data for md in route.match_required_market_data):
                continue
        if route.exclude_required_market_data is not None:
            if any(md in required_market_data for md in route.exclude_required_market_data):
                continue

        matches.append(route)

    return tuple(matches)


def resolve_route_primitives(
    spec: RouteSpec,
    product_ir: ProductIR | None,
) -> tuple[PrimitiveRef, ...]:
    """Apply conditional_primitives based on ProductIR traits.

    Returns the final set of primitives for this route + product combination.
    """
    if not spec.conditional_primitives:
        return spec.primitives

    exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
    payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
    model_family = getattr(product_ir, "model_family", "generic") if product_ir is not None else "generic"

    for cond in spec.conditional_primitives:
        if isinstance(cond.when, str) and cond.when == "default":
            # Default branch — use if no prior branch matched
            return cond.primitives if cond.primitives else spec.primitives
        if isinstance(cond.when, dict):
            if _matches_condition(cond.when, payoff_family, exercise, model_family, product_ir):
                return cond.primitives if cond.primitives else spec.primitives

    # No conditional matched — use base primitives
    return spec.primitives


def resolve_route_adapters(
    spec: RouteSpec,
    product_ir: ProductIR | None,
) -> tuple[str, ...]:
    """Resolve adapters from conditional_primitives, falling back to base."""
    if not spec.conditional_primitives:
        return spec.adapters

    exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
    payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
    model_family = getattr(product_ir, "model_family", "generic") if product_ir is not None else "generic"

    for cond in spec.conditional_primitives:
        if isinstance(cond.when, str) and cond.when == "default":
            return cond.adapters if cond.adapters else spec.adapters
        if isinstance(cond.when, dict):
            if _matches_condition(cond.when, payoff_family, exercise, model_family, product_ir):
                return cond.adapters if cond.adapters else spec.adapters

    return spec.adapters


def resolve_route_notes(
    spec: RouteSpec,
    product_ir: ProductIR | None,
) -> tuple[str, ...]:
    """Resolve notes from conditional_primitives + dynamic notes."""
    base_notes = list(spec.notes)

    if spec.conditional_primitives:
        exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
        payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
        model_family = getattr(product_ir, "model_family", "generic") if product_ir is not None else "generic"

        for cond in spec.conditional_primitives:
            matched = False
            if isinstance(cond.when, str) and cond.when == "default":
                matched = True
            elif isinstance(cond.when, dict):
                matched = _matches_condition(cond.when, payoff_family, exercise, model_family, product_ir)
            if matched and cond.notes:
                base_notes = list(cond.notes)
                break

    # Resolve dynamic notes
    for dn in spec.dynamic_notes:
        try:
            import importlib
            mod = importlib.import_module(dn.source)
            fn = getattr(mod, dn.function)
            result = fn()
            base_notes.append(dn.template.format(result=result))
        except Exception:
            pass  # skip unresolvable dynamic notes

    return tuple(base_notes)


def resolve_route_family(
    spec: RouteSpec,
    product_ir: ProductIR | None,
) -> str:
    """Resolve the route family, applying conditional overrides."""
    if not spec.conditional_route_family:
        return spec.route_family

    exercise = getattr(product_ir, "exercise_style", "none") if product_ir is not None else "none"
    payoff_family = getattr(product_ir, "payoff_family", "") if product_ir is not None else ""
    model_family = getattr(product_ir, "model_family", "generic") if product_ir is not None else "generic"

    default_family = spec.route_family
    explicit_default = None
    for crf in spec.conditional_route_family:
        if isinstance(crf.when, str) and crf.when == "default":
            explicit_default = crf.route_family
            continue
        if isinstance(crf.when, dict) and _matches_condition(crf.when, payoff_family, exercise, model_family, product_ir):
            return crf.route_family

    return explicit_default if explicit_default is not None else default_family


def get_route_modules(route_id: str, registry: RouteRegistry | None = None) -> tuple[str, ...]:
    """Return the module paths for a route's primitives."""
    if registry is None:
        registry = load_route_registry()
    for route in registry.routes:
        if route.id == route_id or route_id in route.aliases:
            return tuple(sorted({p.module for p in route.primitives}))
    return ()


def find_route_by_id(route_id: str, registry: RouteRegistry | None = None) -> RouteSpec | None:
    """Look up a route by ID or alias."""
    if registry is None:
        registry = load_route_registry()
    for route in registry.routes:
        if route.id == route_id or route_id in route.aliases:
            return route
    return None


def compile_route_binding_authority(
    *,
    generation_plan=None,
    validation_contract=None,
    semantic_blueprint=None,
    product_ir: ProductIR | None = None,
    request=None,
    registry: RouteRegistry | None = None,
) -> RouteBindingAuthority | None:
    """Compile the structured route-binding authority packet for one request."""
    route_id = _route_id_for_authority(generation_plan=generation_plan, semantic_blueprint=semantic_blueprint)
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    route_spec = find_route_by_id(route_id, registry) if route_id else None
    if not route_id and primitive_plan is None and route_spec is None:
        return None

    route_family = (
        str(getattr(route_spec, "route_family", "") or "").strip()
        or str(getattr(primitive_plan, "route_family", "") or "").strip()
        or str(getattr(validation_contract, "route_family", "") or "").strip()
    )
    engine_family = (
        str(getattr(route_spec, "engine_family", "") or "").strip()
        or str(getattr(primitive_plan, "engine_family", "") or "").strip()
        or route_family
    )
    exact_target_refs = tuple(getattr(generation_plan, "lane_exact_binding_refs", ()) or ())
    helper_refs = tuple(
        dict.fromkeys(
            str(ref).strip()
            for ref in (
                getattr(generation_plan, "lowering_helper_refs", ())
                or getattr(getattr(semantic_blueprint, "dsl_lowering", None), "helper_refs", ())
                or ()
            )
            if str(ref).strip()
        )
    )
    primitive_refs = _primitive_refs_for(
        primitive_plan=primitive_plan,
        route_spec=route_spec,
        product_ir=product_ir,
    )
    approved_modules = tuple(
        dict.fromkeys(
            str(module).strip()
            for module in (getattr(generation_plan, "approved_modules", ()) or ())
            if str(module).strip()
        )
    )
    exact_backend_fit = bool(
        str(getattr(generation_plan, "lane_plan_kind", "") or "").strip() == "exact_target_binding"
        or exact_target_refs
        or helper_refs
    )
    authority_kind = "exact_backend_fit" if exact_backend_fit else "route_registry_binding"
    admissibility = (
        getattr(route_spec, "admissibility", None)
        or _default_admissibility_for_route(route_id or route_family or engine_family, engine_family or route_family or "")
    )
    admissibility_failures = tuple(
        str(item).strip()
        for item in (getattr(validation_contract, "admissibility_failures", ()) or ())
        if str(item).strip()
    )
    validation_bundle_id = str(getattr(validation_contract, "bundle_id", "") or "")
    validation_check_ids = tuple(
        str(getattr(check, "check_id", "") or "").strip()
        for check in (getattr(validation_contract, "deterministic_checks", ()) or ())
        if str(getattr(check, "check_id", "") or "").strip()
    )
    canary_task_ids = _route_canary_task_ids(
        route_id=route_id,
        route_family=route_family,
        engine_family=engine_family,
        generation_plan=generation_plan,
        validation_contract=validation_contract,
        semantic_blueprint=semantic_blueprint,
        product_ir=product_ir,
        request=request,
    )
    provenance = {
        "semantic_contract_id": str(getattr(semantic_blueprint, "semantic_id", "") or ""),
        "requested_instrument_type": str(getattr(request, "instrument_type", "") or ""),
        "product_instrument_type": str(getattr(product_ir, "instrument", "") or ""),
        "method": str(getattr(generation_plan, "method", "") or ""),
        "lane_family": str(getattr(generation_plan, "lane_family", "") or ""),
        "lane_plan_kind": str(getattr(generation_plan, "lane_plan_kind", "") or ""),
        "repo_revision": str(getattr(generation_plan, "repo_revision", "") or ""),
    }
    return RouteBindingAuthority(
        route_id=route_id or str(getattr(primitive_plan, "route", "") or ""),
        route_family=route_family,
        engine_family=engine_family,
        authority_kind=authority_kind,
        exact_backend_fit=exact_backend_fit,
        exact_target_refs=exact_target_refs,
        approved_modules=approved_modules,
        primitive_refs=primitive_refs,
        helper_refs=helper_refs,
        validation_bundle_id=validation_bundle_id,
        validation_check_ids=validation_check_ids,
        admissibility=admissibility,
        admissibility_failures=admissibility_failures,
        canary_task_ids=canary_task_ids,
        provenance=provenance,
    )


def route_binding_authority_summary(
    authority: RouteBindingAuthority | None,
) -> dict[str, object] | None:
    """Project the route-binding authority packet onto YAML-safe primitives."""
    if authority is None:
        return None
    return {
        "route_id": authority.route_id,
        "route_family": authority.route_family,
        "engine_family": authority.engine_family,
        "authority_kind": authority.authority_kind,
        "exact_backend_fit": authority.exact_backend_fit,
        "exact_target_refs": list(authority.exact_target_refs),
        "approved_modules": list(authority.approved_modules),
        "primitive_refs": list(authority.primitive_refs),
        "helper_refs": list(authority.helper_refs),
        "validation_bundle_id": authority.validation_bundle_id,
        "validation_check_ids": list(authority.validation_check_ids),
        "admissibility": {
            "supported_control_styles": list(authority.admissibility.supported_control_styles),
            "event_support": authority.admissibility.event_support,
            "phase_sensitivity": authority.admissibility.phase_sensitivity,
            "multicurrency_support": authority.admissibility.multicurrency_support,
            "supported_outputs": list(authority.admissibility.supported_outputs),
            "supports_sensitivity_outputs": authority.admissibility.supports_sensitivity_outputs,
            "supported_state_tags": list(authority.admissibility.supported_state_tags),
            "supported_process_families": list(authority.admissibility.supported_process_families),
            "supported_path_requirement_kinds": list(authority.admissibility.supported_path_requirement_kinds),
            "supported_operator_families": list(authority.admissibility.supported_operator_families),
            "supported_event_transform_kinds": list(authority.admissibility.supported_event_transform_kinds),
            "supports_calibration": authority.admissibility.supports_calibration,
        },
        "admissibility_failures": list(authority.admissibility_failures),
        "canary_task_ids": list(authority.canary_task_ids),
        "provenance": dict(authority.provenance),
    }


def _route_id_for_authority(*, generation_plan=None, semantic_blueprint=None) -> str:
    """Return the best available selected route identifier."""
    route_id = str(getattr(generation_plan, "lowering_route_id", "") or "").strip()
    if route_id:
        return route_id
    primitive_plan = getattr(generation_plan, "primitive_plan", None)
    route_id = str(getattr(primitive_plan, "route", "") or "").strip()
    if route_id:
        return route_id
    lowering = getattr(semantic_blueprint, "dsl_lowering", None)
    return str(getattr(lowering, "route_id", "") or "").strip()


def _primitive_refs_for(
    *,
    primitive_plan=None,
    route_spec: RouteSpec | None,
    product_ir: ProductIR | None,
) -> tuple[str, ...]:
    """Return fully qualified primitive refs that define the checked backend surface."""
    if primitive_plan is not None and getattr(primitive_plan, "primitives", None):
        primitives = primitive_plan.primitives
    elif route_spec is not None:
        primitives = resolve_route_primitives(route_spec, product_ir)
    else:
        primitives = ()
    return tuple(
        dict.fromkeys(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitives
            if str(getattr(primitive, "module", "") or "").strip()
            and str(getattr(primitive, "symbol", "") or "").strip()
        )
    )


def _route_canary_task_ids(
    *,
    route_id: str,
    route_family: str,
    engine_family: str,
    generation_plan=None,
    validation_contract=None,
    semantic_blueprint=None,
    product_ir: ProductIR | None,
    request=None,
) -> tuple[str, ...]:
    """Return curated canary task IDs that cover the current route authority surface."""
    canary_path = Path(__file__).resolve().parents[2] / "CANARY_TASKS.yaml"
    if not canary_path.exists():
        return ()
    try:
        raw = yaml.safe_load(canary_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ()
    normalized_terms = {
        str(value).strip().lower()
        for value in (
            route_id,
            route_family,
            engine_family,
            getattr(generation_plan, "method", None),
            getattr(validation_contract, "instrument_type", None),
            getattr(product_ir, "instrument", None),
            getattr(semantic_blueprint, "semantic_id", None),
            getattr(request, "instrument_type", None),
        )
        if str(value or "").strip()
    }
    aliases = {
        "callable_bond": "callable",
        "puttable_bond": "puttable",
        "credit_default_swap": "credit_default_swap",
        "quanto_option": "quanto_option",
        "swaption": "swaption",
        "basket_option": "basket_option",
    }
    normalized_terms.update(
        alias
        for key, alias in aliases.items()
        if key in normalized_terms and alias
    )
    matches: list[str] = []
    for canary in raw.get("canary_set", ()) or ():
        covers = {
            str(item).strip().lower()
            for item in (canary.get("covers") or ())
            if str(item).strip()
        }
        if not covers:
            continue
        if normalized_terms.intersection(covers):
            task_id = str(canary.get("id", "") or "").strip()
            if task_id and task_id not in matches:
                matches.append(task_id)
    return tuple(matches)


def evaluate_route_admissibility(
    spec: RouteSpec,
    *,
    semantic_blueprint=None,
    product_ir: ProductIR | None = None,
) -> RouteAdmissibilityDecision:
    """Evaluate typed route admissibility against one semantic blueprint."""
    if semantic_blueprint is None:
        return RouteAdmissibilityDecision(ok=True)

    contract = getattr(semantic_blueprint, "contract", None)
    valuation_context = getattr(semantic_blueprint, "valuation_context", None)
    market_binding_spec = getattr(semantic_blueprint, "market_binding_spec", None)
    calibration_step = getattr(semantic_blueprint, "calibration_step", None)
    if contract is None:
        return RouteAdmissibilityDecision(ok=True)

    product = getattr(contract, "product", None)
    if product is None:
        return RouteAdmissibilityDecision(ok=True)
    if product_ir is None:
        product_ir = getattr(semantic_blueprint, "product_ir", None)
    family_ir = getattr(getattr(semantic_blueprint, "dsl_lowering", None), "family_ir", None)

    admissibility = spec.admissibility
    failures: list[str] = []

    control_style = str(getattr(getattr(product, "controller_protocol", None), "controller_style", "identity")).strip() or "identity"
    if isinstance(family_ir, (EventAwarePDEIR, EventAwareMonteCarloIR)):
        lowered_control_style = str(
            getattr(getattr(family_ir, "control_spec", None), "control_style", "") or ""
        ).strip()
        if lowered_control_style:
            control_style = lowered_control_style
    if admissibility.supported_control_styles and control_style not in admissibility.supported_control_styles:
        failures.append(f"unsupported_control_style:{control_style}")

    phase_order = tuple(getattr(getattr(product, "timeline", None), "phase_order", ()) or ())
    try:
        from trellis.agent.semantic_contracts import DEFAULT_PHASE_ORDER
    except Exception:  # pragma: no cover - defensive only
        DEFAULT_PHASE_ORDER = ("event", "observation", "decision", "determination", "settlement", "state_update")
    if (
        admissibility.phase_sensitivity == "default_phase_order_only"
        and phase_order
        and phase_order != DEFAULT_PHASE_ORDER
    ):
        failures.append("unsupported_phase_order:custom_same_day_order")

    if _has_automatic_events(product) and admissibility.event_support == "none":
        failures.append("unsupported_event_support:automatic_triggers")

    if spec.engine_family == "lattice":
        try:
            from trellis.models.trees.algebra import lattice_algebra_eligible

            eligibility = lattice_algebra_eligible(product=product, product_ir=product_ir)
            failures.extend(
                f"ineligible_for_lattice_algebra:{reason}"
                for reason in eligibility.reasons
            )
        except Exception:  # pragma: no cover - admissibility should stay fail-open on import issues
            pass

    requested_outputs = normalize_requested_outputs(
        getattr(semantic_blueprint, "requested_outputs", ())
        or getattr(valuation_context, "requested_outputs", ())
    )
    for output in requested_outputs:
        if output in _KNOWN_MEASURE_OUTPUTS:
            if output == DslMeasure.PRICE.value:
                if output not in admissibility.supported_outputs:
                    failures.append(f"unsupported_output:{output}")
                continue
            if not admissibility.supports_sensitivity_outputs and output not in admissibility.supported_outputs:
                failures.append(f"unsupported_output:{output}")
            continue
        if output not in admissibility.supported_outputs:
            failures.append(f"unsupported_output:{output}")

    supported_tags = set(admissibility.supported_state_tags)
    if supported_tags:
        if isinstance(family_ir, EventAwarePDEIR):
            state_tags = {
                str(tag)
                for tag in getattr(getattr(family_ir, "state_spec", None), "state_tags", ()) or ()
                if str(tag).strip()
            }
        elif isinstance(family_ir, EventAwareMonteCarloIR):
            state_tags = {
                str(tag)
                for tag in getattr(getattr(family_ir, "state_spec", None), "state_tags", ()) or ()
                if str(tag).strip()
            }
        elif getattr(family_ir, "state_tags", ()):
            state_tags = {
                str(tag)
                for tag in getattr(family_ir, "state_tags", ()) or ()
                if str(tag).strip()
            }
        else:
            state_tags = {
                str(tag)
                for field_spec in getattr(product, "state_fields", ()) or ()
                for tag in getattr(field_spec, "tags", ()) or ()
            }
        for tag in sorted(state_tags):
            if tag not in supported_tags:
                failures.append(f"unsupported_state_tag:{tag}")

    if isinstance(family_ir, EventAwarePDEIR):
        supported_operators = set(admissibility.supported_operator_families)
        operator_family = str(getattr(getattr(family_ir, "operator_spec", None), "operator_family", "")).strip()
        if supported_operators and operator_family and operator_family not in supported_operators:
            failures.append(f"unsupported_operator_family:{operator_family}")

        supported_transforms = set(admissibility.supported_event_transform_kinds)
        if supported_transforms or family_ir.event_transform_kinds:
            for kind in family_ir.event_transform_kinds:
                if kind not in supported_transforms:
                    failures.append(f"unsupported_event_transform_kind:{kind}")
    if isinstance(family_ir, EventAwareMonteCarloIR):
        supported_processes = set(admissibility.supported_process_families)
        process_family = str(getattr(getattr(family_ir, "process_spec", None), "process_family", "")).strip()
        if supported_processes and process_family and process_family not in supported_processes:
            failures.append(f"unsupported_process_family:{process_family}")

        supported_requirements = set(admissibility.supported_path_requirement_kinds)
        requirement_kind = str(
            getattr(getattr(family_ir, "path_requirement_spec", None), "requirement_kind", "")
        ).strip()
        if supported_requirements and requirement_kind and requirement_kind not in supported_requirements:
            failures.append(f"unsupported_path_requirement_kind:{requirement_kind}")

    if calibration_step is not None and not admissibility.supports_calibration:
        failures.append("unsupported_calibration_step")

    if _requires_cross_currency_support(product, market_binding_spec, valuation_context):
        support = admissibility.multicurrency_support
        if support == "single_currency_only":
            failures.append("unsupported_multicurrency:cross_currency_contract")
        elif (
            support == "native_payout_with_fx"
            and _requires_reporting_currency_conversion(product, market_binding_spec, valuation_context)
        ):
            failures.append("unsupported_reporting_policy:non_native_reporting_currency")

    return RouteAdmissibilityDecision(
        ok=not failures,
        failures=tuple(dict.fromkeys(failures)),
    )


def _has_automatic_events(product) -> bool:
    """Return whether the semantic product relies on automatic event transitions."""
    controller_style = str(getattr(getattr(product, "controller_protocol", None), "controller_style", "identity")).strip() or "identity"
    if controller_style != "identity":
        return False
    if bool(getattr(product, "event_machine", None)):
        transitions = tuple(getattr(getattr(product, "event_machine", None), "transitions", ()) or ())
        return bool(transitions)
    if str(getattr(product, "path_dependence", "")).strip() == "path_dependent":
        typed_surface_present = any(
            (
                tuple(getattr(product, "state_fields", ()) or ()),
                tuple(getattr(product, "observables", ()) or ()),
                tuple(getattr(product, "obligations", ()) or ()),
            )
        )
        if typed_surface_present:
            return True
    return bool(getattr(product, "schedule_dependence", False) and getattr(product, "event_transitions", ()))


def _requires_cross_currency_support(product, market_binding_spec, valuation_context) -> bool:
    """Return whether the semantic/valuation contract is cross-currency."""
    underlier_structure = str(getattr(product, "underlier_structure", "")).lower()
    payoff_traits = {str(item).lower() for item in getattr(product, "payoff_traits", ()) or ()}
    if "cross_currency" in underlier_structure or "fx_translation" in payoff_traits:
        return True
    bindings = getattr(market_binding_spec, "bindings", ()) or ()
    for binding in bindings:
        if str(getattr(binding, "capability", "")) == "fx_rates":
            return True
    reporting_currency = str(getattr(getattr(valuation_context, "reporting_policy", None), "reporting_currency", "")).strip()
    payment_currency = str(getattr(getattr(product, "conventions", None), "payment_currency", "")).strip()
    return bool(payment_currency and reporting_currency and payment_currency != reporting_currency)


def _requires_reporting_currency_conversion(product, market_binding_spec, valuation_context) -> bool:
    """Return whether reporting requires conversion away from the contract payout currency."""
    reporting_currency = str(getattr(getattr(valuation_context, "reporting_policy", None), "reporting_currency", "")).strip()
    payment_currency = str(getattr(getattr(product, "conventions", None), "payment_currency", "")).strip()
    if not reporting_currency or not payment_currency:
        return False
    return reporting_currency != payment_currency


# ---------------------------------------------------------------------------
# Condition matching helper
# ---------------------------------------------------------------------------

def _matches_condition(
    when: dict[str, Any],
    payoff_family: str,
    exercise_style: str,
    model_family: str,
    product_ir: ProductIR | None,
) -> bool:
    """Check whether a ProductIR matches a conditional 'when' clause."""
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
            if product_ir is not None:
                if product_ir.schedule_dependence != expected:
                    return False
        else:
            # Unknown condition key — conservative: don't match
            return False
    return True


def _expanded_payoff_families(
    payoff_family: str,
    product_ir: ProductIR | None,
) -> frozenset[str]:
    """Return payoff-family aliases used for route-condition matching.

    The route registry stores helper-backed fixed-income exercise routes under
    the callable-bond family names. Puttable bonds share the same checked helper
    surface but decompose to ``puttable_fixed_income``. Expand that family here
    so the helper route stays reachable without mutating canonical route YAML.
    """
    families = {str(payoff_family or "")}
    instrument = str(getattr(product_ir, "instrument", "") or "").strip().lower()

    if instrument == "puttable_bond" or payoff_family == "puttable_fixed_income":
        families.update({"puttable_fixed_income", "callable_fixed_income", "callable_bond", "bond"})
    elif instrument == "callable_bond" or payoff_family == "callable_fixed_income":
        families.update({"callable_fixed_income", "callable_bond", "bond"})

    families.discard("")
    return frozenset(families)
