"""Shared parity harness for QUA-887 / QUA-907 Phase 1 route-match rewrites.

The Phase 1 plan (``doc/plan/active__contract-ir-compiler-retiring-route-registry.md``)
rewrites 16 instrument-keyed entries in
``trellis/agent/knowledge/canonical/routes.yaml`` into pattern-keyed match
clauses. Every rewrite must prove parity with the pre-rewrite match clause on
every existing ``(ProductIR, PricingPlan)`` fixture that reached the route
before. This module owns the reusable assertion the rewrite tickets call.

The harness loads the canonical route registry, constructs a variant registry
where one route's match clause has been replaced, and runs
``rank_primitive_routes`` under both registries for each fixture. It asserts
that the old and new match clauses produce identical ranked ``PrimitivePlan``
tuples along the dimensions that define dispatch: the top route id, the
resolved primitives (module / symbol / role), the resolved adapters, and the
blocker-driven admissibility signal.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import patch

from trellis.agent.codegen_guardrails import PrimitivePlan, rank_primitive_routes
from trellis.agent.knowledge.schema import ProductIR
from trellis.agent.quant import PricingPlan
from trellis.agent.route_registry import (
    RouteRegistry,
    RouteSpec,
    load_route_registry,
)


# Match-clause keys that a new_match_clause may specify. Any key outside this
# set is rejected so that the caller cannot silently drop a match dimension by
# typo and claim parity.
_SUPPORTED_MATCH_KEYS: frozenset[str] = frozenset(
    {
        "methods",
        "instruments",
        "exclude_instruments",
        "exercise",
        "exclude_exercise",
        "payoff_family",
        "payoff_traits",
        "required_market_data",
        "exclude_required_market_data",
    }
)


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a scalar or iterable of strings into an immutable tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    """Normalize optional match fields: ``None`` preserves 'match anything'."""
    if value is None:
        return None
    return _str_tuple(value)


def _build_variant_registry(
    registry: RouteRegistry,
    route_id: str,
    new_match_clause: dict,
) -> RouteRegistry:
    """Return a variant registry where ``route_id``'s match clause has been replaced.

    Only the nine match-clause fields on ``RouteSpec`` are overwritten; every
    other field (primitives, admissibility, scoring hints, conditional blocks)
    is preserved so that parity failures point at match-clause drift and not
    at unrelated metadata changes.
    """
    unknown_keys = set(new_match_clause) - _SUPPORTED_MATCH_KEYS
    if unknown_keys:
        raise ValueError(
            f"assert_route_match_parity: unsupported match-clause keys "
            f"{sorted(unknown_keys)}. Supported keys: "
            f"{sorted(_SUPPORTED_MATCH_KEYS)}."
        )

    found = False
    variant_routes: list[RouteSpec] = []
    for spec in registry.routes:
        if spec.id == route_id:
            found = True
            variant_routes.append(
                replace(
                    spec,
                    match_methods=_str_tuple(
                        new_match_clause.get("methods", spec.match_methods)
                    ),
                    match_instruments=_optional_str_tuple(
                        new_match_clause.get("instruments", spec.match_instruments)
                    ),
                    exclude_instruments=_str_tuple(
                        new_match_clause.get(
                            "exclude_instruments", spec.exclude_instruments
                        )
                    ),
                    match_exercise=_optional_str_tuple(
                        new_match_clause.get("exercise", spec.match_exercise)
                    ),
                    exclude_exercise=_str_tuple(
                        new_match_clause.get(
                            "exclude_exercise", spec.exclude_exercise
                        )
                    ),
                    match_payoff_family=_optional_str_tuple(
                        new_match_clause.get(
                            "payoff_family", spec.match_payoff_family
                        )
                    ),
                    match_payoff_traits=_optional_str_tuple(
                        new_match_clause.get(
                            "payoff_traits", spec.match_payoff_traits
                        )
                    ),
                    match_required_market_data=_optional_str_tuple(
                        new_match_clause.get(
                            "required_market_data",
                            spec.match_required_market_data,
                        )
                    ),
                    exclude_required_market_data=_optional_str_tuple(
                        new_match_clause.get(
                            "exclude_required_market_data",
                            spec.exclude_required_market_data,
                        )
                    ),
                )
            )
        else:
            variant_routes.append(spec)

    if not found:
        raise LookupError(
            f"assert_route_match_parity: route_id {route_id!r} not found in "
            f"the canonical route registry."
        )

    # Rebuild the method index so match_candidate_routes still finds the
    # variant spec on its new match_methods set. The index uses "" for routes
    # with no declared methods.
    method_index: dict[str, list[int]] = {}
    for idx, spec in enumerate(variant_routes):
        if spec.match_methods:
            for method in spec.match_methods:
                method_index.setdefault(method, []).append(idx)
        else:
            method_index.setdefault("", []).append(idx)
    frozen_index = {k: tuple(v) for k, v in method_index.items()}

    return RouteRegistry(
        routes=tuple(variant_routes),
        _method_index=frozen_index,
    )


def _primitive_identity(plan: PrimitivePlan) -> tuple[tuple[str, str, str], ...]:
    """Module / symbol / role tuples — the dispatch-critical primitive shape.

    Ordering is preserved because ``rank_primitive_routes`` uses it to drive
    deterministic assembly; a reorder would be a real dispatch change.
    """
    return tuple((p.module, p.symbol, p.role) for p in plan.primitives)


def _ranked_fingerprint(
    ranked: tuple[PrimitivePlan, ...],
) -> tuple[tuple[str, tuple[tuple[str, str, str], ...], tuple[str, ...], tuple[str, ...]], ...]:
    """Return a comparison tuple over the dispatch-critical dimensions.

    The fingerprint intentionally skips numeric scores (FP-sensitive) and
    backend metadata that is derived from the binding catalog, which the match
    clause does not control. It retains route id, primitive identity tuple,
    adapters, and blockers — the contract the parity harness defends.
    """
    return tuple(
        (
            plan.route,
            _primitive_identity(plan),
            tuple(plan.adapters),
            tuple(plan.blockers),
        )
        for plan in ranked
    )


def assert_route_match_parity(
    route_id: str,
    new_match_clause: dict,
    fixtures: list[tuple[ProductIR, PricingPlan]],
) -> None:
    """Assert that rewriting ``route_id``'s match clause preserves dispatch.

    Loads the current ``RouteSpec`` for ``route_id`` from the canonical
    registry, constructs a variant ``RouteSpec`` with the match clause replaced
    per ``new_match_clause``, and runs ``rank_primitive_routes`` under both
    registries for each ``(ProductIR, PricingPlan)`` fixture. Asserts identical
    ``PrimitivePlan.route`` (top of the ranked list), ``PrimitivePlan.primitives``
    as ``(module, symbol, role)`` tuples, ``PrimitivePlan.adapters``, and
    ``PrimitivePlan.blockers`` (the admissibility signal surfaced through
    primitive verification) across the full ranked tuple.

    Parameters
    ----------
    route_id:
        Canonical id of the route being rewritten. Must appear in the live
        registry loaded by ``load_route_registry()`` (aliases are not honored —
        the parity target must be a real route id).
    new_match_clause:
        Mapping of match-clause keys to their replacement values. Supported
        keys: ``methods``, ``instruments``, ``exclude_instruments``,
        ``exercise``, ``exclude_exercise``, ``payoff_family``,
        ``payoff_traits``, ``required_market_data``,
        ``exclude_required_market_data``. Keys omitted from the mapping are
        left unchanged from the canonical spec. Passing the empty dict is a
        legal no-op that exercises the trivial-parity path. A value of
        ``None`` for a key that supports ``match anything`` semantics (e.g.
        ``instruments``) preserves that behavior; ``()`` or ``[]`` collapses
        the match to the empty set.
    fixtures:
        Sequence of ``(ProductIR, PricingPlan)`` pairs to exercise under both
        the original and variant registries. Each fixture must independently
        pass parity; the first divergence raises ``AssertionError``.

    Raises
    ------
    LookupError
        When ``route_id`` is not present in the canonical registry.
    ValueError
        When ``new_match_clause`` contains keys outside the supported set.
    AssertionError
        When any fixture produces a divergent ranked plan between the old and
        new match clauses.
    """
    base_registry = load_route_registry()
    variant_registry = _build_variant_registry(
        base_registry, route_id, new_match_clause
    )

    for idx, (product_ir, pricing_plan) in enumerate(fixtures):
        with patch(
            "trellis.agent.route_registry.load_route_registry",
            return_value=base_registry,
        ):
            old_ranked = rank_primitive_routes(
                pricing_plan=pricing_plan,
                product_ir=product_ir,
            )
        with patch(
            "trellis.agent.route_registry.load_route_registry",
            return_value=variant_registry,
        ):
            new_ranked = rank_primitive_routes(
                pricing_plan=pricing_plan,
                product_ir=product_ir,
            )

        old_fp = _ranked_fingerprint(old_ranked)
        new_fp = _ranked_fingerprint(new_ranked)
        # Length mismatch is a divergence class worth calling out explicitly
        # because the fingerprint tuple dump is long enough that a missing
        # route at the head/tail is easy to miss.
        assert len(old_fp) == len(new_fp), (
            f"Route parity: ranked-list length mismatch on fixture[{idx}] "
            f"for route_id={route_id!r}: old_len={len(old_fp)} "
            f"new_len={len(new_fp)}\n"
            f"  ProductIR.instrument={getattr(product_ir, 'instrument', None)!r}\n"
            f"  PricingPlan.method={getattr(pricing_plan, 'method', None)!r}\n"
            f"  old routes: {[fp[0] for fp in old_fp]}\n"
            f"  new routes: {[fp[0] for fp in new_fp]}"
        )
        assert old_fp == new_fp, (
            f"Route parity divergence on fixture[{idx}] for route_id="
            f"{route_id!r}:\n"
            f"  old ranked: {old_fp}\n"
            f"  new ranked: {new_fp}\n"
            f"  ProductIR.instrument={getattr(product_ir, 'instrument', None)!r}\n"
            f"  ProductIR.payoff_family="
            f"{getattr(product_ir, 'payoff_family', None)!r}\n"
            f"  PricingPlan.method={getattr(pricing_plan, 'method', None)!r}"
        )
