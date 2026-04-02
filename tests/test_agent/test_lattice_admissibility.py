"""Tests for the generalized lattice admissibility boundary."""

from __future__ import annotations

from trellis.agent.route_registry import (
    evaluate_route_admissibility,
    find_route_by_id,
    load_route_registry,
)
from trellis.agent.semantic_contract_compiler import compile_semantic_contract
from trellis.agent.semantic_contracts import (
    make_callable_bond_contract,
    make_ranked_observation_basket_contract,
)


def test_lattice_admissibility_accepts_callable_bond_contract():
    registry = load_route_registry()
    route = find_route_by_id("exercise_lattice", registry)
    assert route is not None

    blueprint = compile_semantic_contract(
        make_callable_bond_contract(
            description="Callable bond",
            observation_schedule=("2027-01-15", "2029-01-15", "2031-01-15"),
        )
    )

    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)

    assert decision.ok


def test_lattice_admissibility_rejects_multi_asset_basket_contract():
    registry = load_route_registry()
    route = find_route_by_id("exercise_lattice", registry)
    assert route is not None

    blueprint = compile_semantic_contract(
        make_ranked_observation_basket_contract(
            description="Basket with ranked observations",
            constituents=("AAPL", "MSFT", "NVDA", "AMZN", "META"),
            observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
        )
    )

    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)

    assert not decision.ok
    assert "ineligible_for_lattice_algebra:multi_asset" in decision.failures
