"""Tests for the generalized lattice admissibility boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from trellis.agent.route_registry import (
    evaluate_route_admissibility,
    find_route_by_id,
    load_route_registry,
)
from trellis.agent.semantic_contract_compiler import compile_semantic_contract
from trellis.agent.semantic_contracts import (
    make_callable_bond_contract,
    make_vanilla_option_contract,
    make_ranked_observation_basket_contract,
)


def test_all_canonical_routes_declare_explicit_admissibility():
    routes_path = (
        Path(__file__).resolve().parents[2]
        / "trellis"
        / "agent"
        / "knowledge"
        / "canonical"
        / "routes.yaml"
    )
    payload = yaml.safe_load(routes_path.read_text()) or {}
    routes = tuple(payload.get("routes", ()) or ())

    missing = [
        str(route.get("id", ""))
        for route in routes
        if isinstance(route, dict) and "admissibility" not in route
    ]

    assert missing == []


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


def test_lattice_admissibility_rejects_non_markov_contract():
    registry = load_route_registry()
    route = find_route_by_id("exercise_lattice", registry)
    assert route is not None

    blueprint = compile_semantic_contract(
        make_vanilla_option_contract(
            description="American call on AAPL",
            underliers=("AAPL",),
            observation_schedule=("2025-11-15",),
        )
    )
    blueprint = replace(
        blueprint,
        contract=replace(
            blueprint.contract,
            product=replace(
                blueprint.contract.product,
                path_dependence="path_dependent",
            ),
        ),
    )

    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)

    assert not decision.ok
    assert "ineligible_for_lattice_algebra:non_markov" in decision.failures


def test_lattice_admissibility_rejects_unsupported_model_family():
    registry = load_route_registry()
    route = find_route_by_id("exercise_lattice", registry)
    assert route is not None

    blueprint = compile_semantic_contract(
        make_vanilla_option_contract(
            description="American call on AAPL",
            underliers=("AAPL",),
            observation_schedule=("2025-11-15",),
        )
    )
    blueprint = replace(
        blueprint,
        contract=replace(
            blueprint.contract,
            product=replace(
                blueprint.contract.product,
                model_family="hjm",
            ),
        ),
    )

    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)

    assert not decision.ok
    assert "ineligible_for_lattice_algebra:unsupported_model_family" in decision.failures


def test_lattice_admissibility_rejects_multi_controller_contract():
    registry = load_route_registry()
    route = find_route_by_id("exercise_lattice", registry)
    assert route is not None

    blueprint = compile_semantic_contract(
        make_callable_bond_contract(
            description="Callable bond",
            observation_schedule=("2027-01-15", "2029-01-15", "2031-01-15"),
        )
    )
    blueprint = replace(
        blueprint,
        contract=replace(
            blueprint.contract,
            product=replace(
                blueprint.contract.product,
                controller_protocol=replace(
                    blueprint.contract.product.controller_protocol,
                    admissible_actions=("call", "put", "continue"),
                ),
            ),
        ),
    )

    decision = evaluate_route_admissibility(route, semantic_blueprint=blueprint)

    assert not decision.ok
    assert "ineligible_for_lattice_algebra:multi_controller" in decision.failures
