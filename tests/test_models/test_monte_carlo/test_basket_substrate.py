"""Tests for the reusable ranked-observation basket Monte Carlo substrate."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest


SETTLE = date(2024, 11, 15)


def _resolved_basket_inputs():
    from trellis.models.resolution.basket_semantics import ResolvedBasketSemantics

    return ResolvedBasketSemantics(
        constituent_names=("SPX", "NDX"),
        constituent_spots=(100.0, 100.0),
        constituent_vols=(0.20, 0.20),
        constituent_carry=(0.0, 0.0),
        correlation_matrix=((1.0, 0.35), (0.35, 1.0)),
        observation_dates=(date(2025, 2, 15), date(2025, 5, 15)),
        observation_times=(0.5, 1.0),
        valuation_date=SETTLE,
        T=1.0,
        domestic_df=0.90,
        selection_rule="best_of_remaining",
        lock_rule="remove_selected",
        aggregation_rule="average_locked_returns",
        selection_count=1,
    )


def test_ranked_observation_basket_state_replays_locked_returns_from_snapshots():
    from trellis.models.monte_carlo.basket_state import (
        evaluate_ranked_observation_basket_paths,
        observation_step_indices,
    )

    resolved = _resolved_basket_inputs()
    paths = np.array(
        [
            [
                [100.0, 100.0],
                [110.0, 105.0],
                [120.0, 112.0],
            ]
        ],
        dtype=float,
    )

    payoff = evaluate_ranked_observation_basket_paths(
        paths,
        resolved.constituent_spots,
        observation_step_indices(resolved.observation_times, resolved.T, paths.shape[1] - 1),
        selection_rule=resolved.selection_rule,
        lock_rule=resolved.lock_rule,
        aggregation_rule=resolved.aggregation_rule,
        selection_count=resolved.selection_count,
    )

    assert payoff.shape == (1,)
    assert payoff[0] == pytest.approx(0.11)


def test_ranked_observation_basket_price_helper_uses_snapshot_state_requirement(monkeypatch):
    from trellis.models.monte_carlo import ranked_observation_payoffs as basket_mc
    from trellis.models.monte_carlo.semantic_basket import RankedObservationBasketSpec

    resolved = _resolved_basket_inputs()
    spec = RankedObservationBasketSpec(
        notional=100.0,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        n_paths=123,
        n_steps=12,
        seed=7,
    )

    paths = np.array(
        [
            [
                [100.0, 100.0],
                [110.0, 105.0],
                [120.0, 112.0],
            ],
            [
                [100.0, 100.0],
                [110.0, 105.0],
                [120.0, 112.0],
            ],
        ],
        dtype=float,
    )

    created = {}

    class FakeEngine:
        def __init__(self, process, **kwargs):
            created["process"] = process
            created["kwargs"] = kwargs

        def price(
            self,
            x0,
            T,
            payoff_fn,
            discount_rate=0.0,
            *,
            storage_policy="auto",
            return_paths=False,
            shocks=None,
            differentiable=False,
        ):
            created["x0"] = x0
            created["T"] = T
            created["storage_policy"] = storage_policy
            created["path_requirement"] = getattr(payoff_fn, "path_requirement", None)
            created["return_paths"] = return_paths
            return {
                "price": 0.06,
                "std_error": 0.0,
                "st_err": 0.0,
                "n_paths": 2,
                "paths": None,
                "path_state": None,
            }

    monkeypatch.setattr(basket_mc, "MonteCarloEngine", FakeEngine)
    monkeypatch.setattr(
        basket_mc,
        "recommended_ranked_observation_basket_mc_engine_kwargs",
        lambda *_args, **_kwargs: {"n_paths": 2, "n_steps": 2, "seed": 7, "method": "exact"},
    )
    monkeypatch.setattr(
        basket_mc,
        "build_ranked_observation_basket_process",
        lambda _resolved: object(),
    )
    monkeypatch.setattr(
        basket_mc,
        "build_ranked_observation_basket_initial_state",
        lambda _resolved: np.array([100.0, 100.0], dtype=float),
    )

    price = basket_mc.price_ranked_observation_basket_monte_carlo(spec, resolved)

    assert price == pytest.approx(5.4)
    assert created["kwargs"]["n_paths"] == 2
    assert created["kwargs"]["n_steps"] == 2
    assert created["kwargs"]["seed"] == 7
    assert created["kwargs"]["method"] == "exact"
    assert created["T"] == pytest.approx(1.0)
    assert created["x0"].shape == (2,)
    assert created["storage_policy"] is not None
    assert created["path_requirement"].snapshot_steps == (1, 2)


def test_ranked_observation_basket_path_contract_matches_snapshot_requirement():
    from trellis.core.runtime_contract import ContractState, ResolvedInputs, RuntimeContext
    from trellis.models.monte_carlo.semantic_basket import (
        RankedObservationBasketSpec,
        build_ranked_observation_basket_path_contract,
    )

    resolved = _resolved_basket_inputs()
    spec = RankedObservationBasketSpec(
        notional=100.0,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        n_paths=123,
        n_steps=12,
        seed=7,
    )

    contract = build_ranked_observation_basket_path_contract(spec, resolved)

    assert contract.observation_dates == (date(2025, 2, 15), date(2025, 5, 15))
    assert contract.observation_times == (0.5, 1.0)
    assert contract.observation_steps == (126, 252)
    assert contract.snapshot_steps == (126, 252)
    assert contract.state_tags == ("pathwise_only", "remaining_pool", "locked_cashflow_state")
    assert contract.event_kinds == ("observation", "settlement")
    assert isinstance(contract.initial_state, ContractState)
    assert isinstance(contract.resolved_inputs, ResolvedInputs)
    assert isinstance(contract.runtime_context, RuntimeContext)
    assert contract.initial_state.require_memory("selection_rule") == "best_of_remaining"
    assert contract.resolved_inputs.require("constituent_spots") == (100.0, 100.0)
    assert contract.runtime_context.phase == "observation"
