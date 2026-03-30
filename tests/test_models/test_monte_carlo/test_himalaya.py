"""Tests for the Himalaya Monte Carlo route helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest


SETTLE = date(2024, 11, 15)


def _resolved_himalaya_inputs():
    from trellis.models.resolution.himalaya import ResolvedHimalayaInputs

    return ResolvedHimalayaInputs(
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


def test_terminal_himalaya_option_payoff_applies_ranked_lock_remove_logic():
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.monte_carlo.himalaya import terminal_himalaya_option_payoff

    spec = HimalayaOptionSpec(
        notional=100.0,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
    )
    resolved = _resolved_himalaya_inputs()
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

    payoff = terminal_himalaya_option_payoff(spec, paths, resolved)

    assert payoff.shape == (1,)
    assert payoff[0] == pytest.approx(0.06)


def test_price_himalaya_option_monte_carlo_uses_engine_and_discount_factor(monkeypatch):
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    import trellis.models.monte_carlo.himalaya as himalaya_mc

    resolved = _resolved_himalaya_inputs()
    spec = HimalayaOptionSpec(
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
            initial_state,
            T,
            payoff_fn,
            discount_rate=0.0,
            *,
            storage_policy="auto",
            return_paths=False,
            shocks=None,
            differentiable=False,
        ):
            created["initial_state"] = initial_state
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

    monkeypatch.setattr(
        "trellis.models.monte_carlo.ranked_observation_payoffs.MonteCarloEngine",
        FakeEngine,
    )

    price = himalaya_mc.price_himalaya_option_monte_carlo(spec, resolved)

    assert price == pytest.approx(5.4)
    assert created["kwargs"]["n_paths"] == 4096
    assert created["kwargs"]["n_steps"] == 252
    assert created["kwargs"]["seed"] == 7
    assert created["kwargs"]["method"] == "exact"
    assert created["T"] == pytest.approx(1.0)
    assert created["initial_state"].shape == (2,)
    assert created["storage_policy"] is not None
    assert created["path_requirement"].snapshot_steps == (126, 252)


def test_recommended_himalaya_mc_engine_kwargs_scales_with_expiry_horizon():
    from trellis.instruments._agent.himalayaoptionmontecarlo import HimalayaOptionSpec
    from trellis.models.monte_carlo.himalaya import recommended_himalaya_mc_engine_kwargs

    resolved = _resolved_himalaya_inputs()
    spec = HimalayaOptionSpec(
        notional=100.0,
        strike=0.05,
        expiry_date=date(2025, 11, 15),
        constituents="SPX,NDX",
        n_paths=64,
        n_steps=8,
        seed=11,
    )

    kwargs = recommended_himalaya_mc_engine_kwargs(spec, resolved)

    assert kwargs["n_paths"] == 4096
    assert kwargs["n_steps"] == 252
    assert kwargs["seed"] == 11
    assert kwargs["method"] == "exact"
