from __future__ import annotations

from dataclasses import fields
from math import exp

import pytest


def test_t102_authored_terminal_basket_matches_seeded_monte_carlo_proof():
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import benchmark_spec_overrides, build_market_state_for_task
    from trellis.core.differentiable import get_numpy
    from trellis.instruments._agent.basketoption import BasketOptionSpec
    from trellis.models.analytical.support import implied_zero_rate
    from trellis.models.analytical.terminal_basket import (
        two_asset_extremum_option_stulz,
        two_asset_terminal_basket_gauss_hermite,
    )
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.payoffs import terminal_basket_option_payoff
    from trellis.models.processes.correlated_gbm import CorrelatedGBM
    from trellis.models.resolution.terminal_basket import resolve_terminal_basket_inputs

    task = next(
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] == "T102"
    )
    market_state, _ = build_market_state_for_task(task)
    overrides = benchmark_spec_overrides(task)
    spec_fields = {field.name for field in fields(BasketOptionSpec)}
    spec = BasketOptionSpec(**{key: value for key, value in overrides.items() if key in spec_fields})
    resolved = resolve_terminal_basket_inputs(
        market_state,
        spec,
        comparison_target="stulz_rainbow",
    )
    semantics = resolved.semantics
    rate = implied_zero_rate(semantics.domestic_df, semantics.T)

    analytical = float(spec.notional) * two_asset_extremum_option_stulz(
        spots=resolved.notional_spots,
        strike=resolved.strike,
        T=semantics.T,
        discount_factor=semantics.domestic_df,
        dividend_yields=resolved.carry,
        volatilities=resolved.vols,
        correlation=resolved.correlation_matrix[0][1],
        basket_style=resolved.basket_style,
        option_type=resolved.option_type,
    )
    independent_reference = float(spec.notional) * two_asset_terminal_basket_gauss_hermite(
        spots=resolved.notional_spots,
        weights=resolved.weights,
        strike=resolved.strike,
        T=semantics.T,
        discount_factor=semantics.domestic_df,
        dividend_yields=resolved.carry,
        volatilities=resolved.vols,
        correlation=resolved.correlation_matrix[0][1],
        basket_style=resolved.basket_style,
        option_type=resolved.option_type,
        n_points=192,
    )

    process = CorrelatedGBM(
        mu=[rate, rate],
        sigma=list(resolved.vols),
        corr=[list(row) for row in resolved.correlation_matrix],
        dividend_yield=list(resolved.carry),
    )
    engine = MonteCarloEngine(
        process,
        n_paths=spec.n_paths,
        n_steps=spec.n_steps,
        seed=spec.seed,
        method=spec.mc_method,
    )

    def payoff(paths):
        return terminal_basket_option_payoff(
            paths[:, -1, :],
            weights=resolved.weights,
            basket_style=resolved.basket_style,
            strike=resolved.strike,
            option_type=resolved.option_type,
        )

    sampled_result = engine.price(
        get_numpy().asarray(semantics.constituent_spots, dtype=float),
        semantics.T,
        payoff,
        discount_rate=rate,
        return_paths=False,
    )
    sampled = float(spec.notional) * float(sampled_result["price"])
    sampled_standard_error = float(spec.notional) * float(sampled_result["std_error"])

    assert semantics.T == pytest.approx(1.0)
    assert semantics.domestic_df == pytest.approx(exp(-0.05))
    assert analytical == pytest.approx(142.81201951370434, rel=1e-12)
    assert independent_reference == pytest.approx(142.781037347788, rel=1e-12)
    assert sampled == pytest.approx(142.37678391694178, rel=1e-12)
    assert sampled_standard_error == pytest.approx(0.8068581682950531, rel=1e-12)
    assert abs(sampled - analytical) < sampled_standard_error
    assert abs(sampled - analytical) / analytical * 100.0 < task["cross_validate"][
        "tolerance_pct"
    ]
