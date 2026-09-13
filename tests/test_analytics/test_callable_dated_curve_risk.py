"""Callable duration can retain the authored dated-market fixture."""

from dataclasses import replace
from datetime import date

import pytest

from trellis.agent.benchmark_contracts import benchmark_spec_overrides
from trellis.agent.task_manifests import load_task_manifest
from trellis.agent.task_runtime import build_market_state_for_task
from trellis.analytics.measures import Duration, OASDuration
from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve
from trellis.instruments.callable_bond import CallableBondPayoff, CallableBondSpec
from trellis.models.callable_bond_tree import price_callable_bond_tree


def test_callable_durations_preserve_authored_dated_curve_and_model_parameters():
    task = next(
        task for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
        if task["id"] == "T02"
    )
    market, _ = build_market_state_for_task(task)
    spec = CallableBondSpec(**benchmark_spec_overrides(task))
    curve = market.discount
    assert isinstance(curve, DateAwareFlatYieldCurve)
    assert curve.value_date == date(2025, 1, 15)
    assert tuple(spec.call_dates) == (date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15))
    assert market.selected_curve_names["discount_curve"] == "usd_callable_flat_5pct"
    assert market.model_parameter_sets["callable_fixed_5pct_proof:hull_white"]["sigma"] == 0.01

    controls = task["cross_validate"]["target_contracts"]["hull_white_tree"]["variant_parameters"]

    def reference_price(bps):
        # Construct reference curves directly so this check does not use shift().
        shifted_curve = replace(curve, flat_rate=curve.flat_rate + bps / 10_000.0)
        shifted_market = replace(market, discount=shifted_curve, forward_curve=None)
        return price_callable_bond_tree(
            shifted_market, spec, model="hull_white",
            mean_reversion=controls["mean_reversion"], sigma=controls["sigma"],
            n_steps=controls["tree_steps"],
        )

    bump_bps = 25.0
    base_price = reference_price(0.0)
    expected = -(reference_price(bump_bps) - reference_price(-bump_bps)) / (
        2.0 * bump_bps / 10_000.0 * base_price
    )
    observed_markets = []

    class ObservedCallableBondPayoff(CallableBondPayoff):
        def evaluate(self, market_state):
            observed_markets.append(market_state)
            return super().evaluate(market_state)

    payoff = ObservedCallableBondPayoff(spec)
    duration = Duration(bump_bps=bump_bps).compute(payoff, market)
    oas_duration = OASDuration(bump_bps=bump_bps).compute(payoff, market)

    assert base_price == pytest.approx(96.8252972854, abs=1e-6)
    assert expected > 0.0
    assert float(duration) == pytest.approx(expected, rel=1e-12)
    assert oas_duration == pytest.approx(expected, rel=1e-12)
    assert duration.metadata["resolved_derivative_method"] == "parallel_curve_bump"
    shifted_markets = [observed for observed in observed_markets if observed is not market]
    assert len(shifted_markets) == 4
    assert sorted(observed.discount.flat_rate for observed in shifted_markets) == pytest.approx(
        [0.0475, 0.0475, 0.0525, 0.0525],
    )
    for observed in shifted_markets:
        assert isinstance(observed.discount, DateAwareFlatYieldCurve)
        assert observed.discount.value_date == curve.value_date
        assert observed.discount.curve_day_count is curve.curve_day_count
        assert observed.discount.max_tenor == curve.max_tenor
        assert observed.model_parameters == market.model_parameters
        assert observed.model_parameter_sets == market.model_parameter_sets
        assert observed.forecast_curves == market.forecast_curves
        assert observed.selected_curve_names == market.selected_curve_names
        assert observed.market_provenance == market.market_provenance
        assert observed.forward_curve.discount_date(spec.end_date) == pytest.approx(
            observed.discount.discount_date(spec.end_date),
        )
    assert market.discount is curve
    assert curve.flat_rate == 0.05
