"""Tests for the generic event-aware Monte Carlo assembly substrate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as raw_np
import pytest

from trellis.core.date_utils import build_payment_timeline as core_build_payment_timeline
from trellis.core.differentiable import gradient, get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.models.rate_style_swaption import price_swaption_black76, resolve_swaption_black76_inputs
from trellis.models.rate_style_swaption_tree import price_swaption_tree
from trellis.models.vol_surface import FlatVol
from trellis.models.monte_carlo.path_state import MonteCarloPathState
from trellis.models.processes.gbm import GBM
from trellis.models.processes.hull_white import HullWhite
from trellis.models.processes.local_vol import LocalVol

np = get_numpy()


class TestEventAwareMonteCarloAssembly:
    def test_event_aware_module_reexports_payment_timeline(self):
        from trellis.models.monte_carlo.event_aware import build_payment_timeline

        assert build_payment_timeline is core_build_payment_timeline

    @pytest.mark.parametrize(
        ("family", "kwargs", "process_type", "simulation_method", "initial_state"),
        [
            (
                "gbm_1d",
                {"risk_free_rate": 0.03, "sigma": 0.20},
                GBM,
                "exact",
                100.0,
            ),
            (
                "local_vol_1d",
                {
                    "risk_free_rate": 0.03,
                    "local_vol_surface": lambda spot, t: 0.15 + 0.02 * raw_np.exp(-t),
                },
                LocalVol,
                "euler",
                100.0,
            ),
            (
                "hull_white_1f",
                {"mean_reversion": 0.10, "sigma": 0.01, "theta": 0.03},
                HullWhite,
                "exact",
                0.03,
            ),
        ],
    )
    def test_build_problem_resolves_supported_process_families(
        self,
        family,
        kwargs,
        process_type,
        simulation_method,
        initial_state,
    ):
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            build_event_aware_monte_carlo_problem,
        )

        spec = EventAwareMonteCarloProblemSpec(
            process_spec=EventAwareMonteCarloProcessSpec(family=family, **kwargs),
            initial_state=initial_state,
            maturity=1.0,
            n_steps=8,
            terminal_payoff=lambda terminal: raw_np.ones_like(raw_np.asarray(terminal, dtype=float)),
        )

        problem = build_event_aware_monte_carlo_problem(spec)

        assert isinstance(problem.process, process_type)
        assert problem.simulation_method == simulation_method
        assert problem.path_requirement.snapshot_steps == ()
        assert problem.event_timeline is None

    def test_build_problem_rejects_unsupported_process_family(self):
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            build_event_aware_monte_carlo_problem,
        )

        spec = EventAwareMonteCarloProblemSpec(
            process_spec=EventAwareMonteCarloProcessSpec(family="unsupported"),
            initial_state=100.0,
            maturity=1.0,
            terminal_payoff=lambda terminal: raw_np.asarray(terminal, dtype=float),
        )

        with pytest.raises(ValueError, match="Unsupported Monte Carlo process family"):
            build_event_aware_monte_carlo_problem(spec)

    def test_event_replay_problem_uses_path_event_state_without_full_paths(self):
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloEvent,
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            build_event_aware_monte_carlo_problem,
        )

        problem = build_event_aware_monte_carlo_problem(
            EventAwareMonteCarloProblemSpec(
                process_spec=EventAwareMonteCarloProcessSpec(
                    family="gbm_1d",
                    risk_free_rate=0.03,
                    sigma=0.20,
                ),
                initial_state=100.0,
                maturity=1.0,
                n_steps=2,
                path_requirement_kind="event_replay",
                reducer_kind="compiled_schedule_payoff",
                settlement_event="callable_settlement",
                event_specs=(
                    EventAwareMonteCarloEvent(
                        time=0.5,
                        name="coupon_1",
                        kind="coupon",
                        payload={"amount": 5.0},
                    ),
                    EventAwareMonteCarloEvent(
                        time=1.0,
                        name="callable_settlement",
                        kind="settlement",
                        payload={"rule": "terminal_value", "coupon_events": ("coupon_1",)},
                    ),
                ),
            )
        )

        paths = raw_np.array(
            [
                [100.0, 101.0, 103.0],
                [100.0, 99.0, 96.0],
            ],
            dtype=float,
        )
        expected = raw_np.array([108.0, 101.0], dtype=float)

        raw_np.testing.assert_allclose(problem.payoff(paths), expected)
        assert problem.path_requirement.full_path is False
        assert problem.path_requirement.snapshot_steps == (1, 2)
        assert problem.event_timeline is not None
        assert tuple(event.name for event in problem.event_timeline) == ("coupon_1", "callable_settlement")

        state = MonteCarloPathState(
            initial_value=100.0,
            n_steps=2,
            terminal_values=paths[:, -1],
            snapshots={1: paths[:, 1]},
        )
        raw_np.testing.assert_allclose(problem.payoff.evaluate_state(state), expected)

    def test_price_event_aware_problem_uses_reduced_state_storage(self):
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloEvent,
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            build_event_aware_monte_carlo_problem,
            price_event_aware_monte_carlo,
        )

        problem = build_event_aware_monte_carlo_problem(
            EventAwareMonteCarloProblemSpec(
                process_spec=EventAwareMonteCarloProcessSpec(
                    family="gbm_1d",
                    risk_free_rate=0.03,
                    sigma=0.20,
                ),
                initial_state=100.0,
                maturity=1.0,
                n_steps=4,
                path_requirement_kind="event_replay",
                reducer_kind="compiled_schedule_payoff",
                settlement_event="terminal_settlement",
                event_specs=(
                    EventAwareMonteCarloEvent(
                        time=1.0,
                        name="terminal_settlement",
                        kind="settlement",
                        payload={"rule": "terminal_value"},
                    ),
                ),
            )
        )

        result = price_event_aware_monte_carlo(
            problem,
            n_paths=64,
            seed=7,
            return_paths=False,
        )

        assert result["paths"] is None
        assert result["path_state"] is not None
        assert result["price"] > 0.0

    def test_price_event_aware_monte_carlo_accepts_problem_spec(self):
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            price_event_aware_monte_carlo,
        )

        result = price_event_aware_monte_carlo(
            EventAwareMonteCarloProblemSpec(
                process_spec=EventAwareMonteCarloProcessSpec(
                    family="gbm_1d",
                    risk_free_rate=0.03,
                    sigma=0.20,
                ),
                initial_state=100.0,
                maturity=1.0,
                n_steps=8,
                terminal_payoff=lambda terminal: raw_np.maximum(
                    raw_np.asarray(terminal, dtype=float) - 100.0,
                    0.0,
                ),
            ),
            n_paths=128,
            seed=7,
            return_paths=False,
        )

        assert result["paths"] is None
        assert result["price"] > 0.0
        assert result["n_paths"] == 128

    def test_price_event_aware_monte_carlo_accepts_runtime_process_bundle(self):
        from trellis.models.monte_carlo.event_aware import price_event_aware_monte_carlo

        result = price_event_aware_monte_carlo(
            process=GBM(mu=0.03, sigma=0.20),
            initial_state=100.0,
            maturity=1.0,
            n_steps=8,
            discount_rate=0.03,
            terminal_payoff=lambda terminal: raw_np.maximum(
                raw_np.asarray(terminal, dtype=float) - 100.0,
                0.0,
            ),
            n_paths=128,
            seed=7,
            return_paths=False,
        )

        assert result["paths"] is None
        assert result["price"] > 0.0

    def test_price_event_aware_monte_carlo_supports_differentiable_event_replay(self):
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloEvent,
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            price_event_aware_monte_carlo,
        )

        shocks = raw_np.random.default_rng(97).standard_normal((256, 8))

        def price_from_spot(spot):
            result = price_event_aware_monte_carlo(
                EventAwareMonteCarloProblemSpec(
                    process_spec=EventAwareMonteCarloProcessSpec(
                        family="gbm_1d",
                        risk_free_rate=0.03,
                        sigma=0.20,
                    ),
                    initial_state=spot,
                    maturity=1.0,
                    n_steps=8,
                    discount_rate=0.03,
                    path_requirement_kind="event_replay",
                    reducer_kind="compiled_schedule_payoff",
                    settlement_event="terminal_settlement",
                    event_specs=(
                        EventAwareMonteCarloEvent(
                            time=0.5,
                            name="coupon_1",
                            kind="coupon",
                            payload={"amount": 2.0},
                        ),
                        EventAwareMonteCarloEvent(
                            time=1.0,
                            name="terminal_settlement",
                            kind="settlement",
                            payload={
                                "rule": "terminal_value",
                                "coupon_events": ("coupon_1",),
                            },
                        ),
                    ),
                ),
                n_paths=256,
                seed=11,
                return_paths=False,
                shocks=shocks,
                differentiable=True,
            )
            assert result["paths"] is None
            assert result["path_state"] is not None
            assert tuple(sorted(result["path_state"].snapshots)) == (4,)
            return result["price"]

        autodiff_delta = gradient(price_from_spot)(100.0)
        eps = 1e-4
        fd_delta = (price_from_spot(100.0 + eps) - price_from_spot(100.0 - eps)) / (2 * eps)
        assert autodiff_delta == pytest.approx(fd_delta, rel=1e-5, abs=1e-5)
    def test_hull_white_swaption_problem_prices_from_reduced_event_state(self):
        from trellis.core.date_utils import build_payment_timeline
        from trellis.models.monte_carlo.event_aware import (
            EventAwareMonteCarloEvent,
            EventAwareMonteCarloProblemSpec,
            EventAwareMonteCarloProcessSpec,
            build_discounted_swap_pv_payload,
            build_event_aware_monte_carlo_problem,
            build_short_rate_discount_reducer,
            price_event_aware_monte_carlo,
        )

        settle = date(2024, 11, 15)

        @dataclass(frozen=True)
        class _SwaptionSpec:
            notional: float = 1_000_000.0
            strike: float = 0.045
            expiry_date: date = date(2025, 11, 15)
            swap_start: date = expiry_date
            swap_end: date = date(2030, 11, 15)
            swap_frequency: Frequency = Frequency.SEMI_ANNUAL
            day_count: DayCountConvention = DayCountConvention.ACT_360
            rate_index: str | None = "USD-SOFR-3M"
            is_payer: bool = True

        spec = _SwaptionSpec()
        market_state = MarketState(
            as_of=settle,
            settlement=settle,
            discount=YieldCurve.flat(0.042, max_tenor=10.0),
            forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.046, max_tenor=10.0)},
            vol_surface=FlatVol(0.20),
            selected_curve_names={"discount_curve": "usd_ois", "forecast_curve": "USD-SOFR-3M"},
        )

        resolved = resolve_swaption_black76_inputs(market_state, spec)
        payment_timeline = build_payment_timeline(
            spec.swap_start,
            spec.swap_end,
            spec.swap_frequency,
            day_count=spec.day_count,
            time_origin=market_state.settlement,
            label="swaption_mc_event_timeline",
        )
        surviving_periods = tuple(
            period
            for period in payment_timeline
            if period.end_date > market_state.settlement
        )
        expiry_years = float(resolved.expiry_years)
        r0 = float(market_state.discount.zero_rate(max(expiry_years / 2.0, 1e-6)))
        mean_reversion = 0.10
        sigma = float(resolved.vol) * max(abs(r0), 1e-6)
        settlement_payload = build_discounted_swap_pv_payload(
            payment_timeline=surviving_periods,
            discount_curve=market_state.discount,
            forward_curve=market_state.forecast_forward_curve(spec.rate_index),
            exercise_time=expiry_years,
            discount_reducer_name="discount_to_expiry",
            mean_reversion=mean_reversion,
            strike=spec.strike,
            notional=spec.notional,
            is_payer=spec.is_payer,
        )

        problem = build_event_aware_monte_carlo_problem(
            EventAwareMonteCarloProblemSpec(
                process_spec=EventAwareMonteCarloProcessSpec(
                    family="hull_white_1f",
                    mean_reversion=mean_reversion,
                    sigma=sigma,
                    theta=mean_reversion * r0,
                ),
                initial_state=r0,
                maturity=expiry_years,
                n_steps=64,
                path_requirement_kind="event_replay",
                reducer_kind="compiled_schedule_payoff",
                path_reducers=(
                    build_short_rate_discount_reducer(
                        name="discount_to_expiry",
                        maturity=expiry_years,
                    ),
                ),
                settlement_event="swaption_settlement",
                event_specs=(
                    EventAwareMonteCarloEvent(
                        time=expiry_years,
                        name="swaption_observation",
                        kind="observation",
                    ),
                    EventAwareMonteCarloEvent(
                        time=expiry_years,
                        name="swaption_settlement",
                        kind="settlement",
                        priority=1,
                        payload=settlement_payload,
                    ),
                ),
            )
        )

        result = price_event_aware_monte_carlo(
            problem,
            n_paths=12_000,
            seed=11,
            return_paths=False,
        )

        black76_price = price_swaption_black76(market_state, spec)
        tree_price = price_swaption_tree(
            market_state,
            spec,
            model="hull_white",
            mean_reversion=mean_reversion,
            sigma=sigma,
        )

        assert result["paths"] is None
        assert result["path_state"] is not None
        assert "discount_to_expiry" in result["path_state"].reducer_values
        assert result["price"] > 0.0
        assert result["price"] == pytest.approx(tree_price, rel=0.12)
        assert result["price"] == pytest.approx(black76_price, rel=0.35)

    def test_build_discounted_swap_pv_payload_preserves_curve_basis_and_schedule(self):
        from trellis.core.date_utils import build_payment_timeline
        from trellis.models.monte_carlo.event_aware import build_discounted_swap_pv_payload

        settle = date(2024, 11, 15)
        discount_curve = YieldCurve.flat(0.042, max_tenor=10.0)
        forward_curve = YieldCurve.flat(0.046, max_tenor=10.0)
        payment_timeline = build_payment_timeline(
            date(2025, 11, 15),
            date(2030, 11, 15),
            Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_360,
            time_origin=settle,
            label="payload_test_timeline",
        )

        payload = build_discounted_swap_pv_payload(
            payment_timeline=payment_timeline,
            discount_curve=discount_curve,
            forward_curve=forward_curve,
            exercise_time=1.0,
            discount_reducer_name="discount_to_expiry",
            mean_reversion=0.10,
            strike=0.045,
            notional=1_000_000.0,
            is_payer=True,
        )

        assert payload["rule"] == "discounted_swap_pv"
        assert payload["discount_reducer_name"] == "discount_to_expiry"
        assert len(payload["payment_times"]) == len(payment_timeline)
        assert len(payload["accrual_fractions"]) == len(payment_timeline)
        assert len(payload["anchor_discount_factors"]) == len(payment_timeline)
        assert payload["curve_basis_spread"] > 0.0

    def test_resolve_hull_white_monte_carlo_process_inputs_reads_market_surface(self):
        from trellis.models.monte_carlo.event_aware import resolve_hull_white_monte_carlo_process_inputs

        settle = date(2024, 11, 15)
        market_state = MarketState(
            as_of=settle,
            settlement=settle,
            discount=YieldCurve.flat(0.042, max_tenor=10.0),
            vol_surface=FlatVol(0.20),
            model_parameters={
                "model_family": "hull_white",
                "mean_reversion": 0.05,
            },
        )

        process_spec, initial_state = resolve_hull_white_monte_carlo_process_inputs(
            market_state,
            option_horizon=1.0,
            strike=0.045,
        )

        expected_r0 = float(market_state.discount.zero_rate(0.5))
        expected_sigma = 0.20 * max(abs(expected_r0), 1e-6)

        assert process_spec.family == "hull_white_1f"
        assert process_spec.mean_reversion == pytest.approx(0.05)
        assert process_spec.sigma == pytest.approx(expected_sigma)
        assert process_spec.theta == pytest.approx(0.05 * expected_r0)
        assert initial_state == pytest.approx(expected_r0)
