from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _market_state() -> MarketState:
    domestic = YieldCurve.flat(0.05)
    foreign = YieldCurve.flat(0.03)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=domestic,
        forecast_curves={"EUR-DISC": foreign},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"quanto_correlation": 0.35},
    )


def test_resolved_input_payoff_routes_expired_and_live_cases():
    from trellis.core.payoff import ResolvedInputPayoff

    @dataclass(frozen=True)
    class DummySpec:
        value: float

    @dataclass(frozen=True)
    class DummyResolved:
        T: float
        value: float

    class DummyPayoff(ResolvedInputPayoff[DummySpec, DummyResolved]):
        @property
        def requirements(self) -> set[str]:
            return {"discount_curve"}

        def resolve_inputs(self, market_state: MarketState) -> DummyResolved:
            return DummyResolved(T=float(self.spec.value), value=2.0)

        def evaluate_from_resolved(self, resolved: DummyResolved) -> float:
            return resolved.value + 1.0

        def evaluate_at_expiry(self, resolved: DummyResolved) -> float:
            return resolved.value - 1.0

    live = DummyPayoff(DummySpec(value=1.0))
    expired = DummyPayoff(DummySpec(value=0.0))

    assert live.evaluate(_market_state()) == pytest.approx(3.0)
    assert live.evaluate_raw(DummyResolved(T=1.0, value=2.0)) == pytest.approx(3.0)
    assert expired.evaluate(_market_state()) == pytest.approx(1.0)


def test_monte_carlo_path_payoff_normalizes_paths_and_discounts_mean():
    from trellis.core.payoff import MonteCarloPathPayoff

    @dataclass(frozen=True)
    class DummySpec:
        notional: float = 10.0
        n_paths: int = 2
        n_steps: int = 2

    @dataclass(frozen=True)
    class DummyResolved:
        T: float
        domestic_df: float

    class FakeEngine:
        def __init__(self):
            self.calls: list[tuple[float, float]] = []

        def simulate(self, x0, T: float):
            self.calls.append((float(x0), float(T)))
            return [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]

    fake_engine = FakeEngine()

    class DummyMonteCarloPayoff(MonteCarloPathPayoff[DummySpec, DummyResolved]):
        @property
        def requirements(self) -> set[str]:
            return {"discount_curve"}

        def resolve_inputs(self, market_state: MarketState) -> DummyResolved:
            return DummyResolved(T=1.5, domestic_df=0.9)

        def build_process(self, resolved: DummyResolved):
            return object()

        def build_initial_state(self, resolved: DummyResolved):
            return 7.0

        def build_engine(self, process, resolved: DummyResolved):
            return fake_engine

        def pathwise_payoff(self, paths, resolved: DummyResolved):
            assert paths.shape == (2, 3, 1)
            return paths[:, -1, 0]

    payoff = DummyMonteCarloPayoff(DummySpec())

    result = payoff.evaluate(_market_state())

    assert fake_engine.calls == [(7.0, 1.5)]
    assert result == pytest.approx(10.0 * 0.9 * 4.5)


def test_quanto_adapters_still_price_consistently_after_scaffold_refactor():
    from trellis.instruments._agent.quantooptionanalytical import (
        QuantoOptionAnalyticalPayoff,
        QuantoOptionSpec as AnalyticalSpec,
    )
    from trellis.instruments._agent.quantooptionmontecarlo import (
        QuantoOptionMonteCarloPayoff,
        QuantoOptionSpec as MonteCarloSpec,
    )

    analytical = QuantoOptionAnalyticalPayoff(
        AnalyticalSpec(
            notional=100_000,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
        )
    )
    mc = QuantoOptionMonteCarloPayoff(
        MonteCarloSpec(
            notional=100_000,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
            n_paths=12_000,
            n_steps=64,
        )
    )

    market_state = _market_state()
    analytical_pv = analytical.evaluate(market_state)
    mc_pv = mc.evaluate(market_state)

    assert analytical_pv > 0.0
    assert mc_pv > 0.0
    assert mc_pv == pytest.approx(analytical_pv, rel=0.12)


def test_quanto_analytical_helper_matches_adapter_price():
    from trellis.instruments._agent.quantooptionanalytical import (
        QuantoOptionAnalyticalPayoff,
        QuantoOptionSpec,
    )
    from trellis.models.analytical.quanto import price_quanto_option_raw
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
    )
    market_state = _market_state()
    resolved = resolve_quanto_inputs(market_state, spec)

    helper_pv = price_quanto_option_raw(spec, resolved)
    adapter_pv = QuantoOptionAnalyticalPayoff(spec).evaluate(market_state)

    assert helper_pv == pytest.approx(adapter_pv, rel=1e-12)


def test_quanto_analytical_adapter_delegates_to_shared_helper():
    from trellis.instruments._agent.quantooptionanalytical import (
        QuantoOptionAnalyticalPayoff,
        QuantoOptionSpec,
    )
    from trellis.models.analytical.quanto import price_quanto_option_raw
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
    )
    helper_pv = price_quanto_option_raw(spec, resolve_quanto_inputs(_market_state(), spec))

    payoff = QuantoOptionAnalyticalPayoff(spec)

    assert payoff.evaluate(_market_state()) == pytest.approx(helper_pv)


def test_quanto_monte_carlo_helper_matches_adapter_price():
    from trellis.instruments._agent.quantooptionmontecarlo import (
        QuantoOptionMonteCarloPayoff,
        QuantoOptionSpec,
    )
    from trellis.models.monte_carlo.quanto import price_quanto_option_monte_carlo
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        n_paths=12_000,
        n_steps=64,
        seed=7,
    )
    market_state = _market_state()
    resolved = resolve_quanto_inputs(market_state, spec)

    helper_pv = price_quanto_option_monte_carlo(spec, resolved)
    adapter_pv = QuantoOptionMonteCarloPayoff(spec).evaluate(market_state)

    assert helper_pv == pytest.approx(adapter_pv, rel=1e-12)


def test_quanto_monte_carlo_helpers_build_joint_process_and_initial_state():
    from trellis.instruments._agent.quantooptionmontecarlo import QuantoOptionSpec
    from trellis.models.monte_carlo.quanto import (
        build_quanto_mc_initial_state,
        build_quanto_mc_process,
        recommended_quanto_mc_engine_kwargs,
        terminal_quanto_option_payoff,
    )
    from trellis.models.resolution.quanto import resolve_quanto_inputs

    spec = QuantoOptionSpec(
        notional=100_000,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        n_paths=12_000,
        n_steps=16,
        seed=123,
    )
    resolved = resolve_quanto_inputs(_market_state(), spec)

    process = build_quanto_mc_process(resolved)
    initial_state = build_quanto_mc_initial_state(resolved)
    engine_kwargs = recommended_quanto_mc_engine_kwargs(spec, resolved)
    payoff = terminal_quanto_option_payoff(
        spec,
        [
            [[100.0, 1.10], [102.0, 1.09]],
            [[100.0, 1.10], [98.0, 1.12]],
        ],
    )

    expected_domestic_rate = -math.log(resolved.domestic_df) / resolved.T
    expected_foreign_rate = -math.log(resolved.foreign_df) / resolved.T
    expected_underlier_mu = (
        expected_domestic_rate
        - expected_foreign_rate
        - resolved.corr * resolved.sigma_underlier * resolved.sigma_fx
    )

    assert process.state_dim == 2
    assert process.mu[0] == pytest.approx(expected_underlier_mu)
    assert process.mu[1] == pytest.approx(expected_domestic_rate)
    assert initial_state.shape == (2,)
    assert initial_state[0] == pytest.approx(resolved.spot)
    assert initial_state[1] == pytest.approx(resolved.fx_spot)
    assert engine_kwargs["method"] == "exact"
    assert engine_kwargs["seed"] == 123
    assert engine_kwargs["n_steps"] >= 16
    assert payoff == pytest.approx([2.0, 0.0])
