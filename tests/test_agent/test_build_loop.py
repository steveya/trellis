"""Tests for the agent build loop with mocked LLM responses."""

import sys
from datetime import date
from unittest.mock import patch

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)

# A known-good complete module matching the static SwaptionSpec schema.
MOCK_MODULE_CODE = '''\
"""Agent-generated payoff: European payer swaption."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put


@dataclass(frozen=True)
class SwaptionSpec:
    """Specification for European payer swaption."""
    notional: float
    strike: float
    expiry_date: date
    swap_start: date
    swap_end: date
    swap_frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_360
    rate_index: str | None = None
    is_payer: bool = True


class SwaptionPayoff:
    """European payer swaption."""

    def __init__(self, spec: SwaptionSpec):
        self._spec = spec

    @property
    def spec(self) -> SwaptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol", "discount", "forward_rate"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        fwd_curve = market_state.forecast_forward_curve(spec.rate_index)

        schedule = generate_schedule(spec.swap_start, spec.swap_end, spec.swap_frequency)
        starts = [spec.swap_start] + schedule[:-1]

        annuity = 0.0
        float_pv = 0.0
        for p_start, p_end in zip(starts, schedule):
            tau = year_fraction(p_start, p_end, spec.day_count)
            t_start = year_fraction(market_state.settlement, p_start, spec.day_count)
            t_end = year_fraction(market_state.settlement, p_end, spec.day_count)
            t_start = max(t_start, 1e-6)

            df = market_state.discount.discount(t_end)
            F = fwd_curve.forward_rate(t_start, t_end)
            annuity += tau * float(df)
            float_pv += float(F) * tau * float(df)

        forward_swap_rate = float_pv / annuity if annuity > 0 else 0.0

        T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
        sigma = market_state.vol_surface.black_vol(T, spec.strike)

        if spec.is_payer:
            black_value = black76_call(forward_swap_rate, spec.strike, sigma, T)
        else:
            black_value = black76_put(forward_swap_rate, spec.strike, sigma, T)

        return spec.notional * annuity * float(black_value)
'''


class TestBuildLoop:

    @patch("trellis.agent.executor._generate_module")
    def test_build_payoff_with_mock(self, mock_gen_mod):
        """Full build loop with mocked LLM returning known-good module."""
        mock_gen_mod.return_value = MOCK_MODULE_CODE

        from trellis.agent.executor import build_payoff

        cls = build_payoff(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
            force_rebuild=True,
        )

        assert cls.__name__ == "SwaptionPayoff"
        assert hasattr(cls, "requirements")
        assert hasattr(cls, "evaluate")

    @patch("trellis.agent.executor._generate_module")
    def test_built_swaption_prices_correctly(self, mock_gen_mod):
        """The mock-built swaption produces a valid positive price."""
        mock_gen_mod.return_value = MOCK_MODULE_CODE

        from trellis.agent.executor import build_payoff

        SwaptionPayoff = build_payoff(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
            force_rebuild=True,
        )

        # Spec is deterministic — we know the exact field names
        mod = sys.modules["trellis.instruments._agent.swaption"]
        SwaptionSpec = mod.SwaptionSpec

        spec = SwaptionSpec(
            notional=1_000_000,
            strike=0.05,
            expiry_date=date(2025, 11, 15),
            swap_start=date(2025, 11, 15),
            swap_end=date(2030, 11, 15),
        )

        ms = MarketState(
            as_of=SETTLE, settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            vol_surface=FlatVol(0.20),
        )

        pv = price_payoff(SwaptionPayoff(spec), ms)
        assert pv > 0
        assert pv < 1_000_000

    @patch("trellis.agent.executor._generate_module")
    def test_built_swaption_passes_invariants(self, mock_gen_mod):
        """The mock-built swaption passes the invariant suite."""
        mock_gen_mod.return_value = MOCK_MODULE_CODE

        from trellis.agent.executor import build_payoff
        from trellis.agent.invariants import run_invariant_suite

        SwaptionPayoff = build_payoff(
            "European payer swaption",
            {"discount", "forward_rate", "black_vol"},
            force_rebuild=True,
        )

        mod = sys.modules["trellis.instruments._agent.swaption"]
        SwaptionSpec = mod.SwaptionSpec

        spec = SwaptionSpec(
            notional=1_000_000, strike=0.05,
            expiry_date=date(2025, 11, 15),
            swap_start=date(2025, 11, 15),
            swap_end=date(2030, 11, 15),
        )

        def payoff_factory():
            return SwaptionPayoff(spec)

        def ms_factory(vol=0.20):
            return MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=YieldCurve.flat(0.05),
                vol_surface=FlatVol(vol),
            )

        passed, failures = run_invariant_suite(
            payoff_factory=payoff_factory,
            market_state_factory=ms_factory,
            is_option=True,
        )
        assert passed, f"Invariant failures: {failures}"
