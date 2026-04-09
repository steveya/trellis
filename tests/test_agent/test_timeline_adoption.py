"""Regression tests for QUA-477 timeline-builder adoption."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.agentcap import AgentCapPayoff, AgentCapSpec
from trellis.instruments._agent.swaption import SwaptionPayoff, SwaptionSpec
from trellis.instruments.cap import CapFloorSpec, CapPayoff
from trellis.models.calibration.rates import swaption_terms
from trellis.models.rate_style_swaption import price_swaption_monte_carlo
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _market_state(rate: float = 0.05, vol: float = 0.20) -> MarketState:
    curve = YieldCurve.flat(rate)
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=curve,
        vol_surface=FlatVol(vol),
    )


def test_agent_cap_matches_reference_cap_payoff_after_timeline_migration():
    spec = AgentCapSpec(
        notional=1_000_000,
        strike=0.05,
        start_date=date(2025, 2, 15),
        end_date=date(2027, 2, 15),
        frequency=Frequency.QUARTERLY,
        day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
    )
    market_state = _market_state()

    agent_price = AgentCapPayoff(spec).evaluate(market_state)
    reference_price = CapPayoff(
        CapFloorSpec(
            notional=spec.notional,
            strike=spec.strike,
            start_date=spec.start_date,
            end_date=spec.end_date,
            frequency=spec.frequency,
            day_count=spec.day_count,
            rate_index=spec.rate_index,
        )
    ).evaluate(market_state)

    assert agent_price == pytest.approx(reference_price)


def test_agent_swaption_matches_swaption_terms_formula_after_timeline_migration():
    spec = SwaptionSpec(
        notional=2_000_000,
        strike=0.045,
        expiry_date=date(2025, 11, 15),
        swap_start=date(2025, 11, 15),
        swap_end=date(2030, 11, 15),
        swap_frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_360,
        rate_index="USD-SOFR-3M",
        is_payer=True,
    )
    market_state = _market_state(rate=0.04, vol=0.18)

    price = SwaptionPayoff(spec).evaluate(market_state)
    _, _, _, payment_count = swaption_terms(spec, market_state)
    expected = price_swaption_monte_carlo(
        market_state,
        spec,
        n_paths=spec.mc_n_paths,
        seed=spec.mc_seed,
    )

    assert payment_count > 0
    assert price == pytest.approx(expected)


def test_generated_schedule_heavy_routes_no_longer_reconstruct_period_starts_manually():
    targets = {
        REPO_ROOT / "trellis" / "instruments" / "_agent" / "agentcap.py": (
            "build_payment_timeline",
        ),
        REPO_ROOT / "trellis" / "instruments" / "_agent" / "agentfloor.py": (
            "build_payment_timeline",
        ),
        REPO_ROOT / "trellis" / "instruments" / "_agent" / "swaption.py": (
            "price_swaption_monte_carlo",
        ),
    }

    for path, required_symbols in targets.items():
        text = path.read_text()
        assert "schedule[:-1]" not in text
        assert "[spec.start_date] +" not in text
        for required_symbol in required_symbols:
            assert required_symbol in text


def test_black76_forward_rate_guidance_prefers_timeline_builders():
    cookbook_text = (
        REPO_ROOT / "trellis" / "agent" / "knowledge" / "canonical" / "cookbooks.yaml"
    ).read_text()
    prompts_text = (REPO_ROOT / "trellis" / "agent" / "prompts.py").read_text()

    assert "build_payment_timeline" in cookbook_text
    assert "starts = [spec.start_date] + schedule[:-1]" not in cookbook_text
    assert "prefer `build_payment_timeline" in prompts_text
