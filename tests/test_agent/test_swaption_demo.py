"""Integration test for the swaption demo (two-step structured pipeline).

Run with: pytest tests/test_agent/test_swaption_demo.py -m integration -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from trellis.agent.builder import TRELLIS_ROOT
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.swap import SwapSpec, par_swap_rate
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)
SWAP_END = date(2030, 11, 15)
NOTIONAL = 1_000_000
STRIKE = 0.05
VOL = 0.20
RATE = 0.05

AGENT_DIR = TRELLIS_ROOT / "instruments" / "_agent"
SWAPTION_FILE = AGENT_DIR / "swaption.py"
SWAPTION_MODULE = "trellis.instruments._agent.swaption"


def _has_api_key():
    import os
    from trellis.agent.config import load_env
    load_env()
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )


def _ms(vol: float = VOL, rate: float = RATE) -> MarketState:
    return MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
    )


def _reference_swaption_price(
    rate: float = RATE, vol: float = VOL,
    strike: float = STRIKE, is_payer: bool = True,
) -> float:
    """Compute swaption price from trusted primitives."""
    from trellis.core.date_utils import generate_schedule, year_fraction

    ms = _ms(vol=vol, rate=rate)
    swap_spec = SwapSpec(
        notional=NOTIONAL, fixed_rate=0.0,
        start_date=EXPIRY, end_date=SWAP_END,
        fixed_frequency=Frequency.SEMI_ANNUAL,
        float_frequency=Frequency.SEMI_ANNUAL,
        fixed_day_count=DayCountConvention.ACT_360,
        float_day_count=DayCountConvention.ACT_360,
    )
    fwd_rate = par_swap_rate(swap_spec, ms)

    schedule = generate_schedule(EXPIRY, SWAP_END, Frequency.SEMI_ANNUAL)
    starts = [EXPIRY] + schedule[:-1]
    annuity = 0.0
    for p_start, p_end in zip(starts, schedule):
        tau = year_fraction(p_start, p_end, DayCountConvention.ACT_360)
        t_end = year_fraction(SETTLE, p_end, DayCountConvention.ACT_360)
        df = float(ms.discount.discount(t_end))
        annuity += tau * df

    T = year_fraction(SETTLE, EXPIRY, DayCountConvention.ACT_360)
    bv = black76_call(fwd_rate, strike, vol, T) if is_payer else black76_put(fwd_rate, strike, vol, T)
    return NOTIONAL * annuity * bv


@pytest.fixture
def clean_swaption():
    """Remove agent-generated swaption to force a fresh build."""
    if SWAPTION_MODULE in sys.modules:
        del sys.modules[SWAPTION_MODULE]
    if SWAPTION_FILE.exists():
        SWAPTION_FILE.unlink()
    cache_dir = AGENT_DIR / "__pycache__"
    if cache_dir.exists():
        for f in cache_dir.glob("swaption*"):
            f.unlink()
    yield


# ---------------------------------------------------------------------------
# Reference price tests (no LLM needed)
# ---------------------------------------------------------------------------

class TestReferencePriceComputation:

    def test_reference_positive(self):
        assert _reference_swaption_price() > 0

    def test_reference_less_than_notional(self):
        assert _reference_swaption_price() < NOTIONAL

    def test_reference_vol_monotonic(self):
        p1 = _reference_swaption_price(vol=0.10)
        p2 = _reference_swaption_price(vol=0.20)
        p3 = _reference_swaption_price(vol=0.40)
        assert p1 < p2 < p3

    def test_reference_payer_receiver_parity(self):
        payer = _reference_swaption_price(is_payer=True)
        receiver = _reference_swaption_price(is_payer=False)
        assert abs(payer - receiver) < NOTIONAL * 0.01

    def test_reference_zero_vol(self):
        price = _reference_swaption_price(vol=1e-10)
        assert price >= -1.0
        assert price < NOTIONAL * 0.01


# ---------------------------------------------------------------------------
# Integration tests (require LLM API key)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBuildSwaptionFromScratch:

    def test_build_and_verify(self, clean_swaption):
        if not _has_api_key():
            pytest.skip("No LLM API key set")

        from trellis.agent.executor import build_payoff

        PayoffCls = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=True,
        )

        # Structural checks
        assert PayoffCls is not None
        assert PayoffCls.__name__ == "SwaptionPayoff"  # deterministic!
        assert hasattr(PayoffCls, "evaluate")
        assert hasattr(PayoffCls, "requirements")
        assert SWAPTION_FILE.exists()

        # Instantiate with deterministic spec field names
        import importlib
        mod = importlib.import_module(SWAPTION_MODULE)
        SwaptionSpec = mod.SwaptionSpec  # deterministic name!

        spec = SwaptionSpec(
            notional=NOTIONAL,
            strike=STRIKE,
            expiry_date=EXPIRY,   # deterministic field name
            swap_start=EXPIRY,    # deterministic
            swap_end=SWAP_END,    # deterministic
        )
        payoff = PayoffCls(spec)

        assert "discount_curve" in payoff.requirements
        assert "forward_curve" in payoff.requirements

        # Price
        pv = price_payoff(payoff, _ms())
        assert isinstance(pv, (float, np.floating))
        assert pv > 0
        assert pv < NOTIONAL

        # Reference comparison (within 50% for different convention choices)
        ref = _reference_swaption_price()
        ratio = pv / ref if ref > 0 else float("inf")
        assert 0.5 < ratio < 2.0, f"Agent {pv:.2f} vs ref {ref:.2f} (ratio {ratio:.2f})"

    def test_vol_monotonicity(self, clean_swaption):
        if not _has_api_key():
            pytest.skip("No LLM API key set")

        from trellis.agent.executor import build_payoff
        from trellis.agent.invariants import check_vol_monotonicity

        PayoffCls = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
        )

        import importlib
        mod = importlib.import_module(SWAPTION_MODULE)
        spec = mod.SwaptionSpec(
            notional=NOTIONAL, strike=STRIKE,
            expiry_date=EXPIRY, swap_start=EXPIRY, swap_end=SWAP_END,
        )

        failures = check_vol_monotonicity(lambda: PayoffCls(spec), _ms)
        assert failures == [], f"Vol monotonicity failures: {failures}"


@pytest.mark.integration
class TestReuseSwaption:

    def test_reuse_existing(self):
        if not _has_api_key():
            pytest.skip("No LLM API key set")
        if not SWAPTION_FILE.exists():
            pytest.skip("No prior build")

        from trellis.agent.executor import build_payoff

        PayoffCls = build_payoff(
            "European payer swaption",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=False,
        )

        assert PayoffCls is not None
        import importlib
        mod = importlib.import_module(SWAPTION_MODULE)
        spec = mod.SwaptionSpec(
            notional=NOTIONAL, strike=STRIKE,
            expiry_date=EXPIRY, swap_start=EXPIRY, swap_end=SWAP_END,
        )

        pv = price_payoff(PayoffCls(spec), _ms())
        assert pv > 0
        assert pv < NOTIONAL
