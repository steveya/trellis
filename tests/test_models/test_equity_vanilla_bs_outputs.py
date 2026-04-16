"""Tests for the Black-Scholes equity vanilla native outputs helper (QUA-862)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical import equity_vanilla_bs_outputs


class _FlatDiscount:
    def __init__(self, rate: float):
        self._rate = float(rate)

    def zero_rate(self, t: float) -> float:
        return self._rate

    def discount(self, t: float) -> float:
        return float(np.exp(-self._rate * float(t)))


class _FlatBlackVol:
    def __init__(self, vol: float):
        self._vol = float(vol)

    def black_vol(self, t: float, k: float) -> float:
        return self._vol


@dataclass(frozen=True)
class _VanillaSpec:
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _make_market(rate: float, vol: float) -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=_FlatDiscount(rate),
        vol_surface=_FlatBlackVol(vol),
    )


def test_atm_call_matches_closed_form_and_returns_expected_keys():
    ms = _make_market(rate=0.05, vol=0.25)
    spec = _VanillaSpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )
    out = equity_vanilla_bs_outputs(ms, spec)
    assert set(out) == {"price", "delta", "gamma", "vega", "theta"}
    # ATM call under q=0, r=0.05, σ=0.25, T=1.0 → classic BS reference values.
    assert out["price"] == pytest.approx(12.336, abs=1e-3)
    assert out["delta"] == pytest.approx(0.6274, abs=1e-3)
    assert out["gamma"] == pytest.approx(0.01514, abs=1e-4)
    assert out["vega"] == pytest.approx(37.842, abs=1e-2)
    assert out["theta"] == pytest.approx(-7.250, abs=1e-2)


def test_put_call_parity_holds_on_delta_and_price():
    ms = _make_market(rate=0.05, vol=0.25)
    call = equity_vanilla_bs_outputs(
        ms,
        _VanillaSpec(
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            option_type="call",
        ),
    )
    put = equity_vanilla_bs_outputs(
        ms,
        _VanillaSpec(
            notional=1.0,
            spot=100.0,
            strike=100.0,
            expiry_date=date(2025, 11, 15),
            option_type="put",
        ),
    )
    # Put-call parity: C - P = S - K*df
    df = float(np.exp(-0.05 * 1.0))
    assert (call["price"] - put["price"]) == pytest.approx(100.0 - 100.0 * df, abs=1e-6)
    # Delta parity: call_delta - put_delta = 1 (under q=0)
    assert (call["delta"] - put["delta"]) == pytest.approx(1.0, abs=1e-8)
    # Gamma/vega identical for call and put
    assert call["gamma"] == pytest.approx(put["gamma"], abs=1e-12)
    assert call["vega"] == pytest.approx(put["vega"], abs=1e-12)


def test_notional_scales_price_only():
    ms = _make_market(rate=0.05, vol=0.25)
    expiry = date(2025, 11, 15)
    unit = equity_vanilla_bs_outputs(
        ms,
        _VanillaSpec(notional=1.0, spot=100.0, strike=100.0, expiry_date=expiry),
    )
    scaled = equity_vanilla_bs_outputs(
        ms,
        _VanillaSpec(notional=10.0, spot=100.0, strike=100.0, expiry_date=expiry),
    )
    # FinancePy's EquityVanillaOption scales value() by num_options but
    # returns single-unit Greeks from delta/gamma/vega/theta.  Match that
    # convention so scorecards compare apples to apples.
    assert scaled["price"] == pytest.approx(10.0 * unit["price"], abs=1e-10)
    for greek in ("delta", "gamma", "vega", "theta"):
        assert scaled[greek] == pytest.approx(unit[greek], abs=1e-12)


def test_expired_option_returns_intrinsic_and_zero_greeks():
    ms = _make_market(rate=0.05, vol=0.25)
    expired_spec = _VanillaSpec(
        notional=1.0,
        spot=120.0,
        strike=100.0,
        expiry_date=date(2024, 11, 15),  # same day as settlement => T == 0
        option_type="call",
    )
    out = equity_vanilla_bs_outputs(ms, expired_spec)
    assert out["price"] == pytest.approx(20.0, abs=1e-9)
    assert out["delta"] == 0.0
    assert out["gamma"] == 0.0
    assert out["vega"] == 0.0
    assert out["theta"] == 0.0


def test_zero_vol_returns_intrinsic_and_zero_greeks():
    ms = _make_market(rate=0.05, vol=0.0)
    spec = _VanillaSpec(
        notional=1.0,
        spot=100.0,
        strike=120.0,
        expiry_date=date(2025, 11, 15),
        option_type="put",
    )
    out = equity_vanilla_bs_outputs(ms, spec)
    # Zero vol ⇒ deterministic payoff; helper reports intrinsic on the zero
    # time branch so scorecards don't divide-by-zero in the d1 expression.
    assert out["price"] == pytest.approx(20.0, abs=1e-9)
    assert out["delta"] == 0.0
    assert out["gamma"] == 0.0
    assert out["vega"] == 0.0
    assert out["theta"] == 0.0


def test_raises_when_required_market_data_is_missing():
    spec = _VanillaSpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
    )
    ms_no_discount = MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=None,
        vol_surface=_FlatBlackVol(0.25),
    )
    with pytest.raises(ValueError, match="discount"):
        equity_vanilla_bs_outputs(ms_no_discount, spec)

    ms_no_vol = MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=_FlatDiscount(0.05),
        vol_surface=None,
    )
    with pytest.raises(ValueError, match="vol_surface"):
        equity_vanilla_bs_outputs(ms_no_vol, spec)


def test_unknown_option_type_raises():
    ms = _make_market(rate=0.05, vol=0.25)
    spec = _VanillaSpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="digital",
    )
    with pytest.raises(ValueError, match="Unsupported option_type"):
        equity_vanilla_bs_outputs(ms, spec)
