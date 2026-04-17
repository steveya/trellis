"""Tests for the Garman-Kohlhagen FX vanilla native outputs helper (QUA-878)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.fx_vanilla_gk import fx_vanilla_gk_outputs


class _FlatDiscount:
    def __init__(self, rate: float):
        self._rate = float(rate)

    def zero_rate(self, t: float) -> float:
        return self._rate

    def discount(self, t: float) -> float:
        return float(np.exp(-self._rate * float(t)))


class _FlatVol:
    def __init__(self, vol: float):
        self._vol = float(vol)

    def black_vol(self, t: float, k: float) -> float:
        return self._vol


class _FXQuote:
    def __init__(self, spot: float):
        self.spot = float(spot)


@dataclass(frozen=True)
class _FxVanillaSpec:
    notional: float
    strike: float
    expiry_date: date
    fx_pair: str = "EURUSD"
    foreign_discount_key: str = "EUR_OIS"
    option_type: str = "call"
    day_count: DayCountConvention = DayCountConvention.ACT_365


def _make_market(
    *,
    domestic_rate: float = 0.04,
    foreign_rate: float = 0.02,
    vol: float = 0.10,
    spot: float = 1.05,
) -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=_FlatDiscount(domestic_rate),
        forecast_curves={"EUR_OIS": _FlatDiscount(foreign_rate)},
        fx_rates={"EURUSD": _FXQuote(spot)},
        vol_surface=_FlatVol(vol),
    )


def test_call_outputs_match_expected_keys_and_reasonable_values():
    ms = _make_market()
    spec = _FxVanillaSpec(
        notional=1_000_000.0,
        strike=1.10,
        expiry_date=date(2025, 11, 15),
        option_type="call",
    )
    out = fx_vanilla_gk_outputs(ms, spec)
    assert set(out) == {"price", "delta", "gamma", "vega", "theta"}
    # Reference values were produced against FinancePy's FXVanillaOption with
    # r_d=4%, r_f=2%, σ=10%, S=1.05, K=1.10, T=1 year.  The Trellis helper
    # matches FinancePy's scalar outputs (``pips_spot_delta`` for delta)
    # within day-count rounding (FinancePy uses 365.2425, Trellis ACT/365).
    assert out["price"] == pytest.approx(29216.84, abs=1.0)
    assert out["delta"] == pytest.approx(0.40659, abs=5e-4)
    assert out["gamma"] == pytest.approx(3.6390, abs=5e-4)
    assert out["vega"] == pytest.approx(0.4012, abs=5e-4)
    assert out["theta"] == pytest.approx(-0.02743, abs=5e-4)


def test_put_outputs_respect_delta_parity_across_call_and_put():
    """call_delta - put_delta = df_foreign under Garman-Kohlhagen (q = r_f)."""
    ms = _make_market()
    expiry = date(2025, 11, 15)
    call = fx_vanilla_gk_outputs(
        ms,
        _FxVanillaSpec(notional=1.0, strike=1.10, expiry_date=expiry, option_type="call"),
    )
    put = fx_vanilla_gk_outputs(
        ms,
        _FxVanillaSpec(notional=1.0, strike=1.10, expiry_date=expiry, option_type="put"),
    )
    df_f = float(np.exp(-0.02 * 1.0))
    assert (call["delta"] - put["delta"]) == pytest.approx(df_f, abs=1e-6)
    # Gamma and vega are parity-invariant.
    assert call["gamma"] == pytest.approx(put["gamma"], abs=1e-12)
    assert call["vega"] == pytest.approx(put["vega"], abs=1e-12)


def test_notional_scales_price_only():
    """Match FinancePy: ``value`` is scaled by num_options; Greeks are per-unit."""
    ms = _make_market()
    expiry = date(2025, 11, 15)
    unit = fx_vanilla_gk_outputs(
        ms,
        _FxVanillaSpec(notional=1.0, strike=1.10, expiry_date=expiry),
    )
    scaled = fx_vanilla_gk_outputs(
        ms,
        _FxVanillaSpec(notional=10.0, strike=1.10, expiry_date=expiry),
    )
    assert scaled["price"] == pytest.approx(10.0 * unit["price"], abs=1e-12)
    for greek in ("delta", "gamma", "vega", "theta"):
        assert scaled[greek] == pytest.approx(unit[greek], abs=1e-12)


def test_price_agrees_with_price_fx_vanilla_analytical():
    """``fx_vanilla_gk_outputs['price']`` must match ``price_fx_vanilla_analytical``.

    Both paths route through ``resolve_fx_vanilla_inputs`` and then compute
    the Garman-Kohlhagen price.  The cold-run evaluate() uses the raw basis
    kernel; our native-outputs helper assembles the same price via a direct
    closed form.  They must agree otherwise the benchmark scorecard would
    see a self-inconsistency between ``evaluate()`` and ``benchmark_outputs
    ()``.
    """
    from trellis.models.fx_vanilla import price_fx_vanilla_analytical

    ms = _make_market()
    for opt_type in ("call", "put"):
        spec = _FxVanillaSpec(
            notional=2_500_000.0,
            strike=1.05,
            expiry_date=date(2025, 11, 15),
            option_type=opt_type,
        )
        native = fx_vanilla_gk_outputs(ms, spec)
        raw = float(price_fx_vanilla_analytical(ms, spec))
        assert native["price"] == pytest.approx(raw, rel=1e-10), opt_type


def test_zero_time_outputs_return_intrinsic_and_zero_greeks():
    ms = _make_market()
    # expiry == settlement -> T == 0
    spec = _FxVanillaSpec(
        notional=1.0,
        strike=1.00,
        expiry_date=date(2024, 11, 15),
        option_type="call",
    )
    out = fx_vanilla_gk_outputs(ms, spec)
    assert out["price"] == pytest.approx(0.05, abs=1e-9)  # spot 1.05 - strike 1.00
    assert out["delta"] == 0.0
    assert out["gamma"] == 0.0
    assert out["vega"] == 0.0
    assert out["theta"] == 0.0


def test_zero_vol_uses_discounted_forward_intrinsic():
    """Zero-vol, T > 0 must match ``df_domestic * max(F - K, 0)`` under GK.

    Plain spot-vs-strike intrinsic would silently misprice parity whenever
    the two rate curves differ, which is the common case for FX.
    """
    ms = _make_market(domestic_rate=0.04, foreign_rate=0.02, vol=0.0)
    call = fx_vanilla_gk_outputs(
        ms,
        _FxVanillaSpec(
            notional=1.0,
            strike=1.02,
            expiry_date=date(2025, 11, 15),
            option_type="call",
        ),
    )
    T = 1.0
    df_d = float(np.exp(-0.04 * T))
    df_f = float(np.exp(-0.02 * T))
    forward = 1.05 * df_f / df_d
    expected = df_d * max(forward - 1.02, 0.0)
    assert call["price"] == pytest.approx(expected, abs=1e-9)
    assert call["delta"] == 0.0


def test_unknown_option_type_raises():
    ms = _make_market()
    spec = _FxVanillaSpec(
        notional=1.0,
        strike=1.10,
        expiry_date=date(2025, 11, 15),
        option_type="digital",
    )
    with pytest.raises(ValueError, match="Unsupported option_type"):
        fx_vanilla_gk_outputs(ms, spec)
