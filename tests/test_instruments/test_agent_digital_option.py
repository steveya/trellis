from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments._agent.digitaloption import DigitalOptionPayoff, DigitalOptionSpec
from trellis.models.analytical.equity_exotics import (
    price_equity_digital_option_analytical,
)
from trellis.models.black import (
    black76_asset_or_nothing_call,
    black76_asset_or_nothing_put,
    black76_cash_or_nothing_call,
    black76_cash_or_nothing_put,
)
from trellis.models.analytical.support import forward_from_dividend_yield
from trellis.models.vol_surface import FlatVol


SETTLEMENT = date(2024, 11, 15)
EXPIRY = date(2025, 11, 15)


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLEMENT,
        settlement=SETTLEMENT,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.22),
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_checked_cash_digital_adapter_matches_reference_wrapper(option_type):
    spec = DigitalOptionSpec(
        notional=2.0,
        spot=100.0,
        strike=105.0,
        expiry_date=EXPIRY,
        option_type=option_type,
        payout_type="cash_or_nothing",
        cash_payoff=10.0,
    )

    actual = DigitalOptionPayoff(spec).evaluate(_market_state())
    expected = price_equity_digital_option_analytical(_market_state(), spec)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("option_type", "payout_type", "kernel"),
    [
        ("call", "cash_or_nothing", black76_cash_or_nothing_call),
        ("put", "cash_or_nothing", black76_cash_or_nothing_put),
        ("call", "asset_or_nothing", black76_asset_or_nothing_call),
        ("put", "asset_or_nothing", black76_asset_or_nothing_put),
    ],
)
def test_checked_digital_adapter_selects_declared_basis_kernel(
    option_type, payout_type, kernel
):
    market_state = _market_state()
    spec = DigitalOptionSpec(
        notional=2.0,
        spot=100.0,
        strike=105.0,
        expiry_date=EXPIRY,
        option_type=option_type,
        payout_type=payout_type,
        cash_payoff=10.0,
        dividend_yield=0.01,
    )
    maturity = 1.0
    discount = market_state.discount.discount(maturity)
    forward = forward_from_dividend_yield(
        spot=spec.spot,
        domestic_rate=market_state.discount.zero_rate(maturity),
        dividend_yield=spec.dividend_yield,
        T=maturity,
    )
    payout_scale = spec.cash_payoff if payout_type == "cash_or_nothing" else 1.0
    expected = (
        spec.notional
        * discount
        * payout_scale
        * kernel(forward, spec.strike, 0.22, maturity)
    )

    actual = DigitalOptionPayoff(spec).evaluate(market_state)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [("option_type", "straddle"), ("payout_type", "deferred_cash")],
)
def test_checked_digital_adapter_rejects_unsupported_contract_values(field, value):
    kwargs = {
        "notional": 1.0,
        "spot": 100.0,
        "strike": 100.0,
        "expiry_date": EXPIRY,
        field: value,
    }

    with pytest.raises(ValueError, match=f"Unsupported {field}"):
        DigitalOptionPayoff(DigitalOptionSpec(**kwargs)).evaluate(_market_state())


def test_checked_digital_adapter_source_has_no_product_pricing_helper():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "trellis/instruments/_agent/digitaloption.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "price_equity_digital_option_analytical" not in source
    assert "resolve_single_state_diffusion_inputs" in source
    assert "discounted_value" in source
    assert "black76_cash_or_nothing_call" in source
    assert "black76_asset_or_nothing_put" in source
    assert "cash-or-nothing and asset-or-nothing" in DigitalOptionSpec.__doc__
