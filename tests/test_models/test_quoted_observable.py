from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.agent.contract_ir import ParRateTenor, VolPoint
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.curves.yield_curve import YieldCurve


class _SurfaceSmile:
    def black_vol(self, expiry: float, strike: float) -> float:
        return 0.18 + 0.01 * float(expiry) + 0.0005 * float(strike)


@dataclass(frozen=True)
class _CurveSpreadSpec:
    notional: float
    curve_id: str
    lhs_coordinate: object
    rhs_coordinate: object
    convention: str
    expiry_date: date
    day_count: DayCountConvention = DayCountConvention.ACT_365


@dataclass(frozen=True)
class _SurfaceSpreadSpec:
    notional: float
    surface_id: str
    lhs_coordinate: object
    rhs_coordinate: object
    convention: str
    expiry_date: date
    day_count: DayCountConvention = DayCountConvention.ACT_365


def test_curve_quote_spread_helper_discounts_terminal_par_rate_difference():
    from trellis.models.quoted_observable import price_curve_quote_spread_analytical

    market_state = MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve([0.0, 2.0, 10.0], [0.02, 0.025, 0.03]),
        selected_curve_names={"discount_curve": "USD_SWAP"},
    )
    spec = _CurveSpreadSpec(
        notional=1_000_000.0,
        curve_id="USD_SWAP",
        lhs_coordinate=ParRateTenor("10Y"),
        rhs_coordinate=ParRateTenor("2Y"),
        convention="par_rate",
        expiry_date=date(2026, 6, 30),
    )

    price = price_curve_quote_spread_analytical(market_state, spec)

    assert price != 0.0


def test_surface_quote_spread_helper_discounts_terminal_vol_difference():
    from trellis.models.quoted_observable import price_surface_quote_spread_analytical

    market_state = MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        spot=100.0,
        vol_surface=_SurfaceSmile(),
    )
    spec = _SurfaceSpreadSpec(
        notional=100_000.0,
        surface_id="SPX_IV",
        lhs_coordinate=VolPoint("1Y", 0.90, "moneyness"),
        rhs_coordinate=VolPoint("1Y", 1.10, "moneyness"),
        convention="black_vol",
        expiry_date=date(2026, 6, 30),
    )

    price = price_surface_quote_spread_analytical(market_state, spec)

    assert price < 0.0


def test_surface_quote_spread_helper_rejects_unsupported_surface_coordinate_kind():
    from trellis.agent.contract_ir import VolDeltaPoint
    from trellis.models.quoted_observable import price_surface_quote_spread_analytical

    market_state = MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        spot=100.0,
        vol_surface=_SurfaceSmile(),
    )
    spec = _SurfaceSpreadSpec(
        notional=100_000.0,
        surface_id="SPX_IV",
        lhs_coordinate=VolDeltaPoint("1Y", 0.25, "spot"),
        rhs_coordinate=VolDeltaPoint("1Y", 0.10, "spot"),
        convention="black_vol",
        expiry_date=date(2026, 6, 30),
    )

    with pytest.raises(ValueError, match="VolPoint"):
        price_surface_quote_spread_analytical(market_state, spec)
