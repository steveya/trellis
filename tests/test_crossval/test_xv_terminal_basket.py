from __future__ import annotations

import math

import pytest

financepy = pytest.importorskip("financepy")


@pytest.mark.parametrize(
    ("basket_style", "option_type", "financepy_type_name"),
    [
        ("best_of", "call", "CALL_ON_MAXIMUM"),
        ("best_of", "put", "PUT_ON_MAXIMUM"),
        ("worst_of", "call", "CALL_ON_MINIMUM"),
        ("worst_of", "put", "PUT_ON_MINIMUM"),
    ],
)
def test_stulz_extremum_kernel_matches_financepy(
    basket_style: str,
    option_type: str,
    financepy_type_name: str,
):
    import numpy as raw_np

    from financepy.market.curves.discount_curve_flat import DiscountCurveFlat
    from financepy.products.equity.equity_rainbow_option import (
        EquityRainbowOption,
        EquityRainbowOptionTypes,
    )
    from financepy.utils.date import Date
    from financepy.utils.global_vars import G_DAYS_IN_YEARS

    from trellis.models.analytical.terminal_basket import (
        two_asset_extremum_option_stulz,
    )

    value_date = Date(15, 11, 2024)
    expiry_date = Date(15, 11, 2025)
    rate = 0.04
    dividend_yields = (0.01, 0.025)
    spots = (100.0, 95.0)
    volatilities = (0.20, 0.27)
    correlation = 0.35
    strike = 100.0
    T = (expiry_date - value_date) / G_DAYS_IN_YEARS

    financepy_option = EquityRainbowOption(
        expiry_date,
        getattr(EquityRainbowOptionTypes, financepy_type_name),
        [strike],
        2,
    )
    discount_curve = DiscountCurveFlat(value_date, rate)
    dividend_curves = [
        DiscountCurveFlat(value_date, dividend_yields[0]),
        DiscountCurveFlat(value_date, dividend_yields[1]),
    ]
    financepy_value = financepy_option.value(
        value_date,
        raw_np.asarray(spots, dtype=float),
        discount_curve,
        dividend_curves,
        raw_np.asarray(volatilities, dtype=float),
        raw_np.asarray(
            [[1.0, correlation], [correlation, 1.0]],
            dtype=float,
        ),
    )
    trellis_value = two_asset_extremum_option_stulz(
        spots=spots,
        strike=strike,
        T=T,
        discount_factor=math.exp(-discount_curve.zero_rate(expiry_date) * T),
        dividend_yields=tuple(
            curve.zero_rate(expiry_date) for curve in dividend_curves
        ),
        volatilities=volatilities,
        correlation=correlation,
        basket_style=basket_style,
        option_type=option_type,
    )

    # The two libraries use different bivariate-normal integrations, so the
    # cross-engine tolerance reflects integration noise rather than formula
    # disagreement.
    assert trellis_value == pytest.approx(financepy_value, rel=3e-6, abs=2e-6)
