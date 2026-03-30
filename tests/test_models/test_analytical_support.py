from __future__ import annotations

import math

import pytest


def test_terminal_intrinsic_handles_call_and_put():
    from trellis.models.analytical.support import (
        normalized_option_type,
        terminal_intrinsic,
    )

    assert terminal_intrinsic("call", spot=105.0, strike=100.0) == pytest.approx(5.0)
    assert terminal_intrinsic("put", spot=95.0, strike=100.0) == pytest.approx(5.0)
    assert terminal_intrinsic("call", spot=95.0, strike=100.0) == pytest.approx(0.0)
    assert normalized_option_type("CALL") == "call"
    assert normalized_option_type("Put") == "put"


def test_terminal_intrinsic_supports_autograd():
    from trellis.core.differentiable import gradient
    from trellis.models.analytical.support import terminal_intrinsic

    call_delta = gradient(lambda spot: terminal_intrinsic("call", spot=spot, strike=100.0))
    put_delta = gradient(lambda spot: terminal_intrinsic("put", spot=spot, strike=100.0))

    assert call_delta(105.0) == pytest.approx(1.0)
    assert call_delta(95.0) == pytest.approx(0.0)
    assert put_delta(95.0) == pytest.approx(-1.0)
    assert put_delta(105.0) == pytest.approx(0.0)


def test_digital_and_asset_intrinsics_reduce_to_expected_terminal_payoffs():
    from trellis.models.analytical.support import (
        asset_or_nothing_intrinsic,
        cash_or_nothing_intrinsic,
    )

    assert cash_or_nothing_intrinsic("call", spot=105.0, strike=100.0, cash=7.5) == pytest.approx(7.5)
    assert cash_or_nothing_intrinsic("put", spot=105.0, strike=100.0, cash=7.5) == pytest.approx(0.0)
    assert asset_or_nothing_intrinsic("call", spot=105.0, strike=100.0) == pytest.approx(105.0)
    assert asset_or_nothing_intrinsic("put", spot=95.0, strike=100.0) == pytest.approx(95.0)


def test_terminal_vanilla_from_basis_reconstructs_vanilla_payoffs():
    from trellis.models.analytical.support import terminal_vanilla_from_basis

    assert terminal_vanilla_from_basis(
        "call",
        asset_value=105.0,
        cash_value=1.0,
        strike=100.0,
    ) == pytest.approx(5.0)
    assert terminal_vanilla_from_basis(
        "put",
        asset_value=95.0,
        cash_value=1.0,
        strike=100.0,
    ) == pytest.approx(5.0)


def test_call_put_parity_gap_is_zero_when_discounted_forward_identity_holds():
    from trellis.models.analytical.support import call_put_parity_gap

    call_value = 12.0
    put_value = 6.0
    forward = 106.0
    strike = 100.0
    discount_factor = 1.0

    assert call_put_parity_gap(
        call_value=call_value,
        put_value=put_value,
        forward=forward,
        strike=strike,
        discount_factor=discount_factor,
    ) == pytest.approx(0.0)


def test_discount_helpers_round_trip_zero_rate_and_discount_factor():
    from trellis.models.analytical.support import (
        continuous_rate_from_simple_rate,
        discount_factor_from_zero_rate,
        discounted_value,
        implied_zero_rate,
        safe_time_fraction,
        simple_rate_from_discount_factor,
    )

    discount_factor = discount_factor_from_zero_rate(0.05, 2.0)
    implied_rate = implied_zero_rate(discount_factor, 2.0)
    continuous_rate = continuous_rate_from_simple_rate(0.05, 1.0)
    round_trip_simple_rate = simple_rate_from_discount_factor(
        discount_factor_from_zero_rate(continuous_rate, 1.0),
        1.0,
    )

    assert discount_factor == pytest.approx(math.exp(-0.10))
    assert implied_rate == pytest.approx(0.05)
    assert continuous_rate == pytest.approx(math.log(1.05))
    assert round_trip_simple_rate == pytest.approx(0.05)
    assert discounted_value(12.0, discount_factor, scale=10.0) == pytest.approx(
        120.0 * discount_factor
    )
    assert safe_time_fraction(-0.25) == pytest.approx(0.0)
    assert safe_time_fraction(1.5) == pytest.approx(1.5)


def test_forward_from_discount_factors_matches_standard_carry_relation():
    from trellis.models.analytical.support import (
        forward_from_carry_rate,
        forward_from_discount_factors,
        forward_from_dividend_yield,
    )

    forward = forward_from_discount_factors(
        spot=100.0,
        domestic_df=math.exp(-0.05),
        foreign_df=math.exp(-0.03),
    )

    assert forward == pytest.approx(100.0 * math.exp(0.02))
    assert forward_from_carry_rate(spot=100.0, carry_rate=0.02, T=1.0) == pytest.approx(forward)
    assert forward_from_dividend_yield(
        spot=100.0,
        domestic_rate=0.05,
        dividend_yield=0.03,
        T=1.0,
    ) == pytest.approx(forward)


def test_quanto_adjusted_forward_reduces_cleanly_when_adjustment_is_zero():
    from trellis.models.analytical.support import (
        effective_covariance_term,
        exchange_option_effective_vol,
        foreign_to_domestic_forward_bridge,
        forward_from_discount_factors,
        quanto_adjusted_forward,
    )

    standard_forward = forward_from_discount_factors(
        spot=100.0,
        domestic_df=math.exp(-0.05),
        foreign_df=math.exp(-0.03),
    )

    zero_corr_forward = quanto_adjusted_forward(
        spot=100.0,
        domestic_df=math.exp(-0.05),
        foreign_df=math.exp(-0.03),
        corr=0.0,
        sigma_underlier=0.20,
        sigma_fx=0.15,
        T=1.0,
    )
    zero_fx_vol_forward = quanto_adjusted_forward(
        spot=100.0,
        domestic_df=math.exp(-0.05),
        foreign_df=math.exp(-0.03),
        corr=0.35,
        sigma_underlier=0.20,
        sigma_fx=0.0,
        T=1.0,
    )

    covariance_term = effective_covariance_term(
        corr=0.35,
        sigma_1=0.20,
        sigma_2=0.15,
    )
    effective_vol = exchange_option_effective_vol(
        sigma_1=0.20,
        sigma_2=0.15,
        corr=0.35,
    )
    bridged_forward = foreign_to_domestic_forward_bridge(
        spot=100.0,
        domestic_df=math.exp(-0.05),
        foreign_df=math.exp(-0.03),
    )

    assert zero_corr_forward == pytest.approx(standard_forward)
    assert zero_fx_vol_forward == pytest.approx(standard_forward)
    assert covariance_term == pytest.approx(0.35 * 0.20 * 0.15)
    assert effective_vol == pytest.approx(
        math.sqrt(0.20 ** 2 + 0.15 ** 2 - 2.0 * 0.35 * 0.20 * 0.15)
    )
    assert bridged_forward == pytest.approx(standard_forward)


def test_quanto_analytical_route_still_prices_through_support_helpers():
    from datetime import date

    from trellis.curves.yield_curve import YieldCurve
    from trellis.core.market_state import MarketState
    from trellis.core.types import DayCountConvention
    from trellis.instruments.fx import FXRate
    from trellis.models.analytical.quanto import price_quanto_option_raw
    from trellis.models.resolution.quanto import resolve_quanto_inputs
    from trellis.models.vol_surface import FlatVol

    class Spec:
        notional = 100_000
        strike = 100.0
        expiry_date = date(2025, 11, 15)
        fx_pair = "EURUSD"
        underlier_currency = "EUR"
        domestic_currency = "USD"
        option_type = "call"
        quanto_correlation_key = None
        day_count = DayCountConvention.ACT_365

    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"quanto_correlation": 0.35},
    )

    resolved = resolve_quanto_inputs(market_state, Spec)
    price = price_quanto_option_raw(Spec, resolved)

    assert price > 0.0


def test_quanto_raw_kernel_autograd_matches_finite_difference():
    from dataclasses import replace
    from datetime import date

    from trellis.core.differentiable import gradient
    from trellis.curves.yield_curve import YieldCurve
    from trellis.core.market_state import MarketState
    from trellis.core.types import DayCountConvention
    from trellis.instruments.fx import FXRate
    from trellis.models.analytical.quanto import price_quanto_option_raw
    from trellis.models.resolution.quanto import resolve_quanto_inputs
    from trellis.models.vol_surface import FlatVol

    class Spec:
        notional = 100_000
        strike = 100.0
        expiry_date = date(2025, 11, 15)
        fx_pair = "EURUSD"
        underlier_currency = "EUR"
        domestic_currency = "USD"
        option_type = "call"
        quanto_correlation_key = None
        day_count = DayCountConvention.ACT_365

    settle = date(2024, 11, 15)
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"quanto_correlation": 0.35},
    )
    resolved = resolve_quanto_inputs(market_state, Spec)

    def _fd(fn, x, eps=1e-6):
        return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)

    spot_delta = gradient(
        lambda spot: price_quanto_option_raw(Spec, replace(resolved, spot=spot))
    )(resolved.spot)
    spot_fd = _fd(
        lambda spot: price_quanto_option_raw(Spec, replace(resolved, spot=spot)),
        resolved.spot,
    )

    sigma_vega = gradient(
        lambda vol: price_quanto_option_raw(
            Spec,
            replace(resolved, sigma_underlier=vol),
        )
    )(resolved.sigma_underlier)
    sigma_fd = _fd(
        lambda vol: price_quanto_option_raw(
            Spec,
            replace(resolved, sigma_underlier=vol),
        ),
        resolved.sigma_underlier,
    )

    assert spot_delta == pytest.approx(spot_fd, rel=1e-5, abs=1e-8)
    assert sigma_vega == pytest.approx(sigma_fd, rel=1e-5, abs=1e-8)
