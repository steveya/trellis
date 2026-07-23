from __future__ import annotations

import numpy as raw_np
import pytest


def test_terminal_basket_payoff_exposes_style_and_option_algebra():
    from trellis.models.payoffs import terminal_basket_option_payoff

    terminal = raw_np.asarray(
        [
            [120.0, 80.0],
            [90.0, 110.0],
        ],
        dtype=float,
    )

    assert terminal_basket_option_payoff(
        terminal,
        weights=(0.5, 0.5),
        basket_style="weighted_sum",
        strike=100.0,
        option_type="call",
    ) == pytest.approx((0.0, 0.0))
    assert terminal_basket_option_payoff(
        terminal,
        weights=(1.0, -1.0),
        basket_style="spread",
        strike=10.0,
        option_type="call",
    ) == pytest.approx((30.0, 0.0))
    assert terminal_basket_option_payoff(
        terminal,
        weights=(0.5, 0.5),
        basket_style="best_of",
        strike=100.0,
        option_type="call",
    ) == pytest.approx((20.0, 10.0))
    assert terminal_basket_option_payoff(
        terminal,
        weights=(0.5, 0.5),
        basket_style="worst_of",
        strike=100.0,
        option_type="put",
    ) == pytest.approx((20.0, 10.0))


@pytest.mark.parametrize("basket_style", ["best_of", "worst_of"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_stulz_extremum_kernel_matches_independent_quadrature(
    basket_style: str,
    option_type: str,
):
    from trellis.models.analytical.terminal_basket import (
        two_asset_extremum_option_stulz,
        two_asset_terminal_basket_gauss_hermite,
    )

    inputs = {
        "spots": (100.0, 95.0),
        "strike": 100.0,
        "T": 1.25,
        "discount_factor": raw_np.exp(-0.04 * 1.25),
        "dividend_yields": (0.01, 0.025),
        "volatilities": (0.20, 0.27),
        "correlation": 0.35,
        "basket_style": basket_style,
        "option_type": option_type,
    }

    stulz = two_asset_extremum_option_stulz(**inputs)
    quadrature = two_asset_terminal_basket_gauss_hermite(
        **inputs,
        weights=(0.5, 0.5),
        n_points=192,
    )

    assert stulz > 0.0
    assert stulz == pytest.approx(quadrature, rel=2e-3, abs=2e-3)


def test_kirk_spread_kernel_tracks_independent_quadrature():
    from trellis.models.analytical.terminal_basket import (
        two_asset_spread_option_kirk,
        two_asset_terminal_basket_gauss_hermite,
    )

    T = 1.0
    discount_factor = raw_np.exp(-0.05 * T)
    spots = (100.0, 95.0)
    dividends = (0.01, 0.02)
    forwards = tuple(
        spot * raw_np.exp((0.05 - dividend) * T)
        for spot, dividend in zip(spots, dividends, strict=True)
    )
    common = {
        "strike": 5.0,
        "T": T,
        "discount_factor": discount_factor,
        "volatilities": (0.20, 0.24),
        "correlation": 0.40,
        "weights": (1.0, -1.0),
        "option_type": "call",
    }

    kirk = two_asset_spread_option_kirk(forwards=forwards, **common)
    quadrature = two_asset_terminal_basket_gauss_hermite(
        spots=spots,
        dividend_yields=dividends,
        basket_style="spread",
        n_points=48,
        **common,
    )

    assert kirk > 0.0
    assert kirk == pytest.approx(quadrature, rel=0.025)


@pytest.mark.parametrize(
    ("strike", "correlation"),
    [
        (1.0, -0.25),
        (5.0, 0.35),
        (15.0, 0.70),
    ],
)
def test_hurd_zhou_2d_fft_tracks_independent_quadrature(
    strike: float,
    correlation: float,
):
    from trellis.models.analytical.terminal_basket import (
        two_asset_terminal_basket_gauss_hermite,
    )
    from trellis.models.transforms.spread_option import (
        correlated_gbm_log_return_characteristic_function,
        hurd_zhou_spread_option_2d_fft,
    )

    T = 1.0
    rate = 0.04
    discount_factor = raw_np.exp(-rate * T)
    spots = (100.0, 95.0)
    dividends = (0.01, 0.02)
    volatilities = (0.20, 0.25)
    weights = (1.0, -1.0)

    characteristic_function = (
        lambda u1, u2: correlated_gbm_log_return_characteristic_function(
            u1,
            u2,
            T=T,
            rate=rate,
            dividend_yields=dividends,
            volatilities=volatilities,
            correlation=correlation,
        )
    )
    transform = hurd_zhou_spread_option_2d_fft(
        characteristic_function,
        spots=spots,
        weights=weights,
        strike=strike,
        discount_factor=discount_factor,
        grid_size=256,
        frequency_step=0.25,
        damping=(-3.0, 1.0),
    )
    quadrature = two_asset_terminal_basket_gauss_hermite(
        spots=spots,
        weights=weights,
        strike=strike,
        T=T,
        discount_factor=discount_factor,
        dividend_yields=dividends,
        volatilities=volatilities,
        correlation=correlation,
        basket_style="spread",
        option_type="call",
        n_points=48,
    )

    assert transform > 0.0
    assert transform == pytest.approx(quadrature, rel=0.015, abs=0.03)


def test_hurd_zhou_2d_fft_rejects_a_non_spread_payoff_contract():
    from trellis.models.transforms.spread_option import (
        correlated_gbm_log_return_characteristic_function,
        hurd_zhou_spread_option_2d_fft,
    )

    characteristic_function = (
        lambda u1, u2: correlated_gbm_log_return_characteristic_function(
            u1,
            u2,
            T=1.0,
            rate=0.04,
            dividend_yields=(0.01, 0.02),
            volatilities=(0.20, 0.25),
            correlation=0.35,
        )
    )

    with pytest.raises(ValueError, match="one positive and one negative"):
        hurd_zhou_spread_option_2d_fft(
            characteristic_function,
            spots=(100.0, 95.0),
            weights=(0.5, 0.5),
            strike=5.0,
            discount_factor=raw_np.exp(-0.04),
        )
