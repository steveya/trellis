from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


class _Spec:
    notional = 100.0
    spot = 100.0
    strike = 100.0
    expiry_date = date(2025, 11, 15)
    option_type = "call"
    n_paths = 120_000
    seed = 17


def _market_state(vol: float = 0.20) -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=5.0),
        vol_surface=FlatVol(vol),
        jump_parameter_sets={
            "merton_equity": {
                "mu": 0.05,
                "sigma": vol,
                "lam": 0.35,
                "jump_mean": -0.08,
                "jump_vol": 0.18,
            }
        },
    )


def test_resolve_merton_jump_diffusion_inputs_reads_market_jump_parameters():
    from trellis.models.merton_jump_diffusion_option import (
        resolve_merton_jump_diffusion_option_inputs,
    )

    resolved = resolve_merton_jump_diffusion_option_inputs(_market_state(), _Spec())

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.05)
    assert resolved.sigma == pytest.approx(0.20)
    assert resolved.jump_intensity == pytest.approx(0.35)
    assert resolved.jump_mean == pytest.approx(-0.08)
    assert resolved.jump_vol == pytest.approx(0.18)


@pytest.mark.parametrize("method", ["fft", "cos"])
def test_merton_jump_diffusion_transform_matches_poisson_series(method: str):
    from trellis.models.merton_jump_diffusion_option import (
        price_merton_jump_diffusion_option_poisson_series,
        price_merton_jump_diffusion_option_transform,
    )

    market_state = _market_state()
    spec = _Spec()

    transform_price = price_merton_jump_diffusion_option_transform(
        market_state,
        spec,
        method=method,
    )
    reference = price_merton_jump_diffusion_option_poisson_series(market_state, spec)

    assert transform_price == pytest.approx(reference, rel=0.03)


def test_merton_cos_default_is_stable_for_low_diffusion_volatility():
    from trellis.models.merton_jump_diffusion_option import (
        price_merton_jump_diffusion_option_poisson_series,
        price_merton_jump_diffusion_option_transform,
    )

    low_vol_market = _market_state(vol=0.05)
    higher_vol_market = _market_state(vol=0.10)
    spec = _Spec()

    low_vol_cos = price_merton_jump_diffusion_option_transform(
        low_vol_market,
        spec,
        method="cos",
    )
    low_vol_reference = price_merton_jump_diffusion_option_poisson_series(
        low_vol_market,
        spec,
    )
    higher_vol_cos = price_merton_jump_diffusion_option_transform(
        higher_vol_market,
        spec,
        method="cos",
    )

    assert low_vol_cos == pytest.approx(low_vol_reference, rel=0.01)
    assert higher_vol_cos > low_vol_cos


def test_merton_jump_diffusion_monte_carlo_agrees_with_poisson_series():
    from trellis.models.merton_jump_diffusion_option import (
        price_merton_jump_diffusion_option_monte_carlo_result,
        price_merton_jump_diffusion_option_poisson_series,
    )

    market_state = _market_state()
    spec = _Spec()

    result = price_merton_jump_diffusion_option_monte_carlo_result(
        market_state,
        spec,
        n_paths=160_000,
        seed=29,
    )
    reference = price_merton_jump_diffusion_option_poisson_series(market_state, spec)

    assert result.price == pytest.approx(reference, rel=0.03)
    assert result.standard_error > 0.0
