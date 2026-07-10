from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve


SETTLE = date(2024, 11, 15)


class _Spec:
    notional = 100.0
    spot = 100.0
    strike = 100.0
    expiry_date = date(2025, 11, 15)
    option_type = "call"
    n_paths = 80_000
    n_steps = 96
    seed = 17


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=5.0),
        spot=100.0,
        model_parameter_sets={
            "heston_equity": {
                "model_family": "heston",
                "kappa": 2.0,
                "theta": 0.04,
                "xi": 0.30,
                "rho": -0.55,
                "v0": 0.04,
            }
        },
        jump_parameter_sets={
            "merton_equity": {
                "sigma": 0.20,
                "lam": 0.25,
                "jump_mean": -0.06,
                "jump_vol": 0.16,
            }
        },
    )


def test_resolve_bates_inputs_reads_heston_and_jump_parameter_sets():
    from trellis.models.bates_option import resolve_bates_option_inputs

    resolved = resolve_bates_option_inputs(_market_state(), _Spec())

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.05)
    assert resolved.jump_intensity == pytest.approx(0.25)
    assert resolved.jump_mean == pytest.approx(-0.06)
    assert resolved.jump_vol == pytest.approx(0.16)
    assert resolved.characteristic_family == "bates_log_spot"
    assert resolved.validation_bundle == "bates:affine_jump_stochastic_vol"


@pytest.mark.parametrize("method", ["fft", "cos"])
def test_bates_transform_and_monte_carlo_agree(method: str):
    from trellis.models.bates_option import (
        price_bates_option_monte_carlo_result,
        price_bates_option_transform,
    )

    market_state = _market_state()
    spec = _Spec()

    transform_price = price_bates_option_transform(
        market_state,
        spec,
        method=method,
    )
    mc_result = price_bates_option_monte_carlo_result(
        market_state,
        spec,
        n_paths=100_000,
        n_steps=96,
        seed=29,
    )

    assert mc_result.price == pytest.approx(transform_price, rel=0.06)
    assert mc_result.standard_error > 0.0
