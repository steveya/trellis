from __future__ import annotations

from datetime import date
from types import SimpleNamespace

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
    n_paths = 120_000
    n_steps = 96
    seed = 17


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=5.0),
        spot=100.0,
        model_parameters={
            "sabr": {
                "alpha": 0.24,
                "beta": 0.5,
                "rho": -0.2,
                "nu": 0.35,
            }
        },
    )


def test_resolve_sabr_forward_option_inputs_reads_market_model_parameters():
    from trellis.models.sabr_option import resolve_sabr_forward_option_inputs

    resolved = resolve_sabr_forward_option_inputs(_market_state(), _Spec())

    assert resolved.forward == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.05)
    assert resolved.alpha == pytest.approx(0.24)
    assert resolved.beta == pytest.approx(0.5)
    assert resolved.rho == pytest.approx(-0.2)
    assert resolved.nu == pytest.approx(0.35)


def test_resolve_sabr_forward_option_inputs_reads_named_model_parameter_set():
    from trellis.models.sabr_option import resolve_sabr_forward_option_inputs

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=5.0),
        spot=100.0,
        model_parameter_sets={
            "sabr_validation": {
                "alpha": 0.21,
                "beta": 0.6,
                "rho": -0.15,
                "nu": 0.31,
            }
        },
    )
    spec = SimpleNamespace(
        notional=100.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
        model_parameter_set="sabr_validation",
    )

    resolved = resolve_sabr_forward_option_inputs(market_state, spec)

    assert resolved.alpha == pytest.approx(0.21)
    assert resolved.beta == pytest.approx(0.6)
    assert resolved.rho == pytest.approx(-0.15)
    assert resolved.nu == pytest.approx(0.31)


def test_sabr_mc_agrees_with_hagan_price_on_stable_fixture():
    from trellis.models.sabr_option import (
        price_sabr_forward_option_hagan,
        price_sabr_forward_option_monte_carlo_result,
    )

    market_state = _market_state()
    spec = _Spec()

    analytical = price_sabr_forward_option_hagan(market_state, spec)
    mc = price_sabr_forward_option_monte_carlo_result(
        market_state,
        spec,
        n_paths=180_000,
        n_steps=128,
        seed=29,
    )

    assert mc.price == pytest.approx(analytical, rel=0.05)
    assert mc.standard_error > 0.0


def test_sabr_hagan_uses_synthetic_market_provenance_rate_vol_pack():
    from trellis.models.sabr_option import resolve_sabr_forward_option_inputs

    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=5.0),
        spot=100.0,
        market_provenance={
            "prior_parameters": {
                "synthetic_generation_contract": {
                    "model_packs": {
                        "rates": {
                            "rate_vol_model": {
                                "family": "sabr",
                                "alpha": 0.18,
                                "beta": 0.5,
                                "rho": -0.1,
                                "nu": 0.42,
                            }
                        }
                    }
                }
            }
        },
    )

    resolved = resolve_sabr_forward_option_inputs(market_state, _Spec())

    assert resolved.alpha == pytest.approx(0.18)
    assert resolved.rho == pytest.approx(-0.1)
    assert resolved.nu == pytest.approx(0.42)
