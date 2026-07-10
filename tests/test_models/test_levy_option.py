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
    n_paths = 120_000
    seed = 17


def _market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.05, max_tenor=5.0),
        spot=100.0,
        model_parameter_sets={
            "variance_gamma_equity": {
                "family": "variance_gamma",
                "sigma": 0.18,
                "theta": -0.08,
                "nu": 0.20,
            },
            "cgmy_equity": {
                "family": "cgmy",
                "C": 0.40,
                "G": 5.0,
                "M": 6.0,
                "Y": 0.55,
            },
            "kou_equity": {
                "family": "kou",
                "sigma": 0.18,
                "jump_intensity": 0.35,
                "up_probability": 0.35,
                "eta_up": 8.0,
                "eta_down": 6.0,
            },
        },
    )


class _VarianceGammaSpec(_Spec):
    model_parameter_set = "variance_gamma_equity"


class _CgmySpec(_Spec):
    model_parameter_set = "cgmy_equity"


class _KouSpec(_Spec):
    model_parameter_set = "kou_equity"


def test_resolve_variance_gamma_inputs_reads_named_model_parameter_set():
    from trellis.models.levy_option import resolve_levy_option_inputs

    resolved = resolve_levy_option_inputs(
        _market_state(),
        _VarianceGammaSpec(),
        model_family="variance_gamma",
    )

    assert resolved.spot == pytest.approx(100.0)
    assert resolved.strike == pytest.approx(100.0)
    assert resolved.maturity == pytest.approx(1.0)
    assert resolved.rate == pytest.approx(0.05)
    assert resolved.model_family == "variance_gamma"
    assert resolved.parameters["sigma"] == pytest.approx(0.18)
    assert resolved.parameters["theta"] == pytest.approx(-0.08)
    assert resolved.parameters["nu"] == pytest.approx(0.20)


@pytest.mark.parametrize("method", ["fft", "cos"])
def test_variance_gamma_transform_and_monte_carlo_agree(method: str):
    from trellis.models.levy_option import (
        price_variance_gamma_option_monte_carlo_result,
        price_variance_gamma_option_transform,
    )

    market_state = _market_state()
    spec = _VarianceGammaSpec()

    transform_price = price_variance_gamma_option_transform(
        market_state,
        spec,
        method=method,
    )
    mc_result = price_variance_gamma_option_monte_carlo_result(
        market_state,
        spec,
        n_paths=180_000,
        seed=29,
    )

    assert mc_result.price == pytest.approx(transform_price, rel=0.04)
    assert mc_result.standard_error > 0.0


def test_cgmy_cos_and_terminal_distribution_monte_carlo_agree():
    from trellis.models.levy_option import (
        price_cgmy_option_monte_carlo_result,
        price_cgmy_option_transform,
    )

    market_state = _market_state()
    spec = _CgmySpec()

    transform_price = price_cgmy_option_transform(
        market_state,
        spec,
        method="cos",
        cos_points=1024,
    )
    mc_result = price_cgmy_option_monte_carlo_result(
        market_state,
        spec,
        n_paths=120_000,
        seed=31,
    )

    assert mc_result.price == pytest.approx(transform_price, rel=0.05)
    assert mc_result.standard_error > 0.0


def test_resolve_kou_inputs_reads_named_model_parameter_set():
    from trellis.models.levy_option import resolve_levy_option_inputs

    resolved = resolve_levy_option_inputs(
        _market_state(),
        _KouSpec(),
        model_family="double_exponential_jump_diffusion",
    )

    assert resolved.model_family == "kou"
    assert resolved.parameters["sigma"] == pytest.approx(0.18)
    assert resolved.parameters["jump_intensity"] == pytest.approx(0.35)
    assert resolved.parameters["up_probability"] == pytest.approx(0.35)
    assert resolved.parameters["eta_up"] == pytest.approx(8.0)
    assert resolved.parameters["eta_down"] == pytest.approx(6.0)


@pytest.mark.parametrize("method", ["fft", "cos"])
def test_kou_transform_and_monte_carlo_agree(method: str):
    from trellis.models.levy_option import (
        price_kou_option_monte_carlo_result,
        price_kou_option_transform,
    )

    market_state = _market_state()
    spec = _KouSpec()

    transform_price = price_kou_option_transform(
        market_state,
        spec,
        method=method,
    )
    mc_result = price_kou_option_monte_carlo_result(
        market_state,
        spec,
        n_paths=180_000,
        seed=37,
    )

    assert mc_result.price == pytest.approx(transform_price, rel=0.04)
    assert mc_result.standard_error > 0.0
