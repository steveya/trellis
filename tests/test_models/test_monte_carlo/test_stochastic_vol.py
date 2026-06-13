from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as raw_np
import pytest

from trellis.core.types import DayCountConvention
from trellis.models.processes.heston import (
    build_heston_parameter_payload,
    resolve_heston_runtime_binding,
)


class FlatDiscountCurve:
    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def zero_rate(self, t: float) -> float:
        return self.rate

    def discount(self, t: float) -> float:
        return float(raw_np.exp(-self.rate * t))


def _market_state(*, v0: float = 0.04):
    return SimpleNamespace(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=FlatDiscountCurve(0.05),
        vol_surface=None,
        selected_curve_names={},
        model_parameters=build_heston_parameter_payload(
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=v0,
            parameter_set_name="heston_equity",
        ),
    )


def _spec(*, option_type: str = "call"):
    return SimpleNamespace(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type=option_type,
        day_count=DayCountConvention.ACT_365,
        dividend_yield=0.0,
    )


def test_heston_qe_scheme_simulates_positive_vector_state_paths():
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.monte_carlo.schemes import (
        HestonQuadraticExponential,
        SCHEME_REGISTRY,
    )

    binding = resolve_heston_runtime_binding(_market_state())
    engine = MonteCarloEngine(
        binding.process,
        n_paths=256,
        n_steps=16,
        seed=123,
        scheme=HestonQuadraticExponential(),
    )

    paths = engine.simulate(raw_np.array([100.0, binding.process.v0]), 1.0)
    repeated = MonteCarloEngine(
        binding.process,
        n_paths=256,
        n_steps=16,
        seed=123,
        scheme=HestonQuadraticExponential(),
    ).simulate(raw_np.array([100.0, binding.process.v0]), 1.0)

    assert SCHEME_REGISTRY["heston_qe"] is HestonQuadraticExponential
    assert paths.shape == (256, 17, 2)
    assert raw_np.all(raw_np.isfinite(paths))
    assert raw_np.all(paths[:, :, 0] > 0.0)
    assert raw_np.all(paths[:, :, 1] >= 0.0)
    assert HestonQuadraticExponential.compatible_process_families == ("heston",)
    raw_np.testing.assert_allclose(paths, repeated, atol=0.0, rtol=0.0)


def test_heston_monte_carlo_problem_exposes_typed_contract_fields():
    from trellis.models.monte_carlo.stochastic_vol import build_heston_monte_carlo_problem

    resolved, problem = build_heston_monte_carlo_problem(
        _market_state(),
        _spec(),
        scheme="qe",
        n_paths=1024,
        n_steps=24,
        seed=99,
    )

    assert resolved.process_family == "heston"
    assert resolved.variance_scheme == "heston_qe"
    assert resolved.payoff_reducer == "terminal_vanilla_option"
    assert resolved.validation_bundle == "heston:monte_carlo"
    assert resolved.n_paths == 1024
    assert resolved.n_steps == 24
    assert resolved.seed == 99
    assert resolved.correlation == pytest.approx(-0.7)
    raw_np.testing.assert_allclose(resolved.initial_state, raw_np.array([100.0, 0.04]))
    assert getattr(problem.scheme, "name") == "heston_qe"
    assert problem.initial_state.shape == (2,)
    assert problem.path_requirement.full_path is False


def test_heston_qe_monte_carlo_price_is_reproducible_and_close_to_transform():
    from trellis.models.monte_carlo.stochastic_vol import (
        price_heston_option_monte_carlo_result,
    )
    from trellis.models.transforms.heston import price_heston_option_transform

    market_state = _market_state()
    spec = _spec()
    qe = price_heston_option_monte_carlo_result(
        market_state,
        spec,
        scheme="qe_heston",
        n_paths=20_000,
        n_steps=80,
        seed=17,
    )
    repeated = price_heston_option_monte_carlo_result(
        market_state,
        spec,
        scheme="qe_heston",
        n_paths=20_000,
        n_steps=80,
        seed=17,
    )
    euler = price_heston_option_monte_carlo_result(
        market_state,
        spec,
        scheme="euler_heston",
        n_paths=20_000,
        n_steps=80,
        seed=17,
    )
    reference = price_heston_option_transform(market_state, spec, method="cos")

    assert qe.variance_scheme == "heston_qe"
    assert euler.variance_scheme == "euler"
    assert qe.validation_bundle == "heston:monte_carlo"
    assert qe.price == pytest.approx(repeated.price, rel=0.0, abs=0.0)
    assert qe.std_error == pytest.approx(repeated.std_error, rel=0.0, abs=0.0)
    assert qe.price > 0.0
    assert qe.std_error > 0.0
    assert abs(qe.price - reference) < max(4.0 * qe.std_error, 0.20 * reference)
