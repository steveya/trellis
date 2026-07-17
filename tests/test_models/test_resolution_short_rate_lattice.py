from __future__ import annotations

from types import SimpleNamespace

import pytest

from trellis.models.short_rate_lattice import (
    ResolvedShortRateLatticeInputs,
    resolve_short_rate_lattice_inputs,
)


class _DiscountCurve:
    def __init__(self, rate: float = 0.04):
        self.rate = float(rate)
        self.requested_times: list[float] = []

    def zero_rate(self, time: float) -> float:
        self.requested_times.append(float(time))
        return self.rate


class _VolSurface:
    def __init__(self, volatility: float = 0.20):
        self.volatility = float(volatility)
        self.requests: list[tuple[float, float]] = []

    def black_vol(self, time: float, strike: float) -> float:
        self.requests.append((float(time), float(strike)))
        return self.volatility


def _market_state(
    *,
    rate: float = 0.04,
    volatility: float | None = 0.20,
    model_parameters: dict[str, object] | None = None,
):
    surface = None if volatility is None else _VolSurface(volatility)
    return SimpleNamespace(
        discount=_DiscountCurve(rate),
        vol_surface=surface,
        model_parameters=model_parameters,
        model_parameter_sets={},
    )


def test_resolve_normal_short_rate_lattice_inputs_converts_black_volatility() -> None:
    market_state = _market_state(rate=0.04, volatility=0.20)

    resolved = resolve_short_rate_lattice_inputs(
        market_state,
        horizon=5.0,
        model="hull_white",
        volatility_time=2.0,
        volatility_strike=0.03,
    )

    assert isinstance(resolved, ResolvedShortRateLatticeInputs)
    assert resolved.model_name == "hull_white"
    assert resolved.horizon == pytest.approx(5.0)
    assert resolved.r0 == pytest.approx(0.04)
    assert resolved.mean_reversion == pytest.approx(0.1)
    assert resolved.sigma == pytest.approx(0.20 * 0.04)
    assert resolved.n_steps == 200
    assert market_state.discount.requested_times == [pytest.approx(2.5)]
    assert market_state.vol_surface.requests == [
        (pytest.approx(2.0), pytest.approx(0.03))
    ]


def test_resolve_lognormal_short_rate_lattice_inputs_retains_black_volatility() -> None:
    market_state = _market_state(rate=0.04, volatility=0.20)

    resolved = resolve_short_rate_lattice_inputs(
        market_state,
        horizon=3.0,
        model="bdt",
        n_steps=80,
    )

    assert resolved.model_name == "bdt"
    assert resolved.sigma == pytest.approx(0.20)
    assert resolved.n_steps == 80
    assert market_state.vol_surface.requests == [
        (pytest.approx(1.5), pytest.approx(0.04))
    ]


def test_resolve_short_rate_lattice_inputs_prefers_model_parameter_payload() -> None:
    market_state = _market_state(
        volatility=None,
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 0.03,
            "sigma": 0.012,
        },
    )

    resolved = resolve_short_rate_lattice_inputs(
        market_state,
        horizon=2.0,
        sigma=None,
        mean_reversion=None,
    )

    assert resolved.mean_reversion == pytest.approx(0.03)
    assert resolved.sigma == pytest.approx(0.012)


def test_resolve_short_rate_lattice_inputs_accepts_explicit_sigma_without_surface() -> None:
    resolved = resolve_short_rate_lattice_inputs(
        _market_state(volatility=None),
        horizon=2.0,
        mean_reversion=0.07,
        sigma=0.009,
    )

    assert resolved.mean_reversion == pytest.approx(0.07)
    assert resolved.sigma == pytest.approx(0.009)


def test_resolve_short_rate_lattice_inputs_applies_route_discretization_policy() -> None:
    resolved = resolve_short_rate_lattice_inputs(
        _market_state(),
        horizon=6.0,
        minimum_steps=100,
        maximum_steps=400,
        steps_per_year=20.0,
    )

    assert resolved.n_steps == 120


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0.0}, "horizon must be positive"),
        ({"horizon": 1.0, "model": "not_a_model"}, "Unsupported short-rate lattice model"),
        ({"horizon": 1.0, "n_steps": 0}, "n_steps must be positive"),
        ({"horizon": 1.0, "n_steps": 1.5}, "n_steps must be a positive integer"),
        (
            {"horizon": 1.0, "minimum_steps": 20, "maximum_steps": 10},
            "maximum_steps must be at least minimum_steps",
        ),
    ],
)
def test_resolve_short_rate_lattice_inputs_rejects_invalid_contracts(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_short_rate_lattice_inputs(_market_state(), **kwargs)


def test_resolve_short_rate_lattice_inputs_requires_discount_curve() -> None:
    with pytest.raises(ValueError, match="requires market_state.discount"):
        resolve_short_rate_lattice_inputs(
            SimpleNamespace(discount=None, vol_surface=_VolSurface()),
            horizon=1.0,
        )


def test_resolve_short_rate_lattice_inputs_requires_volatility_evidence() -> None:
    with pytest.raises(ValueError, match="sigma must be provided"):
        resolve_short_rate_lattice_inputs(
            _market_state(volatility=None),
            horizon=1.0,
        )
