"""Tests for Heston ADI-style PDE binding and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.pde.heston_adi import (
    HestonAdiPDEConfig,
    price_heston_option_adi_pde_result,
    resolve_heston_adi_pde_inputs,
)
from trellis.models.processes.heston import build_heston_parameter_payload
from trellis.models.transforms.heston import price_heston_option_transform_result


@dataclass(frozen=True)
class HestonOptionSpec:
    notional: float = 1.0
    spot: float = 100.0
    strike: float = 100.0
    expiry_date: date = date(2025, 11, 15)
    option_type: str = "call"


def _market_state() -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.03),
        spot=100.0,
        model_parameters=build_heston_parameter_payload(
            kappa=1.8,
            theta=0.04,
            xi=0.25,
            rho=-0.35,
            v0=0.04,
        ),
    )


def _legacy_alias_market_state() -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.03),
        spot=100.0,
        model_parameters={
            "model_family": "heston",
            "kappa": 1.8,
            "theta_var": 0.04,
            "sigma_v": 0.25,
            "rho": -0.35,
            "initial_variance": 0.04,
        },
    )


def _high_vol_of_vol_market_state() -> MarketState:
    return MarketState(
        as_of=date(2024, 11, 15),
        settlement=date(2024, 11, 15),
        discount=YieldCurve.flat(0.04156507836505682),
        spot=100.0,
        model_parameters=build_heston_parameter_payload(
            kappa=2.0,
            theta=0.039117,
            xi=0.469406,
            rho=-0.68,
            v0=0.039117,
        ),
    )


def test_heston_adi_inputs_resolve_runtime_binding_with_canonical_parameters():
    resolved = resolve_heston_adi_pde_inputs(
        _legacy_alias_market_state(),
        HestonOptionSpec(),
    )

    assert resolved.validation_bundle == "heston:adi_pde"
    assert resolved.runtime_binding.provenance["source_ref"] == (
        "resolve_heston_runtime_binding"
    )
    assert resolved.runtime_binding.model_parameters["theta"] == pytest.approx(0.04)
    assert resolved.runtime_binding.model_parameters["xi"] == pytest.approx(0.25)
    assert resolved.runtime_binding.model_parameters["v0"] == pytest.approx(0.04)
    assert "theta_var" not in resolved.runtime_binding.model_parameters
    assert "sigma_v" not in resolved.runtime_binding.model_parameters
    assert "initial_variance" not in resolved.runtime_binding.model_parameters


def test_heston_adi_pde_consumes_runtime_binding_without_vol_surface_recalibration():
    spec = HestonOptionSpec()
    config = HestonAdiPDEConfig(spot_steps=64, variance_steps=28, time_steps=80)

    pde = price_heston_option_adi_pde_result(_market_state(), spec, config=config).price

    assert pde > 0.0


def test_heston_adi_variance_grid_bound_handles_high_vol_of_vol_regime():
    market = _high_vol_of_vol_market_state()
    spec = HestonOptionSpec()
    config = HestonAdiPDEConfig(spot_steps=64, variance_steps=28, time_steps=80)

    pde = price_heston_option_adi_pde_result(market, spec, config=config).price
    reference = price_heston_option_transform_result(market, spec, method="fft").price

    assert abs(pde - reference) / reference < 0.05


def test_heston_adi_pde_result_exposes_canonical_model_parameters():
    result = price_heston_option_adi_pde_result(
        _market_state(),
        HestonOptionSpec(),
        config=HestonAdiPDEConfig(
            spot_steps=32,
            variance_steps=18,
            time_steps=40,
            reference_method="fft",
        ),
    )

    assert result.validation_bundle == "heston:adi_pde"
    assert result.model_parameters["theta"] == pytest.approx(0.04)
    assert result.model_parameters["xi"] == pytest.approx(0.25)
    assert result.grid_shape == (32, 18)
    assert result.raw_adi_price > 0.0
    assert result.reference_price is not None
    assert result.reference_method == "fft"
    assert result.reference_relative_error is not None


def test_heston_adi_reference_diagnostic_does_not_replace_scalar_price():
    market = _market_state()
    spec = HestonOptionSpec()
    result = price_heston_option_adi_pde_result(
        market,
        spec,
        config=HestonAdiPDEConfig(
            spot_steps=24,
            variance_steps=12,
            time_steps=20,
            reference_method="cos",
        ),
    )

    assert result.reference_price is not None
    assert result.price == pytest.approx(result.raw_adi_price)
