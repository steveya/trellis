from __future__ import annotations

import pytest

from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    LATTICE_MODEL_REGISTRY,
    LOG_SPOT_MESH,
    build_lattice,
    price_on_lattice,
)
from trellis.models.trees.algebra import (
    LatticeContractSpec,
    LatticeLinearClaimSpec,
)


def _sum_of_calls_contract(k1: float, k2: float) -> LatticeContractSpec:
    claim = LatticeLinearClaimSpec(
        terminal_payoff=lambda step, node, lattice, obs: max(float(obs["spot_1"]) - k1, 0.0)
        + max(float(obs["spot_2"]) - k2, 0.0)
    )
    return LatticeContractSpec(claim=claim)


def _basket_call_contract(strike: float, w1: float = 0.5, w2: float = 0.5) -> LatticeContractSpec:
    claim = LatticeLinearClaimSpec(
        terminal_payoff=lambda step, node, lattice, obs: max(
            w1 * float(obs["spot_1"]) + w2 * float(obs["spot_2"]) - strike,
            0.0,
        )
    )
    return LatticeContractSpec(claim=claim)


def _one_factor_call_price(*, spot: float, sigma: float, strike: float, n_steps: int = 120) -> float:
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY["crr"],
        calibration_target=None,
        spot=spot,
        rate=0.03,
        sigma=sigma,
        maturity=1.0,
        n_steps=n_steps,
    )
    contract = LatticeContractSpec(
        claim=LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: max(float(obs["spot"]) - strike, 0.0)
        )
    )
    return price_on_lattice(lattice, contract)


def test_two_factor_uncorrelated_lattice_separates_into_sum_of_1d_prices():
    lattice = build_lattice(
        topology=LATTICE_MODEL_REGISTRY["correlated_gbm_2f"].supported_topologies[0],  # type: ignore[index]
        mesh="product_log_spot_2f",  # type: ignore[arg-type]
        model=LATTICE_MODEL_REGISTRY["correlated_gbm_2f"],
        calibration_target=None,
        spots=(100.0, 95.0),
        rate=0.03,
        sigmas=(0.20, 0.25),
        maturity=1.0,
        n_steps=120,
        correlation=0.0,
    )
    price_2d = price_on_lattice(lattice, _sum_of_calls_contract(100.0, 95.0))
    price_1d = _one_factor_call_price(spot=100.0, sigma=0.20, strike=100.0) + _one_factor_call_price(
        spot=95.0,
        sigma=0.25,
        strike=95.0,
    )

    assert price_2d == pytest.approx(price_1d, rel=0.03)


def test_two_factor_correlation_changes_hybrid_basket_price():
    contract = _basket_call_contract(100.0)
    lattice_low = build_lattice(
        topology=LATTICE_MODEL_REGISTRY["correlated_gbm_2f"].supported_topologies[0],  # type: ignore[index]
        mesh="product_log_spot_2f",  # type: ignore[arg-type]
        model=LATTICE_MODEL_REGISTRY["correlated_gbm_2f"],
        calibration_target=None,
        spots=(100.0, 100.0),
        rate=0.03,
        sigmas=(0.20, 0.25),
        maturity=1.0,
        n_steps=100,
        correlation=0.0,
    )
    lattice_high = build_lattice(
        topology=LATTICE_MODEL_REGISTRY["correlated_gbm_2f"].supported_topologies[0],  # type: ignore[index]
        mesh="product_log_spot_2f",  # type: ignore[arg-type]
        model=LATTICE_MODEL_REGISTRY["correlated_gbm_2f"],
        calibration_target=None,
        spots=(100.0, 100.0),
        rate=0.03,
        sigmas=(0.20, 0.25),
        maturity=1.0,
        n_steps=100,
        correlation=0.6,
    )

    low_price = price_on_lattice(lattice_low, contract)
    high_price = price_on_lattice(lattice_high, contract)

    assert low_price > 0.0
    assert high_price > low_price
