"""Proof-obligation tests for the generalized lattice algebra."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as raw_np
import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.callable_bond_tree import price_callable_bond_tree, straight_bond_present_value
from trellis.models.equity_option_pde import price_vanilla_equity_option_pde
from trellis.models.equity_option_tree import price_vanilla_equity_option_tree
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    LOG_SPOT_MESH,
    NO_CALIBRATION_TARGET,
    TERM_STRUCTURE_TARGET,
    UNIFORM_ADDITIVE_MESH,
    LATTICE_MODEL_REGISTRY,
)
from trellis.models.trees.lattice import build_lattice, price_on_lattice
from trellis.models.trees.models import MODEL_REGISTRY
from trellis.models.vol_surface import FlatVol


@dataclass(frozen=True)
class _VanillaEquitySpec:
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    exercise_style: str = "european"


def _equity_market_state() -> MarketState:
    settle = date(2024, 11, 15)
    return MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )


def _short_rate_lattice(model_name: str):
    return build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        MODEL_REGISTRY[model_name].as_lattice_model_spec(),
        calibration_target=TERM_STRUCTURE_TARGET(YieldCurve.flat(0.05)),
        r0=0.05,
        sigma=0.01,
        a=0.1,
        T=2.0,
        n_steps=40,
    )


def _equity_lattice(model_name: str):
    return build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY[model_name],
        calibration_target=NO_CALIBRATION_TARGET(),
        spot=100.0,
        rate=0.05,
        sigma=0.20,
        maturity=1.0,
        n_steps=200,
    )


@pytest.mark.parametrize("model_name", ["hull_white", "bdt", "black_karasinski", "ho_lee"])
def test_operator_probabilities_are_positive_and_row_stochastic_for_short_rate_models(model_name: str):
    lattice = _short_rate_lattice(model_name)

    for step in range(lattice.n_steps):
        probs = lattice._probs[step, : lattice.n_nodes(step), : lattice.branching]
        assert raw_np.all(probs >= 0.0)
        assert raw_np.allclose(raw_np.sum(probs, axis=1), 1.0)


@pytest.mark.parametrize("model_name", ["crr", "jarrow_rudd"])
def test_operator_probabilities_are_positive_and_row_stochastic_for_equity_models(model_name: str):
    lattice = _equity_lattice(model_name)

    for step in range(lattice.n_steps):
        probs = lattice._probs[step, : lattice.n_nodes(step), : lattice.branching]
        assert raw_np.all(probs >= 0.0)
        assert raw_np.allclose(raw_np.sum(probs, axis=1), 1.0)


def test_term_structure_calibration_round_trip_is_exact_to_tolerance():
    curve = YieldCurve.flat(0.05)
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        MODEL_REGISTRY["hull_white"].as_lattice_model_spec(),
        calibration_target=TERM_STRUCTURE_TARGET(curve),
        r0=0.05,
        sigma=0.01,
        a=0.1,
        T=3.0,
        n_steps=60,
    )

    residuals = lattice._lattice_calibration_diagnostics.residuals

    assert residuals
    assert max(float(value) for value in residuals.values()) < 1e-10


@pytest.mark.parametrize(
    ("model_name", "builder"),
    [
        ("hull_white", _short_rate_lattice),
        ("bdt", _short_rate_lattice),
        ("black_karasinski", _short_rate_lattice),
        ("ho_lee", _short_rate_lattice),
        ("crr", _equity_lattice),
        ("jarrow_rudd", _equity_lattice),
    ],
)
def test_registered_lattice_model_families_price_through_unified_surface(model_name: str, builder):
    lattice = builder(model_name)

    if model_name in {"crr", "jarrow_rudd"}:
        contract = LATTICE_MODEL_REGISTRY[model_name]
        assert contract.factor_family == "equity"
        price = price_on_lattice(
            lattice,
            __import__("trellis.models.equity_option_tree", fromlist=["compile_vanilla_equity_contract_spec"]).compile_vanilla_equity_contract_spec(
                strike=100.0,
                option_type="call",
                exercise_style="european",
            ),
        )
        assert price > 0.0
    else:
        def zero_coupon(step, node, lattice_, obs):
            del step, node, lattice_, obs
            return 1.0

        price = price_on_lattice(
            lattice,
            __import__("trellis.models.trees.algebra", fromlist=["LatticeContractSpec", "LatticeLinearClaimSpec"]).LatticeContractSpec(
                claim=__import__("trellis.models.trees.algebra", fromlist=["LatticeLinearClaimSpec"]).LatticeLinearClaimSpec(
                    terminal_payoff=zero_coupon
                )
            ),
        )
        assert price > 0.0


def test_overlay_closure_preserves_finite_price_and_reduces_barrier_value():
    from trellis.models.trees.algebra import EventOverlaySpec, LatticeContractSpec, LatticeLinearClaimSpec

    lattice = _equity_lattice("crr")
    strike = 100.0

    plain = LatticeContractSpec(
        claim=LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: max(float(obs["spot"]) - strike, 0.0),
        ),
    )
    barrier = LatticeContractSpec(
        claim=LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: (
                0.0 if obs["event_state"] == "dead" else max(float(obs["spot"]) - strike, 0.0)
            ),
        ),
        overlay=EventOverlaySpec(
            states=("alive", "dead"),
            initial_state="alive",
            transition_fn=lambda step, parent, child, event_state, lattice_, obs_parent, obs_child: (
                {"dead": 1.0}
                if event_state == "dead" or float(obs_child["spot"]) >= 140.0
                else {"alive": 1.0}
            ),
        ),
    )

    plain_price = price_on_lattice(lattice, plain)
    barrier_price = price_on_lattice(lattice, barrier)

    assert raw_np.isfinite(barrier_price)
    assert barrier_price <= plain_price + 1e-12


def test_equity_tree_agrees_with_pde_for_european_call():
    spec = _VanillaEquitySpec(
        notional=1.0,
        spot=100.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        option_type="call",
        exercise_style="european",
    )
    market_state = _equity_market_state()

    tree_price = price_vanilla_equity_option_tree(market_state, spec, model="crr", n_steps=300)
    pde_price = price_vanilla_equity_option_pde(market_state, spec, theta=0.5, n_x=401, n_t=401)

    assert tree_price == pytest.approx(pde_price, rel=0.03)


def test_callable_bond_tree_respects_straight_bond_upper_bound():
    from trellis.core.types import DayCountConvention, Frequency
    from trellis.instruments.callable_bond import CallableBondSpec

    market_state = _equity_market_state()
    spec = CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=date(2025, 1, 15),
        end_date=date(2035, 1, 15),
        call_dates=[date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15)],
        call_price=100.0,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )

    tree_price = price_callable_bond_tree(market_state, spec, model="hull_white")
    straight = straight_bond_present_value(market_state, spec, settlement=market_state.settlement)

    assert tree_price <= straight + 1e-10
