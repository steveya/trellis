"""Tests for the generalized lattice algebra surface."""

from __future__ import annotations

from math import exp, log, sqrt

import pytest
from scipy.stats import norm

from trellis.curves.yield_curve import YieldCurve
from trellis.models.trees.control import resolve_lattice_exercise_policy
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_generic_lattice,
    build_lattice,
    build_spot_lattice,
    lattice_backward_induction,
    price_on_lattice,
)
from trellis.models.trees.models import MODEL_REGISTRY


def _bs_call(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return spot * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2)


def test_tree_model_round_trips_to_lattice_model_spec():
    spec = MODEL_REGISTRY["hull_white"].as_lattice_model_spec()

    assert spec.name == "hull_white"
    assert spec.factor_family == "short_rate"
    assert spec.calibration_strategy == "term_structure"
    assert spec.state_space_type == "additive_normal"
    assert spec.supported_branchings == (2, 3)
    assert "term_structure" in spec.supported_calibration_targets


def test_build_lattice_short_rate_matches_generic_builder():
    from trellis.models.trees.algebra import (
        BINOMIAL_1F_TOPOLOGY,
        TERM_STRUCTURE_TARGET,
        UNIFORM_ADDITIVE_MESH,
    )

    curve = YieldCurve.flat(0.05)
    model = MODEL_REGISTRY["bdt"].as_lattice_model_spec()

    built = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        UNIFORM_ADDITIVE_MESH,
        model,
        calibration_target=TERM_STRUCTURE_TARGET(curve),
        r0=0.05,
        sigma=0.01,
        a=0.1,
        T=2.0,
        n_steps=40,
    )
    legacy = build_generic_lattice(
        MODEL_REGISTRY["bdt"],
        r0=0.05,
        sigma=0.01,
        a=0.1,
        T=2.0,
        n_steps=40,
        discount_curve=curve,
    )

    assert built.n_steps == legacy.n_steps
    assert built.n_nodes(40) == legacy.n_nodes(40)
    assert built.get_state(0, 0) == pytest.approx(legacy.get_state(0, 0))
    assert built.get_discount(0, 0) == pytest.approx(legacy.get_discount(0, 0))


def test_build_lattice_equity_matches_spot_builder_and_prices_call():
    from trellis.models.trees.algebra import (
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        NO_CALIBRATION_TARGET,
        LATTICE_MODEL_REGISTRY,
    )

    built = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY["crr"],
        calibration_target=NO_CALIBRATION_TARGET(),
        spot=100.0,
        rate=0.05,
        sigma=0.20,
        maturity=1.0,
        n_steps=200,
    )
    legacy = build_spot_lattice(
        100.0,
        0.05,
        0.20,
        1.0,
        200,
        model="crr",
    )

    assert built.get_state(0, 0) == pytest.approx(legacy.get_state(0, 0))

    def payoff(step, node, lattice):
        return max(float(lattice.get_state(step, node)) - 100.0, 0.0)

    assert lattice_backward_induction(built, payoff) == pytest.approx(
        _bs_call(100.0, 100.0, 0.05, 0.20, 1.0),
        rel=0.02,
    )


def test_price_on_lattice_matches_legacy_american_put():
    from trellis.models.trees.algebra import (
        LatticeContractSpec,
        LatticeControlSpec,
        LatticeLinearClaimSpec,
    )

    strike = 100.0
    lattice = build_spot_lattice(100.0, 0.05, 0.20, 1.0, 200, model="crr")

    def payoff(step, node, lattice_):
        return max(strike - float(lattice_.get_state(step, node)), 0.0)

    contract = LatticeContractSpec(
        claim=LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: max(strike - float(obs["spot"]), 0.0),
        ),
        control=LatticeControlSpec(
            objective="holder_max",
            exercise_value_fn=lambda step, node, lattice_, obs: max(strike - float(obs["spot"]), 0.0),
        ),
    )

    built = price_on_lattice(lattice, contract)
    legacy = lattice_backward_induction(
        lattice,
        payoff,
        exercise_value=payoff,
        exercise_policy=resolve_lattice_exercise_policy("american"),
    )

    assert built == pytest.approx(legacy)


def test_price_on_lattice_supports_edge_aware_knock_out_overlay():
    from trellis.models.trees.algebra import (
        EventOverlaySpec,
        LatticeContractSpec,
        LatticeLinearClaimSpec,
    )

    lattice = RecombiningLattice(1, 1.0, branching=2, state_dim=1)
    lattice.set_state(0, 0, 100.0)
    lattice.set_state(1, 0, 90.0)
    lattice.set_state(1, 1, 110.0)
    lattice.set_probabilities(0, 0, [0.5, 0.5])
    lattice.set_discount(0, 0, 1.0)

    barrier = 105.0
    contract = LatticeContractSpec(
        claim=LatticeLinearClaimSpec(
            terminal_payoff=lambda step, node, lattice_, obs: (
                0.0 if obs["event_state"] == "dead" else max(float(obs["spot"]) - 100.0, 0.0)
            ),
        ),
        overlay=EventOverlaySpec(
            states=("alive", "dead"),
            initial_state="alive",
            transition_fn=lambda step, parent, child, event_state, lattice_, obs_parent, obs_child: (
                {"dead": 1.0}
                if event_state == "dead" or float(obs_child["spot"]) >= barrier
                else {"alive": 1.0}
            ),
        ),
    )

    assert price_on_lattice(lattice, contract) == pytest.approx(0.0)


def test_compile_lattice_recipe_builds_american_equity_contract():
    from trellis.models.trees.algebra import (
        compile_lattice_recipe,
        equity_tree,
        with_control,
    )

    recipe = with_control(
        equity_tree(
            model_family="crr",
            strike=100.0,
            option_type="put",
            n_steps=64,
        ),
        control_kind="american",
    )

    topology, mesh, model, contract = compile_lattice_recipe(recipe)

    assert topology.name == "binomial_1f"
    assert mesh.name == "log_spot_1f"
    assert model.name == "crr"
    assert contract.control is not None
    assert contract.control.objective == "holder_max"
