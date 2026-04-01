"""Proof-of-concept tests for composition algebra (QUA-413).

Tests dataclass construction, DAG validation, method conflict resolution,
and the four worked examples from the design document.
"""

from __future__ import annotations

import pytest

from trellis.agent.composition_algebra import (
    CalibrationAcceptanceCriteria,
    CalibrationContract,
    CalibrationTarget,
    ComponentPort,
    ControlBoundary,
    CompositeSemanticContract,
    CompositionEdge,
    MethodResolution,
    PayoffComponent,
    resolve_method_conflicts,
)
from trellis.agent.dsl_algebra import ChoiceExpr, ControlStyle
from trellis.core.types import TimelineRole


# ---------------------------------------------------------------------------
# Component catalog helpers
# ---------------------------------------------------------------------------

def _barrier(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "barrier"),
        component_type="barrier",
        compatible_methods=kw.get("methods", ("monte_carlo", "pde_solver")),
        market_data_requirements=frozenset(kw.get("market_data", ("discount_curve", "black_vol_surface"))),
        description=kw.get("desc", "Knock-in/out barrier condition"),
    )

def _coupon_stream(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "coupon"),
        component_type="coupon_stream",
        compatible_methods=kw.get("methods", ("analytical", "rate_tree", "monte_carlo")),
        market_data_requirements=frozenset(kw.get("market_data", ("discount_curve",))),
    )

def _exercise_policy(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "call"),
        component_type="exercise_policy",
        compatible_methods=kw.get("methods", ("rate_tree",)),
        market_data_requirements=frozenset(kw.get("market_data", ("discount_curve", "black_vol_surface"))),
    )

def _observation_schedule(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "obs"),
        component_type="observation_schedule",
        compatible_methods=kw.get("methods", ("monte_carlo",)),
        market_data_requirements=frozenset(),
    )

def _selection_rule(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "select"),
        component_type="selection_rule",
        compatible_methods=kw.get("methods", ("monte_carlo",)),
        market_data_requirements=frozenset(),
    )

def _lock_remove(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "lock"),
        component_type="lock_remove",
        compatible_methods=kw.get("methods", ("monte_carlo",)),
        market_data_requirements=frozenset(),
    )

def _maturity_settlement(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "settle"),
        component_type="maturity_settlement",
        compatible_methods=kw.get("methods", ("monte_carlo", "analytical")),
        market_data_requirements=frozenset(kw.get("market_data", ("discount_curve",))),
    )

def _knock_condition(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "knock"),
        component_type="knock_condition",
        compatible_methods=kw.get("methods", ("monte_carlo",)),
        market_data_requirements=frozenset(kw.get("market_data", ("discount_curve", "black_vol_surface"))),
    )

def _discount_leg(**kw) -> PayoffComponent:
    return PayoffComponent(
        component_id=kw.get("id", "discount"),
        component_type="discount_leg",
        compatible_methods=kw.get("methods", ("analytical", "rate_tree", "monte_carlo")),
        market_data_requirements=frozenset(kw.get("market_data", ("discount_curve",))),
    )


# ---------------------------------------------------------------------------
# PayoffComponent construction
# ---------------------------------------------------------------------------

class TestPayoffComponent:
    def test_basic_construction(self):
        c = _barrier()
        assert c.component_type == "barrier"
        assert "monte_carlo" in c.compatible_methods

    def test_with_proven_primitive(self):
        c = PayoffComponent(
            component_id="select",
            component_type="selection_rule",
            compatible_methods=("monte_carlo",),
            proven_primitive="trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo",
        )
        assert c.proven_primitive is not None

    def test_with_ports(self):
        c = PayoffComponent(
            component_id="obs",
            component_type="observation_schedule",
            inputs=(ComponentPort("start_date", "scalar"), ComponentPort("end_date", "scalar")),
            outputs=(ComponentPort("schedule", "schedule"),),
        )
        assert len(c.inputs) == 2
        assert c.outputs[0].port_type == "schedule"

    def test_signature_uses_typed_ports_and_roles(self):
        c = PayoffComponent(
            component_id="coupon",
            component_type="coupon_stream",
            inputs=(ComponentPort("state", "state"),),
            outputs=(ComponentPort("cashflow", "array"),),
            market_data_requirements=frozenset({"discount_curve"}),
            timeline_roles=frozenset({TimelineRole.PAYMENT}),
        )
        assert c.signature.inputs == ("state:state",)
        assert c.signature.outputs == ("cashflow:array",)
        assert c.signature.timeline_roles == {TimelineRole.PAYMENT}
        assert c.signature.market_data_requirements == {"discount_curve"}

    def test_component_can_bridge_to_contract_atom(self):
        c = PayoffComponent(
            component_id="discount",
            component_type="discount_leg",
            inputs=(ComponentPort("cashflows", "array"),),
            outputs=(ComponentPort("pv", "scalar"),),
            proven_primitive="trellis.models.discount.present_value_leg",
        )
        atom = c.to_contract_atom()
        assert atom.atom_id == "discount"
        assert atom.signature.inputs == ("cashflows:array",)
        assert atom.primitive_ref == "trellis.models.discount.present_value_leg"


# ---------------------------------------------------------------------------
# Method conflict resolution
# ---------------------------------------------------------------------------

class TestMethodConflictResolution:
    def test_all_agree_on_mc(self):
        components = (
            _observation_schedule(),
            _selection_rule(),
            _lock_remove(),
            _maturity_settlement(methods=("monte_carlo",)),
        )
        result = resolve_method_conflicts(components)
        assert result.resolved_method == "monte_carlo"
        assert result.resolution_kind == "intersection"

    def test_intersection_picks_preferred(self):
        components = (
            _coupon_stream(methods=("analytical", "rate_tree", "monte_carlo")),
            _discount_leg(methods=("analytical", "rate_tree")),
        )
        result = resolve_method_conflicts(components)
        assert result.resolved_method == "analytical"
        assert result.resolution_kind == "intersection"

    def test_exercise_dominates_on_conflict(self):
        components = (
            _coupon_stream(methods=("analytical", "rate_tree")),
            _exercise_policy(methods=("rate_tree",)),
            _barrier(methods=("monte_carlo", "pde_solver")),
        )
        result = resolve_method_conflicts(components)
        assert result.resolved_method == "rate_tree"
        assert result.resolution_kind == "dominance"
        assert result.dominant_component == "call"
        assert "barrier" in result.overridden_components

    def test_unresolvable_conflict(self):
        components = (
            _barrier(methods=("monte_carlo",)),
            _knock_condition(methods=("pde_solver",)),
        )
        result = resolve_method_conflicts(components)
        assert result.resolution_kind == "conflict"


# ---------------------------------------------------------------------------
# DAG validation
# ---------------------------------------------------------------------------

class TestDAGValidation:
    def test_valid_linear_chain(self):
        composite = CompositeSemanticContract(
            composite_id="test",
            description="test",
            components=(_observation_schedule(), _selection_rule(), _lock_remove()),
            edges=(
                CompositionEdge("obs", "select", "sequential"),
                CompositionEdge("select", "lock", "sequential"),
            ),
        )
        errors = composite.validate_dag()
        assert errors == ()

    def test_detects_missing_edge_endpoint(self):
        composite = CompositeSemanticContract(
            composite_id="test",
            description="test",
            components=(_observation_schedule(),),
            edges=(CompositionEdge("obs", "nonexistent", "sequential"),),
        )
        errors = composite.validate_dag()
        assert any("nonexistent" in e for e in errors)

    def test_detects_cycle(self):
        composite = CompositeSemanticContract(
            composite_id="test",
            description="test",
            components=(_observation_schedule(id="a"), _selection_rule(id="b"), _lock_remove(id="c")),
            edges=(
                CompositionEdge("a", "b", "sequential"),
                CompositionEdge("b", "c", "sequential"),
                CompositionEdge("c", "a", "sequential"),
            ),
        )
        errors = composite.validate_dag()
        assert any("Cycle" in e for e in errors)

    def test_detects_disconnected(self):
        composite = CompositeSemanticContract(
            composite_id="test",
            description="test",
            components=(
                _observation_schedule(id="a"),
                _selection_rule(id="b"),
                _lock_remove(id="c"),  # disconnected
            ),
            edges=(CompositionEdge("a", "b", "sequential"),),
        )
        errors = composite.validate_dag()
        assert any("Disconnected" in e for e in errors)

    def test_detects_sequential_signature_mismatch(self):
        composite = CompositeSemanticContract(
            composite_id="typed_mismatch",
            description="test",
            components=(
                PayoffComponent(
                    component_id="coupon",
                    component_type="coupon_stream",
                    outputs=(ComponentPort("cashflow", "array"),),
                ),
                PayoffComponent(
                    component_id="discount",
                    component_type="discount_leg",
                    inputs=(ComponentPort("state", "state"),),
                ),
            ),
            edges=(CompositionEdge("coupon", "discount", "sequential"),),
        )
        errors = composite.validate_dag()
        assert any("Edge signature mismatch (sequential)" in e for e in errors)

    def test_detects_control_boundary_branch_mismatch(self):
        composite = CompositeSemanticContract(
            composite_id="control_mismatch",
            description="test",
            components=(
                _exercise_policy(id="call"),
                PayoffComponent(
                    component_id="continue",
                    component_type="continuation_value",
                    inputs=(ComponentPort("state", "state"),),
                    outputs=(ComponentPort("pv", "scalar"),),
                ),
                PayoffComponent(
                    component_id="redeem",
                    component_type="redeem_now",
                    inputs=(ComponentPort("state", "state"),),
                    outputs=(ComponentPort("cashflow", "array"),),
                ),
            ),
            edges=(),
            control_boundaries=(
                ControlBoundary(
                    boundary_id="issuer_call",
                    controller_component="call",
                    style=ControlStyle.ISSUER_MIN,
                    branches=("continue", "redeem"),
                ),
            ),
        )
        errors = composite.validate_dag()
        assert any("Control boundary branch mismatch" in e for e in errors)

    def test_detects_non_exercise_controller(self):
        composite = CompositeSemanticContract(
            composite_id="bad_controller",
            description="test",
            components=(
                _coupon_stream(id="coupon"),
                PayoffComponent(
                    component_id="continue",
                    component_type="continuation_value",
                    inputs=(ComponentPort("state", "state"),),
                    outputs=(ComponentPort("pv", "scalar"),),
                ),
                PayoffComponent(
                    component_id="redeem",
                    component_type="redeem_now",
                    inputs=(ComponentPort("state", "state"),),
                    outputs=(ComponentPort("pv", "scalar"),),
                ),
            ),
            edges=(),
            control_boundaries=(
                ControlBoundary(
                    boundary_id="issuer_call",
                    controller_component="coupon",
                    style=ControlStyle.ISSUER_MIN,
                    branches=("continue", "redeem"),
                ),
            ),
        )
        errors = composite.validate_dag()
        assert any("controller must be an exercise_policy" in e for e in errors)


# ---------------------------------------------------------------------------
# Worked Example A: Callable Range Accrual
# ---------------------------------------------------------------------------

class TestCallableRangeAccrual:
    @pytest.fixture
    def composite(self):
        return CompositeSemanticContract(
            composite_id="callable_range_accrual",
            description="Callable range accrual: coupon accrues in range, issuer call overlay",
            components=(
                _coupon_stream(id="coupon", methods=("analytical", "rate_tree", "monte_carlo")),
                _knock_condition(id="range", methods=("monte_carlo",)),
                _exercise_policy(id="call", methods=("rate_tree",)),
                _discount_leg(id="discount"),
            ),
            edges=(
                CompositionEdge("coupon", "range", "conditional", condition="coupon accrues only in range"),
                CompositionEdge("range", "call", "parallel"),
                CompositionEdge("call", "discount", "sequential"),
            ),
            control_boundaries=(
                ControlBoundary(
                    boundary_id="issuer_call",
                    controller_component="call",
                    style=ControlStyle.ISSUER_MIN,
                    branches=("range", "discount"),
                    label="issuer_call",
                ),
            ),
        )

    def test_dag_valid(self, composite):
        assert composite.validate_dag() == ()

    def test_method_conflict_resolved_by_dominance(self, composite):
        resolution = composite.compute_method_resolution()
        assert resolution.resolution_kind == "dominance"
        assert resolution.resolved_method == "rate_tree"
        assert resolution.dominant_component == "call"

    def test_market_data_union(self, composite):
        union = composite.compute_market_data_union()
        assert "discount_curve" in union
        assert "black_vol_surface" in union

    def test_control_boundary_lowers_to_choice_expr(self, composite):
        expr = composite.control_boundary_expr("issuer_call")
        assert isinstance(expr, ChoiceExpr)
        assert expr.style == ControlStyle.ISSUER_MIN
        assert expr.label == "issuer_call"
        assert {branch.atom_id for branch in expr.branches} == {"range", "discount"}

    def test_collects_control_styles(self, composite):
        assert composite.collect_control_styles() == (ControlStyle.ISSUER_MIN,)


# ---------------------------------------------------------------------------
# Worked Example B: Ranked-Observation Basket (QUA-284)
# ---------------------------------------------------------------------------

class TestRankedObservationBasket:
    @pytest.fixture
    def composite(self):
        return CompositeSemanticContract(
            composite_id="ranked_observation_basket",
            description="Himalaya-style basket: observe, select best, lock, settle average",
            components=(
                _observation_schedule(id="obs"),
                PayoffComponent(
                    component_id="select",
                    component_type="selection_rule",
                    compatible_methods=("monte_carlo",),
                    proven_primitive="trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo",
                ),
                _lock_remove(id="lock"),
                _maturity_settlement(id="settle", methods=("monte_carlo",)),
            ),
            edges=(
                CompositionEdge("obs", "select", "sequential"),
                CompositionEdge("select", "lock", "sequential"),
                CompositionEdge("lock", "settle", "sequential"),
            ),
        )

    def test_dag_valid(self, composite):
        assert composite.validate_dag() == ()

    def test_no_method_conflict(self, composite):
        resolution = composite.compute_method_resolution()
        assert resolution.resolution_kind == "intersection"
        assert resolution.resolved_method == "monte_carlo"

    def test_has_proven_component(self, composite):
        proven = composite.proven_components()
        assert len(proven) == 1
        assert proven[0].component_id == "select"

    def test_generation_required_components(self, composite):
        gen = composite.generation_required_components()
        gen_ids = {c.component_id for c in gen}
        assert "obs" in gen_ids
        assert "lock" in gen_ids
        assert "settle" in gen_ids
        assert "select" not in gen_ids  # proven


# ---------------------------------------------------------------------------
# Worked Example C: Barrier on Callable Bond (Method Conflict)
# ---------------------------------------------------------------------------

class TestBarrierCallableBond:
    @pytest.fixture
    def composite(self):
        return CompositeSemanticContract(
            composite_id="barrier_callable_bond",
            description="Barrier option on callable bond: barrier (MC) vs call (tree)",
            components=(
                _coupon_stream(id="bond", methods=("analytical", "rate_tree")),
                _exercise_policy(id="call", methods=("rate_tree",)),
                _barrier(id="barrier", methods=("monte_carlo", "pde_solver")),
                _discount_leg(id="discount"),
            ),
            edges=(
                CompositionEdge("bond", "call", "parallel"),
                CompositionEdge("call", "barrier", "conditional"),
                CompositionEdge("barrier", "discount", "sequential"),
            ),
        )

    def test_dag_valid(self, composite):
        assert composite.validate_dag() == ()

    def test_exercise_dominates(self, composite):
        resolution = composite.compute_method_resolution()
        assert resolution.resolved_method == "rate_tree"
        assert resolution.resolution_kind == "dominance"
        assert "barrier" in resolution.overridden_components


# ---------------------------------------------------------------------------
# Worked Example D: Hull-White Calibration
# ---------------------------------------------------------------------------

class TestHullWhiteCalibration:
    def test_calibration_contract_construction(self):
        cal = CalibrationContract(
            calibration_id="hw_callable_bond",
            target=CalibrationTarget(
                parameter="hw_mean_reversion",
                output_capability="hw_short_rate_params",
            ),
            fitting_instruments=("atm_swaptions_1y_10y",),
            optimizer="analytical",
            acceptance_criteria=CalibrationAcceptanceCriteria(
                max_fitting_error_bps=5.0,
            ),
            output_binding="hw_short_rate_params",
            proven_primitive="trellis.models.calibration.rates.calibrate_hull_white",
        )
        assert cal.target.output_capability == "hw_short_rate_params"
        assert cal.proven_primitive is not None
        assert cal.acceptance_criteria.max_fitting_error_bps == 5.0

    def test_calibration_without_proven_primitive(self):
        cal = CalibrationContract(
            calibration_id="sabr_cap_vol",
            target=CalibrationTarget(
                parameter="sabr_params",
                output_capability="sabr_vol_surface",
            ),
            fitting_instruments=("cap_vols_6m_30y",),
            optimizer="least_squares",
            output_binding="sabr_vol_surface",
        )
        assert cal.proven_primitive is None
        assert cal.optimizer == "least_squares"
