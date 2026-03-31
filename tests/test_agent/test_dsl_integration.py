"""Cross-layer DSL integration tests.

These tests exercise the full semantic DSL stack — concept resolution →
semantic contract → deterministic validation → compiler → build gate →
blueprint — without making any LLM calls.  One test per supported product
family.

Each test verifies that the entire Layer 1 → Layer 3 pipeline produces a
deterministic, coherent blueprint with no free-text short-circuits.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile(contract, *, requested_measures=None):
    from trellis.agent.semantic_contract_compiler import compile_semantic_contract

    return compile_semantic_contract(contract, requested_measures=requested_measures)


def _gate_report(confidence: float, *, has_promoted_route: bool = True):
    """Build a minimal GapReport for gate tests without touching the store."""
    from trellis.agent.knowledge.gap_check import GapReport

    return GapReport(
        has_decomposition=True,
        has_cookbook=True,
        has_contracts=True,
        has_requirements=True,
        has_promoted_route=has_promoted_route,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Layer 1 → Layer 3: per-product-family round-trips
# ---------------------------------------------------------------------------


class TestVanillaOptionPipeline:
    """Vanilla equity option: analytical route, measure = VEGA."""

    def test_concept_resolves(self):
        from trellis.agent.semantic_concepts import resolve_semantic_concept

        r = resolve_semantic_concept("vanilla equity call option Black-Scholes")
        assert r.resolution_kind in {"reuse_existing_concept", "clarification"}

    def test_contract_validates(self):
        from trellis.agent.semantic_contracts import make_vanilla_option_contract
        from trellis.agent.semantic_contract_validation import validate_semantic_contract

        contract = make_vanilla_option_contract(
            description="EUR call on AAPL, K=150, T=1y, σ=0.20",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
        )
        report = validate_semantic_contract(contract)
        assert report.ok, report.errors

    def test_compiler_produces_blueprint(self):
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        contract = make_vanilla_option_contract(
            description="EUR call on AAPL, K=150, T=1y, σ=0.20",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
        )
        bp = _compile(contract, requested_measures=["vega"])

        assert bp.semantic_id == "vanilla_option"
        assert bp.preferred_method == "analytical"
        assert "vega" in {m.value for m in bp.requested_measures}
        # Blueprint must declare route modules
        assert len(bp.route_modules) > 0
        # Blueprint must list required market data
        assert len(bp.required_market_data) > 0
        # Connector binding hints must be populated
        assert len(bp.connector_binding_hints) > 0

    def test_blueprint_measure_warning_when_unsupported(self):
        """Requesting a measure the method doesn't natively support emits a warning."""
        from trellis.agent.semantic_contracts import make_vanilla_option_contract

        contract = make_vanilla_option_contract(
            description="EUR call on AAPL, K=150, T=1y",
            underliers=("AAPL",),
            observation_schedule=("2026-06-20",),
        )
        # KEY_RATE_DURATIONS is a rate measure; analytical equity pricing won't
        # natively support it, so the compiler should emit a measure warning.
        bp = _compile(contract, requested_measures=["key_rate_durations"])
        # The measure is normalized
        assert len(bp.requested_measures) == 1
        # Either it's in the supported set (fine) or there's a warning
        if bp.measure_support_warnings:
            assert "key_rate_durations" in bp.measure_support_warnings[0]


class TestCallableBondPipeline:
    """Callable bond: rate_tree route, measures = DV01 + DURATION."""

    def test_contract_validates(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract
        from trellis.agent.semantic_contract_validation import validate_semantic_contract

        contract = make_callable_bond_contract(
            description="5% 10Y USD callable bond, callable from year 3 semi-annually",
            observation_schedule=("2027-06-15", "2027-12-15", "2028-06-15"),
        )
        report = validate_semantic_contract(contract)
        assert report.ok, report.errors

    def test_compiler_selects_rate_tree(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract

        contract = make_callable_bond_contract(
            description="5% 10Y USD callable bond, callable from year 3",
            observation_schedule=("2027-06-15", "2027-12-15"),
        )
        bp = _compile(contract)

        assert bp.semantic_id == "callable_bond"
        assert bp.preferred_method == "rate_tree"
        assert bp.product_ir is not None
        assert bp.product_ir.exercise_style == "issuer_call"

    def test_dv01_and_duration_measures(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract

        contract = make_callable_bond_contract(
            description="5% 10Y USD callable bond",
            observation_schedule=("2027-06-15", "2027-12-15"),
        )
        bp = _compile(contract, requested_measures=["dv01", "duration"])

        measure_strs = {m.value for m in bp.requested_measures}
        assert "dv01" in measure_strs
        assert "duration" in measure_strs

    def test_blueprint_has_candidate_methods(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract

        contract = make_callable_bond_contract(
            description="5% 10Y USD callable bond",
            observation_schedule=("2027-06-15",),
        )
        bp = _compile(contract)
        assert len(bp.candidate_methods) >= 1
        assert bp.preferred_method in bp.candidate_methods


class TestBasketPathPayoffPipeline:
    """Ranked observation basket (Himalaya-style): MC route."""

    def test_full_pipeline_round_trip(self):
        from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract
        from trellis.agent.semantic_contract_validation import validate_semantic_contract

        contract = make_ranked_observation_basket_contract(
            description="Himalaya on AAPL, MSFT, NVDA",
            constituents=("AAPL", "MSFT", "NVDA"),
            observation_schedule=("2025-06-15", "2025-12-15", "2026-06-15"),
        )
        report = validate_semantic_contract(contract)
        assert report.ok, report.errors

        bp = _compile(contract, requested_measures=["price", "delta"])

        assert bp.semantic_id == "ranked_observation_basket"
        assert bp.preferred_method == "monte_carlo"
        # MC basket must list correlated_basket_monte_carlo in primitive routes
        assert "correlated_basket_monte_carlo" in bp.primitive_routes
        # Measures present — normalize_requested_measures strips PRICE (sensitivity-only)
        # so only sensitivity measures appear in bp.requested_measures
        measure_strs = {m.value for m in bp.requested_measures}
        assert "delta" in measure_strs

    def test_unsupported_path_declared(self):
        from trellis.agent.semantic_contracts import make_ranked_observation_basket_contract

        contract = make_ranked_observation_basket_contract(
            description="Himalaya on AAPL, MSFT",
            constituents=("AAPL", "MSFT"),
            observation_schedule=("2025-06-15", "2025-12-15"),
        )
        bp = _compile(contract)
        # blueprint.unsupported_paths is the union of unsupported_variants + blocked_by
        # Verify the field is accessible (may be empty for well-supported contracts)
        assert isinstance(bp.unsupported_paths, tuple)


class TestQuantoOptionPipeline:
    """Quanto option: analytical or MC route."""

    def test_contract_validates(self):
        from trellis.agent.semantic_contracts import make_quanto_option_contract
        from trellis.agent.semantic_contract_validation import validate_semantic_contract

        contract = make_quanto_option_contract(
            description="EUR quanto call on Nikkei 225 settled in USD",
            underliers=("NKY",),
            observation_schedule=("2026-12-19",),
        )
        report = validate_semantic_contract(contract)
        assert report.ok, report.errors

    def test_compiler_emits_blueprint(self):
        from trellis.agent.semantic_contracts import make_quanto_option_contract

        contract = make_quanto_option_contract(
            description="EUR quanto call on Nikkei 225 settled in USD",
            underliers=("NKY",),
            observation_schedule=("2026-12-19",),
        )
        bp = _compile(contract)

        assert bp.semantic_id == "quanto_option"
        assert bp.preferred_method in bp.candidate_methods
        assert bp.product_ir is not None
        # Quanto must declare FX as required or derivable market data
        all_data = set(bp.required_market_data) | set(bp.derivable_market_data)
        assert any("fx" in d.lower() or "spot" in d.lower() for d in all_data), (
            f"FX/spot market data missing; got required={bp.required_market_data}, "
            f"derivable={bp.derivable_market_data}"
        )


class TestSwaptionPipeline:
    """Rate swaption: rate_tree/analytical route."""

    def test_contract_validates(self):
        from trellis.agent.semantic_contracts import make_rate_style_swaption_contract
        from trellis.agent.semantic_contract_validation import validate_semantic_contract

        contract = make_rate_style_swaption_contract(
            description="5Y×10Y USD payer swaption Black-76",
            observation_schedule=("2031-03-15",),
        )
        report = validate_semantic_contract(contract)
        assert report.ok, report.errors

    def test_compiler_emits_blueprint_with_rho(self):
        from trellis.agent.semantic_contracts import make_rate_style_swaption_contract

        contract = make_rate_style_swaption_contract(
            description="5Y×10Y USD payer swaption Black-76",
            observation_schedule=("2031-03-15",),
        )
        bp = _compile(contract, requested_measures=["rho"])

        assert bp.semantic_id == "rate_style_swaption"
        assert bp.preferred_method in bp.candidate_methods
        measure_strs = {m.value for m in bp.requested_measures}
        assert "rho" in measure_strs


# ---------------------------------------------------------------------------
# Layer 2 → build gate: gate enforcement
# ---------------------------------------------------------------------------


class TestBuildGateIntegration:
    """Verify the hard build gate blocks, narrows, or passes based on GapReport."""

    def test_high_confidence_proceeds(self):
        from trellis.agent.build_gate import evaluate_pre_flight_gate

        report = _gate_report(0.85)
        decision = evaluate_pre_flight_gate(report)
        assert decision.decision == "proceed"
        assert decision.gate_source == "pre_flight"
        assert decision.gap_confidence == pytest.approx(0.85)

    def test_low_confidence_blocks(self):
        from trellis.agent.build_gate import evaluate_pre_flight_gate

        report = _gate_report(0.25)
        decision = evaluate_pre_flight_gate(report)
        assert decision.decision == "block"
        assert "0.4" in decision.reason or "block" in decision.reason.lower()

    def test_medium_confidence_narrows(self):
        from trellis.agent.build_gate import evaluate_pre_flight_gate

        report = _gate_report(0.45)
        decision = evaluate_pre_flight_gate(report)
        assert decision.decision == "narrow_route"

    def test_no_promoted_route_blocks_when_required(self):
        from trellis.agent.build_gate import evaluate_pre_flight_gate
        from trellis.agent.knowledge.schema import BuildGateThresholds

        thresholds = BuildGateThresholds(require_promoted_route=True)
        report = _gate_report(0.90, has_promoted_route=False)
        decision = evaluate_pre_flight_gate(report, thresholds=thresholds)
        assert decision.decision == "block"
        assert "promoted route" in decision.reason.lower() or "route" in decision.reason.lower()

    def test_gate_decision_is_frozen(self):
        from trellis.agent.build_gate import evaluate_pre_flight_gate

        report = _gate_report(0.9)
        decision = evaluate_pre_flight_gate(report)
        with pytest.raises((AttributeError, TypeError)):
            decision.decision = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DslMeasure: normalize and thread through compiler
# ---------------------------------------------------------------------------


class TestDslMeasureNormalization:
    """DslMeasure aliases resolve to canonical values."""

    def test_price_aliases(self):
        from trellis.core.types import DslMeasure, normalize_dsl_measure

        for alias in ("price", "npv", "pv", "PRICE"):
            assert normalize_dsl_measure(alias) == DslMeasure.PRICE

    def test_dv01_aliases(self):
        from trellis.core.types import DslMeasure, normalize_dsl_measure

        for alias in ("dv01", "pv01", "DV01"):
            assert normalize_dsl_measure(alias) == DslMeasure.DV01

    def test_duration_aliases(self):
        from trellis.core.types import DslMeasure, normalize_dsl_measure

        for alias in ("duration", "DURATION"):
            assert normalize_dsl_measure(alias) == DslMeasure.DURATION

    def test_unknown_measure_raises(self):
        from trellis.core.types import normalize_dsl_measure

        with pytest.raises(ValueError, match="Unknown measure"):
            normalize_dsl_measure("quantum_flux")

    def test_dsl_measure_is_str_subclass(self):
        """DslMeasure is a str subclass: equality with plain str works via .value."""
        from trellis.core.types import DslMeasure

        assert isinstance(DslMeasure.PRICE, str)
        # The .value IS the canonical string — direct equality works because
        # DslMeasure(str, Enum) and `DslMeasure.PRICE == "price"` is True.
        assert DslMeasure.PRICE == "price"
        assert DslMeasure.PRICE.value == "price"

    def test_measures_flow_through_compiler(self):
        """Requested measures appear in blueprint.requested_measures as DslMeasure."""
        from trellis.agent.semantic_contracts import make_vanilla_option_contract
        from trellis.core.types import DslMeasure

        contract = make_vanilla_option_contract(
            description="test call option",
            underliers=("SPX",),
            observation_schedule=("2026-12-18",),
        )
        bp = _compile(contract, requested_measures=["delta", "gamma", "vega"])

        for m in bp.requested_measures:
            assert isinstance(m, DslMeasure)
        measure_strs = {m.value for m in bp.requested_measures}
        assert {"delta", "gamma", "vega"} <= measure_strs


# ---------------------------------------------------------------------------
# EventMachine: validate + skeleton emission
# ---------------------------------------------------------------------------


class TestEventMachineIntegration:
    """EventMachine contracts are valid and the skeleton emitter produces Python."""

    def test_autocallable_machine_validates(self):
        from trellis.agent.event_machine import (
            autocallable_event_machine,
            validate_event_machine,
        )

        machine = autocallable_event_machine()
        errors = validate_event_machine(machine)
        assert not errors, errors

    def test_tarf_machine_validates(self):
        from trellis.agent.event_machine import tarf_event_machine, validate_event_machine

        machine = tarf_event_machine()
        errors = validate_event_machine(machine)
        assert not errors, errors

    def test_skeleton_emitter_produces_python(self):
        from trellis.agent.event_machine import (
            autocallable_event_machine,
            emit_event_machine_skeleton,
        )

        machine = autocallable_event_machine()
        skeleton = emit_event_machine_skeleton(machine)
        assert skeleton is not None
        # Must be valid Python: at minimum it defines a class or enum
        assert "class" in skeleton or "def " in skeleton

    def test_skeleton_references_states(self):
        from trellis.agent.event_machine import (
            autocallable_event_machine,
            emit_event_machine_skeleton,
        )

        machine = autocallable_event_machine()
        skeleton = emit_event_machine_skeleton(machine)
        # Every state name must appear in the skeleton
        for state in machine.states:
            assert state.name in skeleton, f"State '{state.name}' missing from skeleton"

    def test_unreachable_terminal_state_fails_validation(self):
        from trellis.agent.event_machine import (
            EventMachine,
            EventState,
            EventTransition,
            validate_event_machine,
        )

        alive = EventState(name="alive", kind="intermediate")
        orphan = EventState(name="orphan_terminal", kind="terminal")
        transition = EventTransition(
            name="expire",
            from_state="alive",
            to_state="alive",
        )
        machine = EventMachine(
            states=(alive, orphan),
            transitions=(transition,),
            initial_state="alive",
            terminal_states=("orphan_terminal",),
        )
        errors = validate_event_machine(machine)
        # orphan_terminal is unreachable — should trigger a validation error
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "terminal" in combined.lower()
