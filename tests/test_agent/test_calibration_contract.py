"""Tests for the calibration contract DSL (QUA-438).

Covers:
- CalibrationContract, CalibrationTarget, CalibrationResult construction
- Factory functions (hull_white, sabr, black76)
- Validation (valid/invalid targets, primitives, optimizers)
- OutputBinding structured output
- Compiler integration: calibration_step on blueprint
- Callable bond contract with attached calibration
"""

from __future__ import annotations

import pytest

from trellis.agent.calibration_contract import (
    CalibrationContract,
    CalibrationMethod,
    CalibrationResult,
    CalibrationTarget,
    FittingInstrument,
    OutputBinding,
    black76_flat_vol_calibration_contract,
    hull_white_calibration_contract,
    sabr_smile_calibration_contract,
    validate_calibration_contract,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_calibration_target_frozen(self):
        t = CalibrationTarget(parameter="flat_vol", model_family="black76")
        with pytest.raises(AttributeError):
            t.parameter = "other"  # type: ignore[misc]

    def test_calibration_result_frozen_dicts(self):
        r = CalibrationResult(
            target=CalibrationTarget(parameter="flat_vol", model_family="black76"),
            calibrated_parameters={"vol": 0.2},
            residual=1e-8,
            accepted=True,
        )
        with pytest.raises(TypeError):
            r.calibrated_parameters["new_key"] = 1.0  # type: ignore[index]

    def test_output_binding_defaults(self):
        b = OutputBinding(target_path="market_state.vol_surface")
        assert b.parameter_names == ()
        assert b.consumption_pattern == "inject_parameter"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactories:
    def test_hull_white_is_valid(self):
        c = hull_white_calibration_contract()
        errors = validate_calibration_contract(c)
        assert errors == (), errors
        assert c.proven_primitive == "build_lattice"
        assert c.target.model_family == "hull_white"
        assert c.output.consumption_pattern == "build_lattice"

    def test_hull_white_cap_fitting(self):
        c = hull_white_calibration_contract(fitting="cap")
        errors = validate_calibration_contract(c)
        assert errors == (), errors
        assert c.fitting_instruments[0].instrument_type == "cap"

    def test_sabr_is_valid(self):
        c = sabr_smile_calibration_contract()
        errors = validate_calibration_contract(c)
        assert errors == (), errors
        assert c.proven_primitive == "calibrate_sabr"
        assert "alpha" in c.output.parameter_names

    def test_black76_cap_is_valid(self):
        c = black76_flat_vol_calibration_contract(fitting="cap")
        errors = validate_calibration_contract(c)
        assert errors == (), errors
        assert c.proven_primitive == "calibrate_cap_floor_black_vol"

    def test_black76_swaption_is_valid(self):
        c = black76_flat_vol_calibration_contract(fitting="swaption")
        errors = validate_calibration_contract(c)
        assert errors == (), errors
        assert c.proven_primitive == "calibrate_swaption_black_vol"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_target_parameter(self):
        c = CalibrationContract(
            target=CalibrationTarget(parameter="unknown_param", model_family="hull_white"),
            fitting_instruments=(FittingInstrument(instrument_type="swaption", data_source="market_quote"),),
            output=OutputBinding(target_path="lattice"),
        )
        errors = validate_calibration_contract(c)
        assert any("unknown_param" in e.lower() or "Unknown" in e for e in errors)

    def test_mismatched_model_family(self):
        c = CalibrationContract(
            target=CalibrationTarget(parameter="flat_vol", model_family="hull_white"),  # should be black76
            fitting_instruments=(FittingInstrument(instrument_type="cap", data_source="market_quote"),),
            output=OutputBinding(target_path="market_state.vol_surface"),
        )
        errors = validate_calibration_contract(c)
        assert any("black76" in e for e in errors)

    def test_no_fitting_instruments(self):
        c = CalibrationContract(
            target=CalibrationTarget(parameter="flat_vol", model_family="black76"),
            fitting_instruments=(),
            output=OutputBinding(target_path="market_state.vol_surface"),
        )
        errors = validate_calibration_contract(c)
        assert any("fitting instrument" in e.lower() for e in errors)

    def test_unknown_optimizer(self):
        c = CalibrationContract(
            target=CalibrationTarget(parameter="flat_vol", model_family="black76"),
            fitting_instruments=(FittingInstrument(instrument_type="cap", data_source="market_quote"),),
            method=CalibrationMethod(optimizer="unknown_opt"),
            output=OutputBinding(target_path="market_state.vol_surface"),
        )
        errors = validate_calibration_contract(c)
        assert any("unknown_opt" in e.lower() or "Unknown" in e for e in errors)

    def test_unknown_proven_primitive(self):
        c = CalibrationContract(
            target=CalibrationTarget(parameter="flat_vol", model_family="black76"),
            fitting_instruments=(FittingInstrument(instrument_type="cap", data_source="market_quote"),),
            output=OutputBinding(target_path="market_state.vol_surface"),
            proven_primitive="nonexistent_function",
        )
        errors = validate_calibration_contract(c)
        assert any("nonexistent_function" in e for e in errors)

    def test_empty_output_binding(self):
        c = CalibrationContract(
            target=CalibrationTarget(parameter="flat_vol", model_family="black76"),
            fitting_instruments=(FittingInstrument(instrument_type="cap", data_source="market_quote"),),
        )
        errors = validate_calibration_contract(c)
        assert any("target_path" in e for e in errors)


# ---------------------------------------------------------------------------
# Compiler integration
# ---------------------------------------------------------------------------

class TestCompilerIntegration:
    def test_callable_bond_has_calibration(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract

        contract = make_callable_bond_contract(
            description="Callable bond with HW tree",
            observation_schedule=("2025-06-30", "2026-06-30"),
        )
        assert contract.calibration is not None
        assert contract.calibration.proven_primitive == "build_lattice"

    def test_blueprint_has_calibration_step(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract

        contract = make_callable_bond_contract(
            description="Callable bond with HW tree",
            observation_schedule=("2025-06-30", "2026-06-30"),
        )
        blueprint = compile_semantic_contract(contract)
        assert blueprint.calibration_step is not None
        assert blueprint.calibration_step.proven_primitive == "build_lattice"
        # Calibration module should be in route_modules
        assert any("calibration" in m or "lattice" in m for m in blueprint.route_modules)

    def test_vanilla_option_no_calibration(self):
        from trellis.agent.semantic_contracts import make_vanilla_option_contract
        from trellis.agent.semantic_contract_compiler import compile_semantic_contract

        contract = make_vanilla_option_contract(
            description="European call",
            underliers=("AAPL",),
            observation_schedule=("2025-06-30",),
        )
        assert contract.calibration is None
        blueprint = compile_semantic_contract(contract)
        assert blueprint.calibration_step is None

    def test_contract_validation_with_calibration(self):
        from trellis.agent.semantic_contracts import make_callable_bond_contract
        from trellis.agent.semantic_contract_validation import validate_semantic_contract

        contract = make_callable_bond_contract(
            description="Callable bond",
            observation_schedule=("2025-06-30",),
        )
        report = validate_semantic_contract(contract)
        # Should pass validation (calibration is valid)
        cal_errors = [e for e in report.errors if "calibration" in e.lower()]
        assert cal_errors == [], cal_errors


# ---------------------------------------------------------------------------
# Chained calibration
# ---------------------------------------------------------------------------

class TestChainedCalibration:
    def test_depends_on_field(self):
        c1 = hull_white_calibration_contract()
        c2 = CalibrationContract(
            target=CalibrationTarget(parameter="flat_vol", model_family="black76"),
            fitting_instruments=(FittingInstrument(instrument_type="swaption", data_source="market_quote"),),
            output=OutputBinding(target_path="market_state.vol_surface"),
            proven_primitive="calibrate_swaption_black_vol",
            depends_on="hw_calibration",
        )
        assert c2.depends_on == "hw_calibration"
        errors = validate_calibration_contract(c2)
        assert errors == (), errors
