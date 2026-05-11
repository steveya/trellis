"""Tests for calibration problem IR and the first SABR adapter."""

from __future__ import annotations

from datetime import date

import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.calibration.credit import (
    CreditHazardCalibrationQuote,
    build_single_name_credit_calibration_problem_ir,
    calibrate_single_name_credit_curve_workflow,
    fit_single_name_credit_problem_ir,
)
from trellis.models.calibration.problem_ir import (
    CalibrationObjectiveSpec,
    CalibrationProblemIR,
    CalibrationTargetSpec,
    CalibrationVariableSpec,
)
from trellis.models.calibration.sabr_fit import (
    build_sabr_smile_calibration_problem_ir,
    fit_sabr_smile_problem_ir,
    fit_sabr_smile_surface,
)
from trellis.models.processes.sabr import SABRProcess


SETTLE = date(2024, 11, 15)


def _sabr_surface():
    from trellis.models.calibration.sabr_fit import build_sabr_smile_surface

    forward = 100.0
    expiry = 1.0
    beta = 0.5
    sabr = SABRProcess(0.20, beta, -0.3, 0.4)
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    market_vols = [sabr.implied_vol(forward, strike, expiry) for strike in strikes]
    return build_sabr_smile_surface(
        forward,
        expiry,
        strikes,
        market_vols,
        beta=beta,
        labels=[f"pt_{index}" for index in range(len(strikes))],
        weights=[1.0, 1.0, 1.5, 2.0, 1.5, 1.0, 1.0],
        surface_name="usd_rates_1y_smile",
        metadata={"fixture": "problem_ir"},
    )


def _credit_market_state() -> MarketState:
    return MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        discount=YieldCurve.flat(0.03),
        selected_curve_names={"discount_curve": "usd_ois"},
        market_provenance={"source_kind": "explicit_input", "source_ref": "problem_ir_unit_test"},
    )


def _credit_quotes():
    return (
        CreditHazardCalibrationQuote(1.0, 120.0, "spread", label="spread_1y"),
        CreditHazardCalibrationQuote(5.0, 180.0, "spread", label="spread_5y"),
    )


def test_calibration_problem_ir_validates_unique_names_and_targets():
    variable = CalibrationVariableSpec(
        name="alpha",
        coordinate_chart="positive",
        initial_value=0.2,
        lower_bound=1e-6,
    )
    target = CalibrationTargetSpec(
        target_id="pt_0",
        instrument_id="strike_100",
        quote_family="implied_vol",
        quote_convention="black",
        quote_value=0.2,
    )
    objective = CalibrationObjectiveSpec(
        objective_kind="least_squares",
        loss_function="weighted_sum_of_squares",
        residual_count=1,
        derivative_method="autodiff_scalar_gradient",
    )

    with pytest.raises(ValueError, match="duplicate variable"):
        CalibrationProblemIR(
            problem_id="duplicate_variables",
            workflow_id="demo",
            family_id="demo",
            variables=(variable, variable),
            targets=(target,),
            objective=objective,
        )

    with pytest.raises(ValueError, match="duplicate target"):
        CalibrationProblemIR(
            problem_id="duplicate_targets",
            workflow_id="demo",
            family_id="demo",
            variables=(variable,),
            targets=(target, target),
            objective=objective,
        )


def test_sabr_problem_ir_represents_existing_solve_request_payload():
    surface = _sabr_surface()

    problem = build_sabr_smile_calibration_problem_ir(surface)
    direct = fit_sabr_smile_surface(surface)

    payload = problem.to_payload()

    assert problem.problem_id == "sabr_smile_least_squares"
    assert problem.workflow_id == "sabr_smile"
    assert [variable.name for variable in problem.variables] == ["alpha", "rho", "nu"]
    assert [target.target_id for target in problem.targets] == list(surface.labels)
    assert problem.objective.derivative_method == "autodiff_scalar_gradient"
    assert problem.objective.residual_count == len(surface.points)
    assert problem.solve_request_payload == direct.solve_request.to_payload()
    assert payload["objective"]["metadata"]["quote_map"]["quote_family"] == "implied_vol"
    assert payload["materialization"]["object_kind"] == "model_parameter_set"
    assert payload["metadata"]["engine_backed"] is False


def test_sabr_problem_ir_adapter_matches_direct_workflow():
    surface = _sabr_surface()
    problem = build_sabr_smile_calibration_problem_ir(surface)

    direct = fit_sabr_smile_surface(surface)
    adapted = fit_sabr_smile_problem_ir(problem, surface)

    assert adapted.sabr.alpha == pytest.approx(direct.sabr.alpha, abs=1e-12)
    assert adapted.sabr.rho == pytest.approx(direct.sabr.rho, abs=1e-12)
    assert adapted.sabr.nu == pytest.approx(direct.sabr.nu, abs=1e-12)
    assert adapted.solve_request.to_payload() == direct.solve_request.to_payload()
    assert adapted.solver_replay_artifact.request == direct.solver_replay_artifact.request
    assert adapted.diagnostics.to_payload() == direct.diagnostics.to_payload()
    assert adapted.summary == direct.summary
    assert adapted.provenance["calibration_problem_ir"]["problem_id"] == problem.problem_id
    assert adapted.provenance["calibration_problem_ir"]["adapter_id"] == "sabr_smile_problem_ir_v1"


def test_sabr_problem_ir_adapter_rejects_wrong_problem_family():
    surface = _sabr_surface()
    problem = build_sabr_smile_calibration_problem_ir(surface)
    wrong_problem = CalibrationProblemIR(
        problem_id=problem.problem_id,
        workflow_id="credit_curve",
        family_id=problem.family_id,
        variables=problem.variables,
        targets=problem.targets,
        objective=problem.objective,
        solve_request_payload=problem.solve_request_payload,
    )

    with pytest.raises(ValueError, match="SABR smile"):
        fit_sabr_smile_problem_ir(wrong_problem, surface)


def test_single_name_credit_problem_ir_represents_existing_solve_request_payload():
    market_state = _credit_market_state()
    quotes = _credit_quotes()

    problem = build_single_name_credit_calibration_problem_ir(
        quotes,
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )
    direct = calibrate_single_name_credit_curve_workflow(
        quotes,
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )
    payload = problem.to_payload()

    assert problem.problem_id == "single_name_credit_cds_par_spread_least_squares"
    assert problem.workflow_id == "single_name_credit_curve"
    assert problem.family_id == "credit"
    assert [variable.name for variable in problem.variables] == ["hazard_1", "hazard_2"]
    assert [target.target_id for target in problem.targets] == ["spread_1y", "spread_5y"]
    assert [target.quote_family for target in problem.targets] == ["spread", "spread"]
    assert problem.objective.derivative_method == "scipy_2point_residual_jacobian"
    assert problem.objective.residual_count == len(quotes)
    assert problem.solve_request_payload == direct.solve_request.to_payload()
    assert payload["materialization"]["object_kind"] == "credit_curve"
    assert payload["dependencies"][0]["dependency_id"] == "discount_curve"
    assert payload["metadata"]["engine_backed"] is False


def test_single_name_credit_problem_ir_adapter_matches_direct_workflow():
    market_state = _credit_market_state()
    quotes = _credit_quotes()
    problem = build_single_name_credit_calibration_problem_ir(
        quotes,
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )

    direct = calibrate_single_name_credit_curve_workflow(
        quotes,
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )
    adapted = fit_single_name_credit_problem_ir(
        problem,
        quotes,
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )

    assert adapted.solve_request.to_payload() == direct.solve_request.to_payload()
    assert adapted.solver_replay_artifact.request == direct.solver_replay_artifact.request
    assert adapted.target_hazards == pytest.approx(direct.target_hazards, abs=1e-12)
    assert adapted.model_hazards == pytest.approx(direct.model_hazards, abs=1e-12)
    assert adapted.credit_curve.hazard_rates == pytest.approx(direct.credit_curve.hazard_rates, abs=1e-12)
    assert adapted.summary == direct.summary
    assert adapted.provenance["calibration_problem_ir"]["problem_id"] == problem.problem_id
    assert adapted.provenance["calibration_problem_ir"]["adapter_id"] == "single_name_credit_problem_ir_v1"


def test_single_name_credit_problem_ir_adapter_rejects_wrong_problem_family():
    market_state = _credit_market_state()
    problem = build_single_name_credit_calibration_problem_ir(
        _credit_quotes(),
        market_state,
        recovery=0.4,
        curve_name="acme_credit",
    )
    wrong_problem = CalibrationProblemIR(
        problem_id=problem.problem_id,
        workflow_id="sabr_smile",
        family_id="sabr",
        variables=problem.variables,
        targets=problem.targets,
        objective=problem.objective,
        solve_request_payload=problem.solve_request_payload,
    )

    with pytest.raises(ValueError, match="single-name credit"):
        fit_single_name_credit_problem_ir(wrong_problem, _credit_quotes(), market_state)
