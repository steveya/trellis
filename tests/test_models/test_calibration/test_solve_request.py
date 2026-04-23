"""Tests for the typed calibration solve-request substrate."""

from __future__ import annotations

import numpy as raw_np
import pytest

from trellis.models.calibration.solve_request import (
    ConstraintSpec,
    ObjectiveBundle,
    SolveBackendRecord,
    SolveBackendRegistry,
    SolveBounds,
    SolveProvenance,
    SolveRequest,
    SolveReplayArtifact,
    SolveResult,
    build_solve_provenance,
    build_solve_replay_artifact,
    UnsupportedSolveCapabilityError,
    WarmStart,
    execute_solve_request,
)


def test_root_solve_request_serializes_and_executes():
    request = SolveRequest(
        request_id="root_demo",
        problem_kind="root_scalar",
        parameter_names=("vol",),
        initial_guess=(3.0,),
        bounds=SolveBounds(lower=(0.0,), upper=(5.0,)),
        objective=ObjectiveBundle(
            objective_kind="root_scalar",
            labels=("price_residual",),
            target_values=(0.0,),
            scalar_objective_fn=lambda vol: float(vol) ** 2 - 4.0,
            metadata={"quote_kind": "demo"},
        ),
        solver_hint="brentq",
        warm_start=WarmStart(parameter_values=(3.0,), source="midpoint_seed"),
    )

    payload = request.to_payload()

    assert payload["problem_kind"] == "root_scalar"
    assert payload["objective"]["labels"] == ["price_residual"]
    assert payload["warm_start"]["source"] == "midpoint_seed"

    result = execute_solve_request(request)

    assert result.solution == pytest.approx((2.0,), abs=1e-10)
    assert result.method == "brentq"


def test_least_squares_request_serializes_constraints_and_executes():
    request = SolveRequest(
        request_id="least_squares_demo",
        problem_kind="least_squares",
        parameter_names=("alpha", "beta"),
        initial_guess=(0.0, 0.0),
        bounds=SolveBounds(lower=(-5.0, -5.0), upper=(5.0, 5.0)),
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=("eq_alpha", "eq_beta"),
            target_values=(0.0, 0.0),
            vector_objective_fn=lambda params: raw_np.array(
                [params[0] - 1.0, params[1] - 2.0],
                dtype=float,
            ),
            scalar_objective_fn=lambda params: float((params[0] - 1.0) ** 2 + (params[1] - 2.0) ** 2),
            jacobian_fn=lambda params: raw_np.array(
                [2.0 * (params[0] - 1.0), 2.0 * (params[1] - 2.0)],
                dtype=float,
            ),
            metadata={"target_family": "demo"},
        ),
        solver_hint="L-BFGS-B",
        warm_start=WarmStart(parameter_values=(0.5, 0.5), source="previous_fit"),
    )

    payload = request.to_payload()

    assert payload["problem_kind"] == "least_squares"
    assert payload["constraints"] == []
    assert payload["objective"]["derivatives"]["jacobian"] == "provided"

    result = execute_solve_request(request)

    assert result.solution == pytest.approx((1.0, 2.0), abs=1e-6)
    assert result.objective_value == pytest.approx(0.0, abs=1e-10)
    assert result.method == "L-BFGS-B"
    assert result.metadata["derivative_method"] == "provided_scalar_gradient"


def test_least_squares_request_uses_vector_solver_when_residual_jacobian_is_provided():
    request = SolveRequest(
        request_id="vector_least_squares_demo",
        problem_kind="least_squares",
        parameter_names=("alpha", "beta"),
        initial_guess=(0.0, 0.0),
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=("eq_alpha", "eq_beta"),
            target_values=(0.0, 0.0),
            vector_objective_fn=lambda params: raw_np.array(
                [params[0] - 1.0, params[1] - 2.0],
                dtype=float,
            ),
            jacobian_fn=lambda _params: raw_np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
        ),
        solver_hint="trf",
        options={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
    )

    result = execute_solve_request(request)

    assert result.solution == pytest.approx((1.0, 2.0), abs=1e-10)
    assert result.objective_value == pytest.approx(0.0, abs=1e-12)
    assert result.method == "trf"
    assert result.metadata["derivative_method"] == "provided_vector_jacobian"


def test_least_squares_request_records_two_point_fallback_when_no_vector_jacobian_is_supplied():
    request = SolveRequest(
        request_id="vector_least_squares_fallback_demo",
        problem_kind="least_squares",
        parameter_names=("alpha", "beta"),
        initial_guess=(0.0, 0.0),
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=("eq_alpha", "eq_beta"),
            target_values=(0.0, 0.0),
            vector_objective_fn=lambda params: raw_np.array(
                [params[0] - 1.0, params[1] - 2.0],
                dtype=float,
            ),
        ),
        solver_hint="trf",
        options={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
    )

    result = execute_solve_request(request)

    assert result.solution == pytest.approx((1.0, 2.0), abs=1e-10)
    assert result.objective_value == pytest.approx(0.0, abs=1e-12)
    assert result.method == "trf"
    assert result.metadata["derivative_method"] == "scipy_2point_residual_jacobian"


def test_backend_registry_exposes_default_scipy_capabilities():
    registry = SolveBackendRegistry()

    backend = registry.get_backend("scipy")

    assert backend.backend_id == "scipy"
    assert backend.problem_kinds == ("root_scalar", "least_squares")
    assert backend.supports_bounds is True
    assert backend.supports_constraints is False
    assert backend.supports_warm_start is True


def test_execute_solve_request_rejects_unsupported_backend_capabilities():
    request = SolveRequest(
        request_id="constrained_demo",
        problem_kind="least_squares",
        parameter_names=("alpha", "beta"),
        initial_guess=(0.0, 0.0),
        bounds=SolveBounds(lower=(-5.0, -5.0), upper=(5.0, 5.0)),
        constraints=(
            ConstraintSpec(
                kind="equality",
                parameter_names=("alpha", "beta"),
                metadata={"expression": "beta - alpha - 1"},
            ),
        ),
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=("eq_alpha", "eq_beta"),
            target_values=(0.0, 0.0),
            scalar_objective_fn=lambda params: float((params[0] - 1.0) ** 2 + (params[1] - 2.0) ** 2),
        ),
        solver_hint="L-BFGS-B",
    )
    registry = SolveBackendRegistry(
        records=(
            SolveBackendRecord(
                backend_id="limited",
                executor=lambda _request: SolveResult(solution=(0.0, 0.0), objective_value=0.0, method="limited"),
                problem_kinds=("least_squares",),
                supports_bounds=True,
                supports_constraints=False,
                supports_warm_start=False,
                supports_jacobian=False,
                supports_hessian=False,
            ),
        ),
        default_backend_id="limited",
    )

    with pytest.raises(UnsupportedSolveCapabilityError, match="constraints"):
        execute_solve_request(request, registry=registry)


def test_execute_solve_request_can_explicitly_fallback_to_scipy():
    request = SolveRequest(
        request_id="fallback_demo",
        problem_kind="root_scalar",
        parameter_names=("vol",),
        initial_guess=(3.0,),
        bounds=SolveBounds(lower=(0.0,), upper=(5.0,)),
        objective=ObjectiveBundle(
            objective_kind="root_scalar",
            labels=("price_residual",),
            target_values=(0.0,),
            scalar_objective_fn=lambda vol: float(vol) ** 2 - 4.0,
        ),
        solver_hint="brentq",
        warm_start=WarmStart(parameter_values=(3.0,), source="previous_fit"),
    )
    registry = SolveBackendRegistry(
        records=(
            SolveBackendRecord(
                backend_id="no_warm_start",
                executor=lambda _request: SolveResult(solution=(0.0,), objective_value=1.0, method="noop"),
                problem_kinds=("root_scalar",),
                supports_bounds=True,
                supports_constraints=False,
                supports_warm_start=False,
                supports_jacobian=False,
                supports_hessian=False,
            ),
        ),
    )

    result = execute_solve_request(
        request,
        backend="no_warm_start",
        fallback_backend="scipy",
        registry=registry,
    )

    assert result.solution == pytest.approx((2.0,), abs=1e-10)
    assert result.metadata["backend_id"] == "scipy"
    assert result.metadata["fallback_from"] == "no_warm_start"


def test_solver_provenance_and_replay_artifact_capture_backend_and_residuals():
    request = SolveRequest(
        request_id="artifact_demo",
        problem_kind="root_scalar",
        parameter_names=("vol",),
        initial_guess=(3.0,),
        bounds=SolveBounds(lower=(0.0,), upper=(5.0,)),
        objective=ObjectiveBundle(
            objective_kind="root_scalar",
            labels=("price_residual",),
            target_values=(0.0,),
            scalar_objective_fn=lambda vol: float(vol) ** 2 - 4.0,
        ),
        solver_hint="brentq",
        warm_start=WarmStart(parameter_values=(3.0,), source="previous_fit"),
        options={"tol": 1e-10},
    )
    result = execute_solve_request(request)

    provenance = build_solve_provenance(request, result)
    replay = build_solve_replay_artifact(request, result)

    assert isinstance(provenance, SolveProvenance)
    assert isinstance(replay, SolveReplayArtifact)
    assert provenance.backend["backend_id"] == "scipy"
    assert provenance.backend["derivative_method"] == "not_applicable_root_scalar"
    assert provenance.options["tol"] == pytest.approx(1e-10)
    assert provenance.termination["success"] is True
    assert provenance.residual_diagnostics["max_abs_residual"] < 1e-8
    assert replay.request["request_id"] == "artifact_demo"
    assert replay.backend["backend_id"] == "scipy"
    assert replay.backend["derivative_method"] == "not_applicable_root_scalar"
    assert replay.termination["success"] is True
