"""T82 preserves its bond fixture without inventing callable risk semantics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION = (
    "Retain the named USD fixed-coupon callable-bond fixture for the requested "
    "vega, OAS, duration, and scenario analysis. Do not build or price T82 until "
    "its missing analytics definitions and callable references are authored."
)
REASON = (
    "T82 is a proof hold: the bond and market are authored, but OAS target price "
    "and clean/dirty/holder-PV basis, volatility coordinate and bump, duration "
    "definition and curve bump, scenario coordinates and ladder, required "
    "outputs and units, acceptance tolerances, and callable-analytics references "
    "are missing. Generic Vega moves the Black volatility surface, not the "
    "explicit Hull-White sigma in this fixture. Straight-bond Greeks are not a "
    "callable-analytics reference. Bounded duration, OAS-duration, and scenario "
    "APIs exist; their existence does not supply these missing definitions."
)
MISSING_INPUTS = [
    "oas_target_price",
    "oas_price_basis_clean_dirty_or_holder_pv",
    "volatility_coordinate_and_units",
    "volatility_bump_size_and_units",
    "duration_definition",
    "duration_curve_coordinate_bump_size_and_units",
    "scenario_curve_coordinates",
    "scenario_ladder_and_bump_units",
    "required_analytics_outputs_and_units",
    "acceptance_metrics_and_tolerances",
    "callable_analytics_references",
]
HOLD_CONTRACT = {
    "requested_analytics": ["vega", "oas", "duration", "scenarios"],
    "model_parameter_set": "callable_fixed_5pct_proof:hull_white",
    "missing_input_policy": "requires_authored_values_no_defaults",
    "vega_boundary": {
        "generic_coordinate": "black_vol_surface",
        "fixture_coordinate": "hull_white_sigma",
        "generic_vega_is_model_sigma_vega": False,
    },
    "reference_policy": "straight_bond_greeks_are_not_callable_analytics_reference",
    "acceptance_policy": "not_evaluated_while_held",
    "execution_policy": "zero_market_builder_llm_pricer_attempts",
}


def _task(task_id="T82"):
    from trellis.agent.task_manifests import load_task_manifest

    return deepcopy(
        next(
            task
            for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml")
            if task["id"] == task_id
        )
    )


def _expected_hold():
    task = _task("T02")
    for field in ("cross_validate", "new_component", "construct", "description"):
        task.pop(field, None)
    task.update(
        id="T82",
        title="Callable bond full analytics: vega, OAS, duration, scenarios",
        description=DESCRIPTION,
        task_disposition="proof_hold",
        disposition_reason=REASON,
        missing_inputs=deepcopy(MISSING_INPUTS),
        analytics_hold_contract=deepcopy(HOLD_CONTRACT),
        validation_policy="exact_nonpricing_hold",
        construct=["analytics", "lattice"],
        new_component=None,
        status="blocked",
    )
    return task


def _exact_issues(task, *, root=ROOT):
    from trellis.agent.task_manifest_validation import _validate_legacy_task

    return _validate_legacy_task("TASKS_PROOF_LEGACY.yaml", task, "task", root=root)


def test_t82_repository_row_is_the_exact_authored_hold():
    assert _task() == _expected_hold()
    assert _exact_issues(_task()) == []


@pytest.mark.parametrize(
    "field",
    (
        "description",
        "task_disposition",
        "disposition_reason",
        "missing_inputs",
        "analytics_hold_contract",
        "proof_fixture_id",
        "proof_fixture_digest",
        "proof_fixture_schema_version",
        "market_scenario_id",
        "benchmark_contract",
        "instrument_type",
        "validation_policy",
        "construct",
        "new_component",
        "status",
    ),
)
def test_t82_exact_validation_rejects_missing_hold_fields(field):
    task = _expected_hold()
    del task[field]
    assert "legacy.callable_analytics_hold_invalid_contract" in {
        issue.code for issue in _exact_issues(task)
    }


@pytest.mark.parametrize("missing_input", MISSING_INPUTS)
def test_t82_exact_validation_rejects_omitted_missing_input(missing_input):
    task = _expected_hold()
    task["missing_inputs"].remove(missing_input)
    assert "legacy.callable_analytics_hold_invalid_contract" in {
        issue.code for issue in _exact_issues(task)
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda task: task.update(task_disposition="named_proof_fixture"),
        lambda task: task.update(task_disposition="executable_pricing"),
        lambda task: task.update(status="done"),
        lambda task: task.update(cross_validate={"internal": ["straight_bond_greeks"]}),
        lambda task: task.update(seed=42),
        lambda task: task.update(simulation_seed=42),
        lambda task: task["benchmark_contract"].update(coupon=0.06),
        lambda task: task["benchmark_contract"]["call_dates"].append("2034-01-15"),
        lambda task: task["analytics_hold_contract"]["vega_boundary"].update(
            generic_vega_is_model_sigma_vega=True
        ),
        lambda task: task["analytics_hold_contract"]["vega_boundary"].update(
            generic_vega_is_model_sigma_vega=0
        ),
        lambda task: task.update(proof_fixture_schema_version=True),
        lambda task: task["analytics_hold_contract"].update(acceptance_policy="passed"),
        lambda task: task["market"].update(as_of="2025-01-16"),
        lambda task: task["market"]["scenario_contract"].update(domestic_rate=0.06),
    ),
)
def test_t82_exact_validation_and_runtime_reject_contract_drift(mutation):
    from trellis.agent.task_manifest_validation import TaskManifestValidationError
    from trellis.agent.task_runtime import run_task

    task = _expected_hold()
    mutation(task)
    assert "legacy.callable_analytics_hold_invalid_contract" in {
        issue.code for issue in _exact_issues(task)
    }

    def forbidden(*args, **kwargs):
        pytest.fail("malformed T82 must not invoke a builder")

    with pytest.raises(
        TaskManifestValidationError, match="callable_analytics_hold_invalid_contract"
    ):
        run_task(task, object(), build_fn=forbidden)


def test_t82_structured_callable_evidence_is_title_independent(monkeypatch):
    from trellis.agent import task_runtime

    def forbidden(*args, **kwargs):
        pytest.fail("the held fixture must not borrow title/prose economics")

    monkeypatch.setattr(task_runtime, "_bootstrap_callable_bond_description", forbidden)
    monkeypatch.setattr(
        "trellis.agent.semantic_contracts.draft_semantic_contract", forbidden
    )
    task = _expected_hold()
    original = task_runtime.task_to_semantic_contract(task)
    task["title"] = "Uninformative label"
    renamed = task_runtime.task_to_semantic_contract(task)

    assert original == renamed
    assert task_runtime._effective_task_description(task) == DESCRIPTION
    assert task_runtime.task_to_description(task) == DESCRIPTION
    assert renamed.product.observation_schedule == (
        "2028-01-15",
        "2030-01-15",
        "2032-01-15",
    )
    assert renamed.product.term_fields["coupon"] == 0.05
    assert renamed.product.term_fields["notional"] == 100.0


@pytest.mark.global_workflow
@pytest.mark.parametrize("entrypoint", ("runtime", "runner"))
def test_t82_hold_stops_before_market_builder_and_pricer(
    monkeypatch, tmp_path, entrypoint
):
    from scripts import run_tasks
    from trellis.agent import task_runtime
    from trellis.agent.task_manifest_validation import TaskManifestValidationError

    calls = []

    def forbidden(*args, **kwargs):
        calls.append("unexpected execution")
        pytest.fail("T82 must stop at admission with zero execution attempts")

    monkeypatch.setattr(task_runtime, "_effective_task_description", forbidden)
    monkeypatch.setattr(task_runtime, "build_market_state_for_task", forbidden)
    monkeypatch.setattr("trellis.engine.payoff_pricer.price_payoff", forbidden)
    monkeypatch.setattr(run_tasks, "build_market_state", forbidden)
    task = _expected_hold()
    task["title"] = "Uninformative label"

    with pytest.raises(TaskManifestValidationError) as exc:
        if entrypoint == "runtime":
            task_runtime.run_task(
                task, object(), build_fn=forbidden, price_fn=forbidden
            )
        else:
            run_tasks.run_block(
                [task], str(tmp_path / "results.json"), offline_local_agents=True
            )

    assert {issue.code for issue in exc.value.report.blocking_issues} == {
        "legacy.non_executable_disposition"
    }
    assert REASON in str(exc.value)
    assert calls == []
    assert not (tmp_path / "results.json").exists()


@pytest.mark.global_workflow
def test_t82_offline_cli_reports_hold_without_a_pricing_result(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "t82_results.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_tasks.py"),
            "--task-id",
            "T82",
            "--status",
            "all",
            "--offline-local-agents",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 1
    assert "legacy.non_executable_disposition" in completed.stderr
    assert REASON in completed.stderr
    assert "# Running" not in completed.stdout
    assert not output.exists()
