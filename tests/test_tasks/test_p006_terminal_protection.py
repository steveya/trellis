"""End-to-end regression for the authored P006 terminal-protection contract."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
ANALYTICAL_TARGET = "weighted_rank_analytical"
MONTE_CARLO_TARGET = "weighted_rank_monte_carlo"


def _p006_task() -> dict:
    from trellis.agent.task_manifests import load_task_manifest

    return next(
        task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
        if task["id"] == "P006"
    )


def test_p006_market_keeps_named_curves_but_authored_spread_controls_hazard():
    """The selected credit curve is provenance, not P006 marginal authority."""
    from trellis.agent.benchmark_contracts import benchmark_spec_overrides
    from trellis.agent.market_scenarios import (
        construct_market_state_for_scenario,
        market_scenario_contract_from_task,
    )
    from trellis.agent.task_runtime import build_market_state
    from trellis.models.credit_basket_copula import resolve_credit_basket_inputs

    task = _p006_task()
    scenario = market_scenario_contract_from_task(task, root=ROOT)
    assert scenario is not None
    market_state, _ = construct_market_state_for_scenario(
        scenario,
        build_market_state(),
        task_id="P006",
    )
    overrides = benchmark_spec_overrides(task, root=ROOT)
    resolved = resolve_credit_basket_inputs(
        market_state,
        SimpleNamespace(**overrides),
    )

    assert market_state.selected_curve_names["discount_curve"] == "usd_ois"
    assert market_state.selected_curve_names["credit_curve"] == "usd_ig"
    assert resolved.hazard_rate == pytest.approx(0.025 / (1.0 - 0.4))
    assert resolved.hazard_rate != pytest.approx(0.025)


@pytest.mark.global_workflow
def test_p006_prices_both_declared_terminal_protection_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run real P006 through scenario selection, compilation, and comparison."""
    import trellis.agent.analytical_traces as analytical_traces
    import trellis.agent.executor as executor
    import trellis.agent.model_audit as model_audit
    import trellis.agent.platform_requests as platform_requests
    import trellis.agent.platform_traces as platform_traces
    from trellis.agent.task_runtime import build_market_state, run_task

    task = _p006_task()
    generated_root = tmp_path / "generated"

    def write_generated_module(module_path: str, code: str) -> Path:
        output_path = generated_root / module_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)
        return output_path

    monkeypatch.setattr(executor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(executor, "TRELLIS_PACKAGE_ROOT", tmp_path / "trellis")
    monkeypatch.setattr(executor, "_REPO_REVISION", "test")
    monkeypatch.setattr(executor, "write_module", write_generated_module)
    monkeypatch.setattr(
        analytical_traces,
        "TRACE_ROOT",
        tmp_path / "traces" / "analytical",
    )
    monkeypatch.setattr(
        platform_traces,
        "TRACE_ROOT",
        tmp_path / "traces" / "platform",
    )
    monkeypatch.setattr(model_audit, "_AUDIT_DIR", tmp_path / "audits")
    monkeypatch.setattr(
        platform_requests,
        "_record_semantic_extension_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setenv("TRELLIS_OFFLINE_LOCAL_AGENTS", "1")
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", "1")
    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", "1")
    monkeypatch.setenv("TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST", "1")

    generated_module_name = "trellis.instruments._agent._fresh.nthtodefault"
    prior_generated_module = sys.modules.get(generated_module_name)
    try:
        result = run_task(
            task,
            build_market_state(),
            fresh_build=True,
            recovery_mode="strict",
            execution_mode_override="deterministic_replay",
            task_run_storage_root=tmp_path / "task-runs",
            task_run_storage_layout="standalone",
        )
    finally:
        if prior_generated_module is None:
            sys.modules.pop(generated_module_name, None)
        else:
            sys.modules[generated_module_name] = prior_generated_module

    analytical = result["method_results"][ANALYTICAL_TARGET]
    monte_carlo = result["method_results"][MONTE_CARLO_TARGET]

    assert analytical["success"] is True, analytical
    assert monte_carlo["success"] is True, monte_carlo
    assert analytical["comparison_target_contract"]["backend_binding_id"] == (
        "trellis.models.contingent_cashflows.nth_to_default_probability"
    )
    assert monte_carlo["comparison_target_contract"]["backend_binding_id"] == (
        "trellis.models.copulas.gaussian.GaussianCopula"
    )
    assert analytical["artifact_binding"]["status"] == "bound_unique_artifact", (
        analytical["artifact_binding"]["failures"]
    )
    assert monte_carlo["artifact_binding"]["status"] == "bound_unique_artifact", (
        monte_carlo["artifact_binding"]["failures"]
    )
    assert analytical["artifact_binding"]["artifact_identity"] != (
        monte_carlo["artifact_binding"]["artifact_identity"]
    )

    outputs = result["comparison_outputs"]
    assert outputs[ANALYTICAL_TARGET] == pytest.approx(
        {"price": 119_989.38538859207, "spread_cs01": 569.163943123931},
        abs=0.01,
    )
    assert outputs[MONTE_CARLO_TARGET] == pytest.approx(
        {"price": 120_341.630979619, "spread_cs01": 516.7828513428103},
        abs=0.01,
    )
    assert result["cross_validation"]["status"] == "passed"
    assert result["attempts"] == 0
    assert result["success"] is True
