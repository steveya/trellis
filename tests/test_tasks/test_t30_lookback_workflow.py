"""End-to-end regression for the authored T30 lookback comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.global_workflow
def test_t30_authored_lookback_comparison_passes_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the real task contract through materialization, pricing, and comparison."""
    import trellis.agent.analytical_traces as analytical_traces
    import trellis.agent.executor as executor
    import trellis.agent.model_audit as model_audit
    import trellis.agent.platform_requests as platform_requests
    import trellis.agent.platform_traces as platform_traces
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import build_market_state, run_task

    task = next(
        task
        for task in load_task_manifest("TASKS_PROOF_LEGACY.yaml", root=ROOT)
        if task["id"] == "T30"
    )
    generated_root = tmp_path / "generated"

    def write_generated_module(module_path: str, code: str) -> Path:
        output_path = generated_root / module_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)
        return output_path

    # Exercise the production compiler and evaluator while redirecting their
    # append-only diagnostics and generated snapshots out of the worktree.
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
    monkeypatch.setenv("TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST", "1")

    generated_module_name = "trellis.instruments._agent._fresh.lookbackoption"
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

    comparison = result["cross_validation"]
    assert result["success"] is True
    assert result["passed_expectation"] is True
    assert result["attempts"] == 0
    assert comparison["status"] == "passed"
    assert comparison["reference_target"] == "conze_viswanathan_analytical"
    assert comparison["failed_targets"] == []
    assert comparison["passed_targets"] == ["mc_lookback"]
    assert comparison["deviations_pct"]["mc_lookback"] <= 1.25
    assert {
        target: binding["status"]
        for target, binding in comparison["artifact_coherence"].items()
    } == {
        "mc_lookback": "bound_unique_artifact",
        "conze_viswanathan_analytical": "bound_unique_artifact",
    }
