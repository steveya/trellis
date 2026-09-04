"""End-to-end regression for the authored P005 physical Bermudan contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TREE_TARGET = "physical_bermudan_swaption_lattice"
MC_TARGET = "physical_bermudan_swaption_monte_carlo"
MC_BLOCKER = "missing_composition_surface:physical_bermudan_swaption:monte_carlo"


@pytest.mark.global_workflow
def test_p005_prices_strict_tree_and_honestly_blocks_missing_mc_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the real task through scenario selection, compilation, and pricing."""
    import trellis.agent.analytical_traces as analytical_traces
    import trellis.agent.executor as executor
    import trellis.agent.model_audit as model_audit
    import trellis.agent.platform_requests as platform_requests
    import trellis.agent.platform_traces as platform_traces
    from trellis.agent.evals import classify_task_result
    from trellis.agent.task_manifests import load_task_manifest
    from trellis.agent.task_runtime import build_market_state, run_task

    task = next(
        task
        for task in load_task_manifest("TASKS_EXTENSION.yaml", root=ROOT)
        if task["id"] == "P005"
    )
    generated_root = tmp_path / "generated"

    def write_generated_module(module_path: str, code: str) -> Path:
        output_path = generated_root / module_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)
        return output_path

    # Keep production compilation and pricing intact while redirecting every
    # generated snapshot and append-only diagnostic out of the worktree.
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

    generated_module_name = (
        "trellis.instruments._agent._fresh.physicalbermudanswaption"
    )
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

    tree = result["method_results"][TREE_TARGET]
    monte_carlo = result["method_results"][MC_TARGET]

    assert tree["success"] is True, tree
    assert tree["price"] == pytest.approx(58_469.10443135, abs=1.0e-6)
    assert monte_carlo["success"] is False
    assert monte_carlo["attempts"] == 0
    assert monte_carlo["blocker_details"]["blocker_codes"] == [MC_BLOCKER]
    assert result["comparison_outputs"] == {
        TREE_TARGET: {"price": pytest.approx(58_469.10443135, abs=1.0e-6)}
    }
    assert result["cross_validation"]["status"] == "insufficient_results"
    assert result["attempts"] == 0
    assert result["success"] is False
    assert result["expected_honest_block"] is True
    assert result["outcome_class"] == "honest_block"
    assert result["passed_expectation"] is True
    assert classify_task_result(result) == "blocked"
    assert result["blocker_details"]["blocker_codes"] == [MC_BLOCKER]
