"""Verification tests for checked pod-risk benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_live_supported_pod_risk_benchmark_report_covers_workflow_shape():
    from trellis.analytics.benchmarking import build_supported_pod_risk_benchmark_report

    report = build_supported_pod_risk_benchmark_report(repeats=1, warmups=0)

    assert report["benchmark_name"] == "pod_risk_workflows"
    assert report["summary"]["workflow_count"] == 6
    assert report["summary"]["steady_state_workflow_count"] == 6
    workflows = {case["workflow"] for case in report["cases"]}
    assert workflows == {
        "pipeline_scenarios",
        "key_rate_durations",
        "portfolio_aad",
        "scenario_pnl",
        "vega",
        "spot_greeks",
    }


def test_checked_pod_risk_benchmark_artifact_covers_supported_workflows():
    payload = json.loads((REPO_ROOT / "docs" / "benchmarks" / "pod_risk_workflows.json").read_text())

    assert payload["benchmark_name"] == "pod_risk_workflows"
    assert payload["summary"]["workflow_count"] == 6
    assert payload["summary"]["steady_state_workflow_count"] == 6
    workflows = {case["workflow"] for case in payload["cases"]}
    assert workflows == {
        "pipeline_scenarios",
        "key_rate_durations",
        "portfolio_aad",
        "scenario_pnl",
        "vega",
        "spot_greeks",
    }
