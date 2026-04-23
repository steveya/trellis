"""Tests for pod-risk benchmark report helpers."""

from __future__ import annotations

import pytest


def test_benchmark_risk_workflow_returns_stable_summary():
    from trellis.analytics.benchmarking import benchmark_risk_workflow

    timer_values = iter([1.0, 1.2, 2.0, 2.25, 3.0, 3.4])

    def fake_timer() -> float:
        return next(timer_values)

    measurement = benchmark_risk_workflow(
        label="scenario_pnl",
        mode="cold",
        runner=lambda: {"status": "ok"},
        repeats=3,
        warmups=1,
        timer=fake_timer,
        notes=("curve_rebuild",),
        metadata={"scenario_count": 4},
    )

    assert measurement.label == "scenario_pnl"
    assert measurement.mode == "cold"
    assert measurement.mean_seconds == pytest.approx(0.2833333333333334)
    assert measurement.runs_per_second > 3.0
    assert measurement.to_dict()["notes"] == ["curve_rebuild"]
    assert measurement.to_dict()["metadata"]["scenario_count"] == 4


def test_build_and_save_risk_benchmark_report(tmp_path):
    from trellis.analytics.benchmarking import (
        RiskBenchmarkScenario,
        benchmark_risk_scenario,
        build_risk_benchmark_report,
        render_risk_benchmark_report,
        save_risk_benchmark_report,
    )

    timer_values = iter([1.0, 1.25, 2.0, 2.15, 3.0, 3.08, 4.0, 4.04])

    def fake_timer() -> float:
        return next(timer_values)

    scenario = RiskBenchmarkScenario(
        workflow="vega",
        label="bucketed_surface",
        cold_runner=lambda: {"mode": "cold"},
        steady_runner=lambda: {"mode": "steady"},
        notes=("bucketed_vega",),
        metadata={"grid_shape": [3, 3]},
    )
    case = benchmark_risk_scenario(scenario, repeats=2, warmups=0, timer=fake_timer)
    report = build_risk_benchmark_report(
        benchmark_name="pod_risk_workflows",
        cases=[case],
        notes=["synthetic benchmark"],
        environment={"python_version": "3.10.0", "platform": "test"},
    )

    assert report["summary"]["workflow_count"] == 1
    assert report["summary"]["steady_state_workflow_count"] == 1
    assert report["summary"]["average_steady_speedup"] > 1.0

    rendered = render_risk_benchmark_report(report)
    assert "Pod Risk Benchmark" in rendered
    assert "Steady speedup" in rendered
    assert "bucketed_surface" in rendered

    artifacts = save_risk_benchmark_report(report, root=tmp_path, stem="pod_risk_workflows")
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    assert "json_path" not in artifacts.report
    assert "text_path" not in artifacts.report
    assert str(tmp_path) not in artifacts.json_path.read_text()


def test_supported_pod_risk_benchmark_scenarios_cover_workflows():
    from trellis.analytics.benchmarking import supported_pod_risk_benchmark_scenarios

    scenarios = supported_pod_risk_benchmark_scenarios()
    workflows = {scenario.workflow for scenario in scenarios}

    assert workflows == {
        "pipeline_scenarios",
        "key_rate_durations",
        "portfolio_aad",
        "scenario_pnl",
        "vega",
        "spot_greeks",
    }
    assert all(scenario.steady_runner is not None for scenario in scenarios)


def test_portfolio_aad_benchmark_uses_distinct_cold_and_reverse_mode_lanes():
    from trellis.analytics.benchmarking import supported_pod_risk_benchmark_scenarios

    scenario = next(
        scenario
        for scenario in supported_pod_risk_benchmark_scenarios()
        if scenario.workflow == "portfolio_aad"
    )

    cold = scenario.cold_runner()
    steady = scenario.steady_runner()

    assert isinstance(cold, dict)
    assert steady.metadata["resolved_derivative_method"] == "portfolio_aad_vjp"
    assert steady.metadata["backend_operator"] == "vjp"
