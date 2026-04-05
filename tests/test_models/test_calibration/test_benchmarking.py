"""Tests for calibration benchmark report helpers."""

from __future__ import annotations

import pytest


def test_benchmark_calibration_workflow_returns_stable_summary():
    from trellis.models.calibration.benchmarking import benchmark_calibration_workflow

    timer_values = iter([1.0, 1.25, 2.0, 2.3, 3.0, 3.4])

    def fake_timer() -> float:
        return next(timer_values)

    measurement = benchmark_calibration_workflow(
        label="heston_smile",
        mode="cold",
        runner=lambda: {"status": "ok"},
        repeats=3,
        warmups=1,
        timer=fake_timer,
        notes=("fft_pricing",),
        metadata={"point_count": 5},
    )

    assert measurement.label == "heston_smile"
    assert measurement.mode == "cold"
    assert measurement.mean_seconds == pytest.approx(0.31666666666666665)
    assert measurement.calibrations_per_second > 3.0
    assert measurement.to_dict()["notes"] == ["fft_pricing"]
    assert measurement.to_dict()["metadata"]["point_count"] == 5


def test_build_and_save_calibration_benchmark_report(tmp_path):
    from trellis.models.calibration.benchmarking import (
        CalibrationBenchmarkScenario,
        benchmark_calibration_scenario,
        build_calibration_benchmark_report,
        render_calibration_benchmark_report,
        save_calibration_benchmark_report,
    )

    timer_values = iter([1.0, 1.2, 2.0, 2.1, 3.0, 3.08, 4.0, 4.05])

    def fake_timer() -> float:
        return next(timer_values)

    scenario = CalibrationBenchmarkScenario(
        workflow="sabr",
        label="single_smile",
        cold_runner=lambda: {"mode": "cold"},
        warm_runner=lambda: {"mode": "warm"},
        notes=("least_squares",),
        metadata={"point_count": 7},
    )
    case = benchmark_calibration_scenario(scenario, repeats=2, warmups=0, timer=fake_timer)
    report = build_calibration_benchmark_report(
        benchmark_name="supported_calibration_workflows",
        cases=[case],
        notes=["synthetic benchmark"],
        environment={"python_version": "3.10.0", "platform": "test"},
    )

    assert report["summary"]["workflow_count"] == 1
    assert report["summary"]["warm_start_workflow_count"] == 1
    assert report["summary"]["average_warm_speedup"] > 1.0

    rendered = render_calibration_benchmark_report(report)
    assert "Calibration Benchmark" in rendered
    assert "Warm speedup" in rendered
    assert "single_smile" in rendered

    artifacts = save_calibration_benchmark_report(report, root=tmp_path, stem="supported_calibration_workflows")
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    assert artifacts.report["json_path"] == str(artifacts.json_path)
    assert artifacts.report["text_path"] == str(artifacts.text_path)


def test_supported_calibration_benchmark_scenarios_cover_workflows():
    from trellis.models.calibration.benchmarking import supported_calibration_benchmark_scenarios

    scenarios = supported_calibration_benchmark_scenarios()
    workflows = {scenario.workflow for scenario in scenarios}

    assert workflows == {"hull_white", "sabr", "heston", "local_vol"}
    assert next(s for s in scenarios if s.workflow == "local_vol").warm_runner is None
