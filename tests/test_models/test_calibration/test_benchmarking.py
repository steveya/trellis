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
    assert "json_path" not in artifacts.report
    assert "text_path" not in artifacts.report
    assert str(tmp_path) not in artifacts.json_path.read_text()


def test_supported_calibration_benchmark_scenarios_cover_workflows():
    from trellis.models.calibration.benchmarking import supported_calibration_benchmark_scenarios

    scenarios = supported_calibration_benchmark_scenarios()
    scenario_map = {scenario.workflow: scenario for scenario in scenarios}
    workflows = set(scenario_map)

    assert workflows == {"hull_white", "sabr", "heston", "local_vol", "credit"}
    hull_white = scenario_map["hull_white"]
    assert hull_white.metadata["multi_curve_roles"]["discount_curve"] == "usd_ois"
    assert hull_white.metadata["multi_curve_roles"]["forecast_curve"] == "USD-SOFR-3M"
    assert hull_white.metadata["multi_curve_roles"]["rate_index"] == "USD-SOFR-3M"
    assert scenario_map["local_vol"].warm_runner is None
    assert scenario_map["credit"].warm_runner is None
    assert scenario_map["sabr"].metadata["surface_name"] == "usd_rates_smile"
    assert scenario_map["sabr"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["heston"].metadata["surface_name"] == "spx_heston_implied_vol"
    assert scenario_map["heston"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["local_vol"].metadata["source_surface_name"] == "spx_heston_implied_vol"
    assert scenario_map["local_vol"].metadata["surface_name"] == "spx_local_vol"
    assert scenario_map["local_vol"].metadata["synthetic_generation_contract_version"] == "v2"
