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

    assert workflows == {
        "hull_white",
        "caplet_strip",
        "sabr",
        "swaption_cube",
        "equity_vol_surface",
        "heston",
        "heston_surface",
        "local_vol",
        "credit",
        "basket_credit",
    }
    hull_white = scenario_map["hull_white"]
    assert hull_white.metadata["multi_curve_roles"]["discount_curve"] == "usd_ois"
    assert hull_white.metadata["multi_curve_roles"]["forecast_curve"] == "USD-SOFR-3M"
    assert hull_white.metadata["multi_curve_roles"]["rate_index"] == "USD-SOFR-3M"
    assert scenario_map["caplet_strip"].warm_runner is None
    assert scenario_map["caplet_strip"].metadata["surface_name"] == "usd_caplet_strip"
    assert scenario_map["equity_vol_surface"].warm_runner is None
    assert scenario_map["local_vol"].warm_runner is None
    assert scenario_map["credit"].warm_runner is None
    assert scenario_map["sabr"].metadata["surface_name"] == "usd_rates_smile"
    assert scenario_map["sabr"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["swaption_cube"].warm_runner is None
    assert scenario_map["swaption_cube"].metadata["surface_name"] == "usd_swaption_cube"
    assert scenario_map["swaption_cube"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["equity_vol_surface"].metadata["surface_name"] == "spx_surface_authority"
    assert scenario_map["equity_vol_surface"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["heston"].metadata["surface_name"] == "spx_heston_implied_vol"
    assert scenario_map["heston"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["heston_surface"].metadata["surface_name"] == "spx_surface_authority"
    assert scenario_map["heston_surface"].metadata["synthetic_generation_contract_version"] == "v2"
    assert scenario_map["local_vol"].metadata["source_surface_name"] == "spx_heston_implied_vol"
    assert scenario_map["local_vol"].metadata["surface_name"] == "spx_local_vol"
    assert scenario_map["local_vol"].metadata["synthetic_generation_contract_version"] == "v2"
    basket_credit = scenario_map["basket_credit"]
    assert basket_credit.warm_runner is None
    assert basket_credit.metadata["fixture_style"] == "desk_like"
    assert basket_credit.metadata["surface_name"] == "benchmark_tranche_correlation"
    assert basket_credit.metadata["quote_count"] == 6
    assert basket_credit.metadata["tranche_count"] == 3
    assert basket_credit.metadata["maturity_count"] == 2
    assert basket_credit.metadata["linked_credit_curve"] == "benchmark_single_name_credit"
    assert basket_credit.metadata["perturbation_diagnostic"]["label"] == "basket_credit_parallel_quote_up"
    assert basket_credit.metadata["perturbation_diagnostic"]["max_abs_change"] > 0.0
    assert basket_credit.metadata["perturbation_diagnostic"]["threshold_breaches"] == {}
    assert basket_credit.metadata["latency_envelope"]["fixture_style"] == "desk_like"
    assert basket_credit.metadata["latency_envelope"]["cold_mean_limit_seconds"] >= 5.0


def test_perturbation_diagnostic_reports_metric_changes_and_breaches():
    from trellis.models.calibration.benchmarking import diagnose_metric_perturbation

    diagnostic = diagnose_metric_perturbation(
        label="credit_parallel_spread_up",
        perturbation_size=1.0e-4,
        baseline_metrics={"hazard_5y": 0.025, "hazard_7y": 0.03},
        perturbed_metrics={"hazard_5y": 0.0252, "hazard_7y": 0.031},
        instability_thresholds={"hazard_5y": 0.001, "hazard_7y": 0.0005},
    )
    payload = diagnostic.to_dict()

    assert payload["absolute_changes"]["hazard_5y"] == pytest.approx(0.0002)
    assert payload["absolute_changes"]["hazard_7y"] == pytest.approx(0.001)
    assert payload["max_abs_change"] == pytest.approx(0.001)
    assert payload["threshold_breaches"]["hazard_7y"] == pytest.approx(0.001)


def test_latency_envelope_payload_marks_pass_fail_against_case_measurements():
    from trellis.models.calibration.benchmarking import (
        CalibrationLatencyEnvelope,
        evaluate_latency_envelope,
    )

    case = {
        "workflow": "basket_credit",
        "label": "desk_tranche_surface",
        "metadata": {"quote_count": 6},
        "cold": {"mean_seconds": 1.25, "max_seconds": 1.4},
        "warm": None,
    }
    envelope = CalibrationLatencyEnvelope(
        workflow="basket_credit",
        label="desk_tranche_surface",
        fixture_style="desk_like",
        quote_count=6,
        cold_mean_limit_seconds=2.0,
        cold_max_limit_seconds=3.0,
    )

    payload = evaluate_latency_envelope(case, envelope)

    assert payload["status"] == "pass"
    assert payload["cold_mean_seconds"] == pytest.approx(1.25)
    assert payload["cold_mean_limit_seconds"] == pytest.approx(2.0)

    failing = evaluate_latency_envelope(
        {**case, "cold": {"mean_seconds": 2.25, "max_seconds": 3.5}},
        envelope,
    )
    assert failing["status"] == "fail"
    assert set(failing["breaches"]) == {"cold_mean_seconds", "cold_max_seconds"}
