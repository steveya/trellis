"""Tests for the bounded portfolio-AAD benchmark gate."""

from __future__ import annotations

import pytest


def test_supported_portfolio_aad_benchmark_scenarios_cover_expected_lanes():
    from trellis.analytics.benchmarking import (
        supported_portfolio_aad_benchmark_scenarios,
    )

    scenarios = supported_portfolio_aad_benchmark_scenarios()
    case_ids = {scenario.case_id for scenario in scenarios}

    assert case_ids == {
        "bond_curve_nodes",
        "flat_vol_options",
        "grid_vol_options",
        "mixed_supported_book",
    }
    assert all(scenario.book_size > 0 for scenario in scenarios)
    assert all(scenario.factor_count > 0 for scenario in scenarios)
    assert next(
        scenario
        for scenario in scenarios
        if scenario.case_id == "mixed_supported_book"
    ).lane_mix == ("bond_curve", "equity_option_flat_vol")


def test_build_supported_portfolio_aad_benchmark_report_has_stable_schema(tmp_path):
    from trellis.analytics.benchmarking import (
        build_supported_portfolio_aad_benchmark_report,
        render_portfolio_aad_benchmark_report,
        save_portfolio_aad_benchmark_report,
    )

    timer_values = iter(float(value) for value in range(1, 17))
    report = build_supported_portfolio_aad_benchmark_report(
        repeats=1,
        warmups=0,
        timer=lambda: next(timer_values),
    )

    assert report["benchmark_name"] == "portfolio_aad_workflows"
    assert report["report_title"] == "Portfolio AAD Benchmark"
    assert report["summary"]["case_count"] == 4
    assert report["summary"]["total_book_size"] > 0
    assert report["summary"]["total_factor_count"] >= 10
    assert report["summary"]["average_relative_speedup"] == pytest.approx(1.0)

    cases = {case["case_id"]: case for case in report["cases"]}
    assert cases["bond_curve_nodes"]["lane_mix"] == ["bond_curve"]
    assert cases["flat_vol_options"]["factor_count"] == 1
    assert cases["grid_vol_options"]["factor_count"] == 4
    assert cases["mixed_supported_book"]["factor_count"] == 6
    assert all(case["aad_elapsed_seconds"] == 1.0 for case in report["cases"])
    assert all(case["baseline_elapsed_seconds"] == 1.0 for case in report["cases"])
    assert all(case["aad_support_status"] == "supported" for case in report["cases"])
    assert all(case["unsupported_position_count"] == 0 for case in report["cases"])

    rendered = render_portfolio_aad_benchmark_report(report)
    assert "Portfolio AAD Benchmark" in rendered
    assert "mixed_supported_book" in rendered

    artifacts = save_portfolio_aad_benchmark_report(report, root=tmp_path)
    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    assert "json_path" not in artifacts.json_path.read_text()
    assert str(tmp_path) not in artifacts.json_path.read_text()
