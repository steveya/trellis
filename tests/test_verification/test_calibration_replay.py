"""Replay and tolerance fixtures for supported calibration workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as raw_np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _scenario_map():
    from trellis.models.calibration.benchmarking import supported_calibration_benchmark_scenarios

    return {scenario.workflow: scenario for scenario in supported_calibration_benchmark_scenarios()}


def test_supported_calibration_workflows_preserve_replay_contracts_and_fit_tolerances():
    scenarios = _scenario_map()

    hull_white = scenarios["hull_white"].cold_runner()
    assert hull_white.solver_provenance.backend["backend_id"] == "scipy"
    assert hull_white.solver_replay_artifact.request["request_id"] == "hull_white_swaption_least_squares"
    assert hull_white.max_abs_quote_residual < 1e-8

    caplet_strip = scenarios["caplet_strip"].cold_runner()
    assert caplet_strip.provenance["source_ref"] == "calibrate_caplet_vol_strip_workflow"
    assert caplet_strip.summary["surface_model_family"] == "caplet_vol_strip"
    assert caplet_strip.diagnostics.max_abs_repricing_error < 5e-8

    sabr = scenarios["sabr"].cold_runner()
    assert sabr.solver_provenance.backend["backend_id"] == "scipy"
    assert sabr.solver_replay_artifact.request["request_id"] == "sabr_smile_least_squares"
    assert sabr.diagnostics.max_abs_vol_error < 5e-4

    swaption_cube = scenarios["swaption_cube"].cold_runner()
    assert swaption_cube.provenance["source_ref"] == "calibrate_swaption_vol_cube_workflow"
    assert swaption_cube.summary["surface_model_family"] == "swaption_vol_cube"
    assert swaption_cube.diagnostics.max_abs_quote_residual < 1e-10

    heston = scenarios["heston"].cold_runner()
    assert heston.solver_provenance.backend["backend_id"] == "scipy"
    assert heston.solver_replay_artifact.request["request_id"] == "heston_smile_least_squares"
    assert heston.diagnostics.max_abs_vol_error < 1e-4
    assert heston.runtime_binding.model_parameters["model_family"] == "heston"

    equity_vol_surface = scenarios["equity_vol_surface"].cold_runner()
    assert equity_vol_surface.provenance["source_ref"] == "calibrate_equity_vol_surface_workflow"
    assert equity_vol_surface.quote_cleaning is not None
    assert equity_vol_surface.summary["surface_model_family"] == "raw_svi_surface"

    heston_surface = scenarios["heston_surface"].cold_runner()
    assert heston_surface.solver_provenance.backend["backend_id"] == "scipy"
    assert heston_surface.solver_replay_artifact.request["request_id"] == "heston_surface_least_squares"
    assert heston_surface.diagnostics.max_abs_vol_error < 0.03
    assert heston_surface.provenance["source_ref"] == "calibrate_heston_surface_from_equity_vol_surface_workflow"
    assert heston_surface.runtime_binding.model_parameters["model_family"] == "heston"

    local_vol = scenarios["local_vol"].cold_runner()
    assert local_vol.provenance["source_ref"] == "calibrate_local_vol_surface_workflow"
    assert local_vol.diagnostics.unstable_point_count == 0
    assert local_vol.local_vol_surface(100.0, 1.0) == pytest.approx(0.22274408792538325, abs=1e-10)

    credit = scenarios["credit"].cold_runner()
    assert credit.solver_provenance.backend["backend_id"] == "scipy"
    assert (
        credit.solver_replay_artifact.request["request_id"]
        == "single_name_credit_cds_par_spread_least_squares"
    )
    assert credit.max_abs_repricing_error < 5e-12
    assert credit.max_abs_quote_residual < 1e-8
    assert credit.provenance["potential_binding"]["discount_curve_name"] == "usd_ois"
    assert credit.provenance["calibration_target"]["quote_maps"][0]["quote_family"] == "spread"
    assert credit.provenance["calibration_target"]["quote_maps"][1]["quote_family"] == "spread"
    assert credit.provenance["calibration_target"]["quote_maps"][2]["quote_family"] == "spread"
    assert credit.provenance["calibration_target"]["quote_maps"][3]["quote_family"] == "spread"


def test_supported_calibration_workflows_are_replay_stable_within_tolerance():
    scenarios = _scenario_map()

    first_hw = scenarios["hull_white"].cold_runner()
    second_hw = scenarios["hull_white"].cold_runner()
    assert second_hw.solver_replay_artifact.request == first_hw.solver_replay_artifact.request
    assert second_hw.mean_reversion == pytest.approx(first_hw.mean_reversion, abs=1e-10)
    assert second_hw.sigma == pytest.approx(first_hw.sigma, abs=1e-10)

    first_caplet_strip = scenarios["caplet_strip"].cold_runner()
    second_caplet_strip = scenarios["caplet_strip"].cold_runner()
    assert second_caplet_strip.provenance["quotes"] == first_caplet_strip.provenance["quotes"]
    assert raw_np.asarray(second_caplet_strip.stripped_vols, dtype=float) == pytest.approx(
        raw_np.asarray(first_caplet_strip.stripped_vols, dtype=float),
        abs=1e-10,
    )

    first_sabr = scenarios["sabr"].cold_runner()
    second_sabr = scenarios["sabr"].cold_runner()
    assert second_sabr.solver_replay_artifact.request == first_sabr.solver_replay_artifact.request
    assert second_sabr.sabr.alpha == pytest.approx(first_sabr.sabr.alpha, abs=1e-8)
    assert second_sabr.sabr.rho == pytest.approx(first_sabr.sabr.rho, abs=1e-8)
    assert second_sabr.sabr.nu == pytest.approx(first_sabr.sabr.nu, abs=1e-8)

    first_swaption_cube = scenarios["swaption_cube"].cold_runner()
    second_swaption_cube = scenarios["swaption_cube"].cold_runner()
    assert second_swaption_cube.provenance["quotes"] == first_swaption_cube.provenance["quotes"]
    assert raw_np.asarray(second_swaption_cube.market_vols, dtype=float) == pytest.approx(
        raw_np.asarray(first_swaption_cube.market_vols, dtype=float),
        abs=1e-10,
    )

    first_heston = scenarios["heston"].cold_runner()
    second_heston = scenarios["heston"].cold_runner()
    assert second_heston.solver_replay_artifact.request == first_heston.solver_replay_artifact.request
    for key in ("kappa", "theta", "xi", "rho", "v0"):
        assert second_heston.model_parameters[key] == pytest.approx(first_heston.model_parameters[key], abs=1e-8)

    first_equity_vol_surface = scenarios["equity_vol_surface"].cold_runner()
    second_equity_vol_surface = scenarios["equity_vol_surface"].cold_runner()
    assert second_equity_vol_surface.provenance["quote_cleaning"] == first_equity_vol_surface.provenance["quote_cleaning"]
    assert raw_np.asarray(second_equity_vol_surface.repaired_vols, dtype=float) == pytest.approx(
        raw_np.asarray(first_equity_vol_surface.repaired_vols, dtype=float),
        abs=1e-10,
    )

    first_heston_surface = scenarios["heston_surface"].cold_runner()
    second_heston_surface = scenarios["heston_surface"].cold_runner()
    assert second_heston_surface.solver_replay_artifact.request == first_heston_surface.solver_replay_artifact.request
    for key in ("kappa", "theta", "xi", "rho", "v0"):
        assert second_heston_surface.model_parameters[key] == pytest.approx(first_heston_surface.model_parameters[key], abs=1e-8)

    first_local_vol = scenarios["local_vol"].cold_runner()
    second_local_vol = scenarios["local_vol"].cold_runner()
    assert second_local_vol.provenance["calibration_target"] == first_local_vol.provenance["calibration_target"]
    assert second_local_vol.diagnostics.to_payload() == first_local_vol.diagnostics.to_payload()

    first_credit = scenarios["credit"].cold_runner()
    second_credit = scenarios["credit"].cold_runner()
    assert second_credit.solver_replay_artifact.request == first_credit.solver_replay_artifact.request
    assert second_credit.target_hazards == pytest.approx(first_credit.target_hazards, abs=1e-12)
    assert second_credit.model_hazards == pytest.approx(first_credit.model_hazards, abs=1e-12)
    assert second_credit.provenance["potential_binding"] == first_credit.provenance["potential_binding"]


def test_live_supported_calibration_benchmark_report_covers_warm_start_shape():
    from trellis.models.calibration.benchmarking import build_supported_calibration_benchmark_report

    report = build_supported_calibration_benchmark_report(repeats=1, warmups=0)
    cases = {case["workflow"]: case for case in report["cases"]}

    assert report["summary"]["workflow_count"] == 9
    assert report["summary"]["warm_start_workflow_count"] == 4
    assert set(cases) == {
        "hull_white",
        "caplet_strip",
        "sabr",
        "swaption_cube",
        "equity_vol_surface",
        "heston",
        "heston_surface",
        "local_vol",
        "credit",
    }
    assert cases["caplet_strip"]["warm"] is None
    assert cases["equity_vol_surface"]["warm"] is None
    assert cases["local_vol"]["warm"] is None
    assert cases["credit"]["warm"] is None
    assert cases["swaption_cube"]["warm"] is None
    assert cases["hull_white"]["warm"] is not None
    assert cases["sabr"]["warm"] is not None
    assert cases["heston"]["warm"] is not None
    assert cases["heston_surface"]["warm"] is not None
    roles = cases["hull_white"]["metadata"]["multi_curve_roles"]
    assert roles["discount_curve"] == "usd_ois"
    assert roles["forecast_curve"] == "USD-SOFR-3M"
    assert roles["rate_index"] == "USD-SOFR-3M"
    assert cases["caplet_strip"]["metadata"]["surface_name"] == "usd_caplet_strip"
    assert cases["sabr"]["metadata"]["surface_name"] == "usd_rates_smile"
    assert cases["sabr"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["swaption_cube"]["metadata"]["surface_name"] == "usd_swaption_cube"
    assert cases["swaption_cube"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["equity_vol_surface"]["metadata"]["surface_name"] == "spx_surface_authority"
    assert cases["equity_vol_surface"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["heston"]["metadata"]["surface_name"] == "spx_heston_implied_vol"
    assert cases["heston"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["heston_surface"]["metadata"]["surface_name"] == "spx_surface_authority"
    assert cases["heston_surface"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["local_vol"]["metadata"]["source_surface_name"] == "spx_heston_implied_vol"
    assert cases["local_vol"]["metadata"]["surface_name"] == "spx_local_vol"
    assert cases["local_vol"]["metadata"]["synthetic_generation_contract_version"] == "v2"


def test_checked_calibration_benchmark_artifact_covers_supported_workflows():
    payload = json.loads((REPO_ROOT / "docs" / "benchmarks" / "calibration_workflows.json").read_text())
    cases = {case["workflow"]: case for case in payload["cases"]}

    assert payload["benchmark_name"] == "supported_calibration_workflows"
    assert payload["summary"]["workflow_count"] == 9
    assert payload["summary"]["warm_start_workflow_count"] == 4
    assert set(cases) == {
        "hull_white",
        "caplet_strip",
        "sabr",
        "swaption_cube",
        "equity_vol_surface",
        "heston",
        "heston_surface",
        "local_vol",
        "credit",
    }
    assert cases["caplet_strip"]["warm"] is None
    assert cases["equity_vol_surface"]["warm"] is None
    assert cases["local_vol"]["warm"] is None
    assert cases["credit"]["warm"] is None
    assert cases["swaption_cube"]["warm"] is None
    for workflow in ("hull_white", "sabr", "heston", "heston_surface"):
        assert cases[workflow]["warm"] is not None
        assert cases[workflow]["warm_speedup"] > 1.0
    roles = cases["hull_white"]["metadata"]["multi_curve_roles"]
    assert roles["discount_curve"] == "usd_ois"
    assert roles["forecast_curve"] == "USD-SOFR-3M"
    assert roles["rate_index"] == "USD-SOFR-3M"
    assert cases["caplet_strip"]["metadata"]["surface_name"] == "usd_caplet_strip"
    assert cases["sabr"]["metadata"]["surface_name"] == "usd_rates_smile"
    assert cases["sabr"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["swaption_cube"]["metadata"]["surface_name"] == "usd_swaption_cube"
    assert cases["swaption_cube"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["equity_vol_surface"]["metadata"]["surface_name"] == "spx_surface_authority"
    assert cases["equity_vol_surface"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["heston"]["metadata"]["surface_name"] == "spx_heston_implied_vol"
    assert cases["heston"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["heston_surface"]["metadata"]["surface_name"] == "spx_surface_authority"
    assert cases["heston_surface"]["metadata"]["synthetic_generation_contract_version"] == "v2"
    assert cases["local_vol"]["metadata"]["source_surface_name"] == "spx_heston_implied_vol"
    assert cases["local_vol"]["metadata"]["surface_name"] == "spx_local_vol"
    assert cases["local_vol"]["metadata"]["synthetic_generation_contract_version"] == "v2"
