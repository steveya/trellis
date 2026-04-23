"""Tests for repaired equity-vol surface calibration and staged fits."""

from datetime import date

import numpy as raw_np
import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.data.resolver import resolve_market_snapshot
from trellis.models.calibration.equity_vol_surface import (
    EquityVolStageComparisonResult,
    EquityVolSurfaceStageComparisonResult,
    EquityVolSurfaceAuthorityResult,
    EquityVolQuoteCleaningResult,
    calibrate_equity_vol_surface_workflow,
    clean_equity_vol_surface_quotes,
    calibrate_local_vol_surface_from_equity_vol_surface_workflow,
    compare_heston_to_equity_vol_surface_workflow,
    compare_heston_surface_to_equity_vol_surface_workflow,
)
from trellis.models.calibration.heston_fit import HestonSurfaceCalibrationResult, calibrate_heston_surface_from_equity_vol_surface_workflow
from trellis.models.calibration.local_vol import LocalVolCalibrationResult


SETTLE = date(2024, 11, 15)


def _mock_spx_surface_inputs():
    snapshot = resolve_market_snapshot(as_of=SETTLE, source="mock")
    surface = snapshot.vol_surfaces["spx_heston_implied_vol"]
    spot = float(snapshot.underlier_spots["SPX"])
    rate = float(snapshot.discount_curve("usd_ois").zero_rate(1.0))
    return snapshot, surface, spot, rate


def test_calibrate_equity_vol_surface_workflow_repairs_smile_violations_and_materializes():
    _, surface, spot, rate = _mock_spx_surface_inputs()
    distorted_vols = raw_np.asarray(surface.vols, dtype=float).copy()
    distorted_vols[1, 2] += 0.06

    result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        distorted_vols,
        rate=rate,
        surface_name="spx_repaired_surface",
    )
    market_state = MarketState(
        as_of=SETTLE,
        settlement=SETTLE,
        spot=spot,
        discount=YieldCurve.flat(rate),
    )
    enriched_state = result.apply_to_market_state(market_state)

    assert isinstance(result, EquityVolSurfaceAuthorityResult)
    assert result.summary["surface_model_family"] == "raw_svi_surface"
    assert result.quote_cleaning is not None
    assert result.quote_cleaning.diagnostics.adjusted_point_count >= 1
    assert result.diagnostics.repaired_smile_violation_count <= result.diagnostics.raw_smile_violation_count
    assert result.diagnostics.repaired_calendar_violation_count <= result.diagnostics.raw_calendar_violation_count
    assert result.vol_surface.black_vol(1.0, spot) > 0.0
    record = enriched_state.materialized_calibrated_object(object_kind="black_vol_surface")
    assert record is not None
    assert record["object_name"] == "spx_repaired_surface"
    assert record["metadata"]["surface_model_family"] == "raw_svi_surface"


def test_local_vol_from_equity_surface_uses_repaired_surface_provenance():
    _, surface, spot, rate = _mock_spx_surface_inputs()
    authority_result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        surface.vols,
        rate=rate,
        surface_name="spx_surface_authority",
    )

    local_result = calibrate_local_vol_surface_from_equity_vol_surface_workflow(
        authority_result,
        surface_name="spx_local_from_repaired_surface",
    )

    assert isinstance(local_result, LocalVolCalibrationResult)
    assert local_result.calibration_target["source_surface_name"] == "spx_surface_authority"
    assert local_result.calibration_target["source_surface_kind"] == "repaired_equity_vol_surface"
    assert local_result.summary["source_surface_kind"] == "repaired_equity_vol_surface"
    assert local_result.local_vol_surface(spot, 1.0) > 0.0


def test_equity_vol_quote_cleaning_flags_and_trims_outlier_before_surface_fit():
    _, surface, spot, rate = _mock_spx_surface_inputs()
    distorted_vols = raw_np.asarray(surface.vols, dtype=float).copy()
    distorted_vols[2, 1] += 0.18

    cleaning_result = clean_equity_vol_surface_quotes(
        spot,
        surface.expiries,
        surface.strikes,
        distorted_vols,
        rate=rate,
        surface_name="spx_dirty_surface",
    )

    assert isinstance(cleaning_result, EquityVolQuoteCleaningResult)
    assert cleaning_result.diagnostics.adjusted_point_count >= 1
    assert cleaning_result.cleaned_surface.market_vols[2][1] < distorted_vols[2, 1]
    assert cleaning_result.diagnostics.cleaned_smile_violation_count <= cleaning_result.diagnostics.raw_smile_violation_count
    assert cleaning_result.summary["cleaning_stage"] == "quote_governance"


def test_compare_heston_to_equity_surface_reports_staged_errors():
    snapshot, surface, spot, rate = _mock_spx_surface_inputs()
    authority_result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        surface.vols,
        rate=rate,
        surface_name="spx_surface_authority",
    )
    model_pack = snapshot.provenance["prior_parameters"]["synthetic_generation_contract"]["model_packs"]["volatility"]["model_parameter_sets"]["heston_equity"]

    comparison = compare_heston_to_equity_vol_surface_workflow(
        authority_result,
        expiry_years=1.0,
        parameter_set_name="heston_equity_stage",
        warm_start=(
            float(model_pack["kappa"]),
            float(model_pack["theta"]),
            float(model_pack["xi"]),
            float(model_pack["rho"]),
            float(model_pack["v0"]),
        ),
    )

    assert isinstance(comparison, EquityVolStageComparisonResult)
    assert comparison.summary["expiry_years"] == pytest.approx(1.0)
    assert comparison.surface_max_abs_vol_error < 0.03
    assert comparison.model_max_abs_vol_error < 0.02
    assert comparison.preferred_stage in {"surface_authority", "model_fit"}
    assert comparison.heston_result.parameter_set_name == "heston_equity_stage"


def test_heston_surface_fit_from_equity_surface_is_replayable_and_market_state_ready():
    snapshot, surface, spot, rate = _mock_spx_surface_inputs()
    authority_result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        surface.vols,
        rate=rate,
        surface_name="spx_surface_authority",
    )
    model_pack = snapshot.provenance["prior_parameters"]["synthetic_generation_contract"]["model_packs"]["volatility"]["model_parameter_sets"]["heston_equity"]

    result = calibrate_heston_surface_from_equity_vol_surface_workflow(
        authority_result,
        parameter_set_name="heston_equity_surface_stage",
        warm_start=(
            float(model_pack["kappa"]),
            float(model_pack["theta"]),
            float(model_pack["xi"]),
            float(model_pack["rho"]),
            float(model_pack["v0"]),
        ),
    )

    assert isinstance(result, HestonSurfaceCalibrationResult)
    assert result.solve_request.request_id == "heston_surface_least_squares"
    assert result.provenance["source_ref"] == "calibrate_heston_surface_from_equity_vol_surface_workflow"
    assert result.summary["surface_name"] == "spx_surface_authority"
    assert result.summary["expiry_count"] == len(surface.expiries)
    assert result.diagnostics.max_abs_vol_error < 0.03


def test_compare_heston_surface_to_equity_surface_reports_full_surface_stage_metrics():
    snapshot, surface, spot, rate = _mock_spx_surface_inputs()
    authority_result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        surface.vols,
        rate=rate,
        surface_name="spx_surface_authority",
    )
    model_pack = snapshot.provenance["prior_parameters"]["synthetic_generation_contract"]["model_packs"]["volatility"]["model_parameter_sets"]["heston_equity"]

    comparison = compare_heston_surface_to_equity_vol_surface_workflow(
        authority_result,
        parameter_set_name="heston_equity_surface_stage",
        warm_start=(
            float(model_pack["kappa"]),
            float(model_pack["theta"]),
            float(model_pack["xi"]),
            float(model_pack["rho"]),
            float(model_pack["v0"]),
        ),
    )

    assert isinstance(comparison, EquityVolSurfaceStageComparisonResult)
    assert comparison.summary["point_count"] == len(surface.expiries) * len(surface.strikes)
    assert comparison.surface_max_abs_vol_error < 0.03
    assert comparison.model_max_abs_vol_error < 0.03
    assert comparison.preferred_stage in {"surface_authority", "model_fit"}


def test_equity_vol_surface_fit_is_stable_under_small_quote_perturbations():
    _, surface, spot, rate = _mock_spx_surface_inputs()
    base_result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        surface.vols,
        rate=rate,
        surface_name="spx_surface_authority_base",
    )
    perturbation = raw_np.asarray(
        [
            [0.0000, 0.0008, -0.0005, 0.0006, 0.0000],
            [0.0000, 0.0010, -0.0008, 0.0009, 0.0000],
            [0.0000, 0.0011, -0.0009, 0.0010, 0.0000],
            [0.0000, 0.0012, -0.0010, 0.0011, 0.0000],
            [0.0000, 0.0013, -0.0011, 0.0012, 0.0000],
        ],
        dtype=float,
    )
    perturbed_vols = raw_np.asarray(surface.vols, dtype=float) + perturbation
    perturbed_result = calibrate_equity_vol_surface_workflow(
        spot,
        surface.expiries,
        surface.strikes,
        perturbed_vols,
        rate=rate,
        surface_name="spx_surface_authority_perturbed",
    )

    base_grid = raw_np.asarray(base_result.repaired_vols, dtype=float)
    perturbed_grid = raw_np.asarray(perturbed_result.repaired_vols, dtype=float)
    assert float(raw_np.max(raw_np.abs(base_grid - perturbed_grid))) < 0.02
