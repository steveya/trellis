"""Tests for the package-level public API surface.

These tests lock down the canonical package entry points introduced in Tranche 2C.
They complement the existing v2 API tests by covering `trellis.core`,
`trellis.models`, and the public docs/metadata surface.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_top_level_package_exports_bootstrap_bundle_helpers():
    import trellis
    from trellis.book import ScenarioResultCube
    from trellis.curves.bootstrap import (
        BootstrapConventionBundle,
        BootstrapCurveInputBundle,
        BootstrapInstrument,
        bootstrap_yield_curve,
    )

    assert trellis.BootstrapConventionBundle is BootstrapConventionBundle
    assert trellis.BootstrapCurveInputBundle is BootstrapCurveInputBundle
    assert trellis.BootstrapInstrument is BootstrapInstrument
    assert trellis.bootstrap_yield_curve is bootstrap_yield_curve
    assert trellis.ScenarioResultCube is ScenarioResultCube


def test_top_level_ask_uses_default_mock_session_and_seeds_vol_surface():
    import trellis
    from trellis.curves.yield_curve import YieldCurve
    from trellis.models.vol_surface import FlatVol
    from trellis.session import Session

    seeded_session = Session(curve=YieldCurve.flat(0.045))
    captured = {}

    def fake_ask_session(query, session, **kwargs):
        captured["query"] = query
        captured["session"] = session
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    with patch("trellis.samples.sample_session", return_value=seeded_session):
        with patch("trellis.agent.ask.ask_session", side_effect=fake_ask_session):
            result = trellis.ask("Price a 5Y cap", model="test-model")

    assert result == {"status": "ok"}
    assert captured["query"] == "Price a 5Y cap"
    assert captured["kwargs"] == {"model": "test-model"}
    assert captured["session"] is not seeded_session
    assert isinstance(captured["session"].vol_surface, FlatVol)
    assert captured["session"].vol_surface.black_vol(1.0, 0.05) == 0.20


def test_core_package_exports():
    import trellis.core as core
    from trellis.core.capabilities import (
        analyze_gap,
        capability_summary,
        check_market_data,
        discover_capabilities,
    )
    from trellis.core.market_state import MarketState, MissingCapabilityError
    from trellis.core.payoff import (
        Cashflows,
        DeterministicCashflowPayoff,
        Payoff,
        PresentValue,
    )
    from trellis.core.state_space import StateSpace
    from trellis.core.types import DayCountConvention, Frequency, PricingResult

    assert core.MarketState is MarketState
    assert core.MissingCapabilityError is MissingCapabilityError
    assert core.Payoff is Payoff
    assert core.DeterministicCashflowPayoff is DeterministicCashflowPayoff
    assert core.Cashflows is Cashflows
    assert core.PresentValue is PresentValue
    assert core.StateSpace is StateSpace
    assert core.Frequency is Frequency
    assert core.DayCountConvention is DayCountConvention
    assert core.PricingResult is PricingResult
    assert core.analyze_gap is analyze_gap
    assert core.check_market_data is check_market_data
    assert core.discover_capabilities is discover_capabilities
    assert core.capability_summary is capability_summary


def test_curves_package_exports_curve_shock_helpers():
    import trellis.curves as curves
    from trellis.curves.credit_curve import CreditCurve
    from trellis.curves.forward_curve import ForwardCurve
    from trellis.curves.scenario_packs import (
        DEFAULT_RATE_SCENARIO_BUCKET_TENORS,
        RateCurveScenario,
        build_rate_curve_scenario_pack,
    )
    from trellis.curves.shocks import (
        CurveShockBucket,
        CurveShockSurface,
        CurveShockWarning,
        build_curve_shock_surface,
    )
    from trellis.curves.yield_curve import YieldCurve

    assert curves.CreditCurve is CreditCurve
    assert curves.ForwardCurve is ForwardCurve
    assert curves.CurveShockBucket is CurveShockBucket
    assert curves.CurveShockSurface is CurveShockSurface
    assert curves.CurveShockWarning is CurveShockWarning
    assert curves.DEFAULT_RATE_SCENARIO_BUCKET_TENORS is DEFAULT_RATE_SCENARIO_BUCKET_TENORS
    assert curves.RateCurveScenario is RateCurveScenario
    assert curves.YieldCurve is YieldCurve
    assert curves.build_curve_shock_surface is build_curve_shock_surface
    assert curves.build_rate_curve_scenario_pack is build_rate_curve_scenario_pack


def test_models_package_exports():
    import trellis.models as models
    from trellis.models.black import (
        black76_call,
        black76_put,
        garman_kohlhagen_call,
        garman_kohlhagen_put,
    )
    from trellis.models.calibration import (
        ConstraintSpec,
        CreditHazardCalibrationQuote,
        CreditHazardCalibrationResult,
        HestonSmileCalibrationResult,
        HestonSmileFitDiagnostics,
        HestonSmilePoint,
        HestonSmileSurface,
        HullWhiteCalibrationInstrument,
        HullWhiteCalibrationResult,
        LocalVolCalibrationResult,
        ObjectiveBundle,
        SABRSmileCalibrationResult,
        SABRSmileFitDiagnostics,
        SABRSmilePoint,
        SABRSmileSurface,
        SolveBackendRecord,
        SolveBackendRegistry,
        RatesCalibrationResult,
        SolveBounds,
        SolveProvenance,
        SolveRequest,
        SolveReplayArtifact,
        SolveResult,
        UnsupportedSolveCapabilityError,
        WarmStart,
        build_heston_smile_surface,
        build_supported_calibration_benchmark_report,
        build_sabr_smile_surface,
        build_solve_provenance,
        build_solve_replay_artifact,
        calibrate_heston_smile_workflow,
        calibrate_hull_white,
        calibrate_local_vol_surface_workflow,
        calibrate_single_name_credit_curve_workflow,
        calibrate_sabr_smile_workflow,
        calibrate_cap_floor_black_vol,
        calibrate_swaption_black_vol,
        dupire_local_vol_result,
        execute_solve_request,
        fit_heston_smile_surface,
        fit_sabr_smile_surface,
        save_calibration_benchmark_report,
        swaption_terms,
    )
    from trellis.models.vol_surface import FlatVol, GridVolSurface, VolSurface
    from trellis.models.vol_surface_shocks import (
        VolSurfaceShockBucket,
        VolSurfaceShockSurface,
        VolSurfaceShockWarning,
        build_vol_surface_shock_surface,
    )

    assert models.black76_call is black76_call
    assert models.black76_put is black76_put
    assert models.garman_kohlhagen_call is garman_kohlhagen_call
    assert models.garman_kohlhagen_put is garman_kohlhagen_put
    assert models.calibration.HestonSmilePoint is HestonSmilePoint
    assert models.calibration.HestonSmileSurface is HestonSmileSurface
    assert models.calibration.HestonSmileFitDiagnostics is HestonSmileFitDiagnostics
    assert models.calibration.HestonSmileCalibrationResult is HestonSmileCalibrationResult
    assert models.calibration.HullWhiteCalibrationInstrument is HullWhiteCalibrationInstrument
    assert models.calibration.HullWhiteCalibrationResult is HullWhiteCalibrationResult
    assert models.calibration.LocalVolCalibrationResult is LocalVolCalibrationResult
    assert models.calibration.CreditHazardCalibrationQuote is CreditHazardCalibrationQuote
    assert models.calibration.CreditHazardCalibrationResult is CreditHazardCalibrationResult
    assert models.calibration.SABRSmilePoint is SABRSmilePoint
    assert models.calibration.SABRSmileSurface is SABRSmileSurface
    assert models.calibration.SABRSmileFitDiagnostics is SABRSmileFitDiagnostics
    assert models.calibration.SABRSmileCalibrationResult is SABRSmileCalibrationResult
    assert models.calibration.RatesCalibrationResult is RatesCalibrationResult
    assert models.calibration.ConstraintSpec is ConstraintSpec
    assert models.calibration.ObjectiveBundle is ObjectiveBundle
    assert models.calibration.SolveBackendRecord is SolveBackendRecord
    assert models.calibration.SolveBackendRegistry is SolveBackendRegistry
    assert models.calibration.SolveBounds is SolveBounds
    assert models.calibration.SolveProvenance is SolveProvenance
    assert models.calibration.SolveRequest is SolveRequest
    assert models.calibration.SolveReplayArtifact is SolveReplayArtifact
    assert models.calibration.SolveResult is SolveResult
    assert models.calibration.UnsupportedSolveCapabilityError is UnsupportedSolveCapabilityError
    assert models.calibration.WarmStart is WarmStart
    assert models.calibration.build_supported_calibration_benchmark_report is build_supported_calibration_benchmark_report
    assert models.calibration.build_solve_provenance is build_solve_provenance
    assert models.calibration.build_solve_replay_artifact is build_solve_replay_artifact
    assert models.calibration.build_heston_smile_surface is build_heston_smile_surface
    assert models.calibration.build_sabr_smile_surface is build_sabr_smile_surface
    assert models.calibration.calibrate_heston_smile_workflow is calibrate_heston_smile_workflow
    assert models.calibration.calibrate_hull_white is calibrate_hull_white
    assert models.calibration.calibrate_local_vol_surface_workflow is calibrate_local_vol_surface_workflow
    assert (
        models.calibration.calibrate_single_name_credit_curve_workflow
        is calibrate_single_name_credit_curve_workflow
    )
    assert models.calibration.calibrate_sabr_smile_workflow is calibrate_sabr_smile_workflow
    assert models.calibration.calibrate_cap_floor_black_vol is calibrate_cap_floor_black_vol
    assert models.calibration.calibrate_swaption_black_vol is calibrate_swaption_black_vol
    assert models.calibration.dupire_local_vol_result is dupire_local_vol_result
    assert models.calibration.execute_solve_request is execute_solve_request
    assert models.calibration.fit_heston_smile_surface is fit_heston_smile_surface
    assert models.calibration.fit_sabr_smile_surface is fit_sabr_smile_surface
    assert models.calibration.save_calibration_benchmark_report is save_calibration_benchmark_report
    assert models.calibration.swaption_terms is swaption_terms
    assert models.FlatVol is FlatVol
    assert models.GridVolSurface is GridVolSurface
    assert models.VolSurface is VolSurface
    assert models.VolSurfaceShockBucket is VolSurfaceShockBucket
    assert models.VolSurfaceShockSurface is VolSurfaceShockSurface
    assert models.VolSurfaceShockWarning is VolSurfaceShockWarning
    assert models.build_vol_surface_shock_surface is build_vol_surface_shock_surface

    for name in (
        "analytical",
        "trees",
        "monte_carlo",
        "qmc",
        "pde",
        "transforms",
        "processes",
        "copulas",
        "calibration",
        "cashflow_engine",
        "vol_surface_shocks",
    ):
        assert hasattr(models, name), f"trellis.models missing `{name}` package export"


def test_analytical_package_exports_quanto_helpers():
    import trellis.models.analytical as analytical
    from trellis.models.analytical.quanto import (
        price_quanto_option_analytical,
        price_quanto_option_raw,
    )

    assert analytical.price_quanto_option_analytical is price_quanto_option_analytical
    assert analytical.price_quanto_option_raw is price_quanto_option_raw


def test_family_package_exports_are_canonical():
    import trellis.models.monte_carlo as monte_carlo
    import trellis.models.pde as pde
    import trellis.models.processes as processes
    import trellis.models.qmc as qmc
    import trellis.models.transforms as transforms
    import trellis.models.trees as trees
    from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
    from trellis.models.monte_carlo.variance_reduction import sobol_normals
    from trellis.models.processes.heston import (
        Heston,
        HestonRuntimeBinding,
        build_heston_parameter_payload,
        resolve_heston_runtime_binding,
    )

    assert hasattr(trees, "BinomialTree")
    assert hasattr(trees, "TrinomialTree")
    assert hasattr(trees, "backward_induction")

    assert hasattr(monte_carlo, "MonteCarloEngine")
    assert hasattr(monte_carlo, "euler_maruyama")
    assert hasattr(monte_carlo, "milstein")

    assert qmc.sobol_normals is sobol_normals
    assert qmc.brownian_bridge is brownian_bridge

    assert hasattr(pde, "theta_method_1d")
    assert hasattr(pde, "crank_nicolson_1d")
    assert hasattr(pde, "implicit_fd_1d")
    assert hasattr(pde, "Grid")

    assert hasattr(transforms, "fft_price")
    assert hasattr(transforms, "cos_price")
    assert processes.Heston is Heston
    assert processes.HestonRuntimeBinding is HestonRuntimeBinding
    assert processes.build_heston_parameter_payload is build_heston_parameter_payload
    assert processes.resolve_heston_runtime_binding is resolve_heston_runtime_binding


def test_models_docs_use_package_level_entry_points():
    text = (REPO_ROOT / "docs" / "api" / "models.rst").read_text()
    assert "trellis.models.pde.theta_method_1d" in text
    assert "trellis.models.pde.crank_nicolson.crank_nicolson_1d" not in text
    assert "trellis.models.pde.implicit_fd.implicit_fd_1d" not in text
    assert "trellis.models.GridVolSurface" in text
    assert "trellis.models.trees.BinomialTree" in text
    assert "trellis.models.monte_carlo.MonteCarloEngine" in text
    assert "trellis.models.qmc.sobol_normals" in text
    assert "trellis.models.qmc.brownian_bridge" in text
    assert "trellis.models.transforms.fft_price" in text


def test_core_docs_use_package_level_entry_points():
    text = (REPO_ROOT / "docs" / "api" / "core.rst").read_text()
    assert "trellis.core.MarketState" in text
    assert "trellis.core.Payoff" in text
    assert "trellis.core.Frequency" in text
    assert "trellis.core.capability_summary" in text


def test_readme_uses_trellis_name():
    first_line = (REPO_ROOT / "README.md").read_text().splitlines()[0].strip().lower()
    assert first_line == "# trellis"


def test_setup_metadata_uses_trellis_name():
    setup_text = (REPO_ROOT / "setup.py").read_text()
    assert 'name="trellis"' in setup_text or "name='trellis'" in setup_text


def test_migration_notes_cover_qmc_canonical_path():
    doc_path = REPO_ROOT / "docs" / "migration_notes.md"
    if not doc_path.exists():
        doc_path = REPO_ROOT / "docs" / "mathematical" / "monte_carlo.rst"
    text = doc_path.read_text()
    assert "trellis.models.qmc" in text
    assert (
        "trellis.models.monte_carlo.variance_reduction" in text
        or "trellis.models.monte_carlo" in text
    )
    assert (
        "trellis.models.monte_carlo.brownian_bridge" in text
        or "trellis.models.qmc.brownian_bridge" in text
    )
