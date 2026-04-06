"""Tests for trellis.pipeline — Pipeline (declarative batch)."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from trellis.book import Book, BookResult, ScenarioResultCube
from trellis.conventions.day_count import DayCountConvention
from trellis.core.types import PricingResult
from trellis.core.types import Frequency
from trellis.curves.bootstrap import (
    BootstrapConventionBundle,
    BootstrapCurveInputBundle,
    BootstrapInstrument,
    bootstrap_curve_result,
)
from trellis.data.schema import MarketSnapshot
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.pipeline import Pipeline


def _curve():
    return YieldCurve.flat(0.045)


def _book():
    bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                maturity=10, frequency=2)
    return Book({"10Y": bond})


def _snapshot():
    return MarketSnapshot(
        as_of=date(2024, 11, 15),
        source="unit",
        discount_curves={
            "usd_ois": YieldCurve.flat(0.045),
            "eur_ois": YieldCurve.flat(0.025),
        },
        vol_surfaces={
            "atm": FlatVol(0.20),
            "smile": GridVolSurface(
                expiries=(1.0, 2.0),
                strikes=(90.0, 110.0),
                vols=((0.25, 0.22), (0.27, 0.24)),
            ),
        },
        default_discount_curve="usd_ois",
        default_vol_surface="atm",
    )


def _bootstrapped_snapshot():
    bundle = BootstrapCurveInputBundle(
        curve_name="usd_ois_boot",
        currency="USD",
        rate_index="USD-SOFR-3M",
        conventions=BootstrapConventionBundle(
            deposit_day_count=DayCountConvention.ACT_360,
            swap_fixed_frequency=Frequency.ANNUAL,
            swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
            swap_float_frequency=Frequency.QUARTERLY,
            swap_float_day_count=DayCountConvention.ACT_360,
        ),
        instruments=(
            BootstrapInstrument(tenor=1.0, quote=0.045, instrument_type="deposit", label="DEP1Y"),
            BootstrapInstrument(tenor=2.0, quote=0.046, instrument_type="swap", label="SWAP2Y"),
            BootstrapInstrument(tenor=5.0, quote=0.0475, instrument_type="swap", label="SWAP5Y"),
            BootstrapInstrument(tenor=10.0, quote=0.0485, instrument_type="swap", label="SWAP10Y"),
        ),
    )
    result = bootstrap_curve_result(bundle, max_iter=75, tol=1e-12)
    return MarketSnapshot(
        as_of=date(2024, 11, 15),
        source="unit",
        discount_curves={"usd_ois_boot": result.curve},
        default_discount_curve="usd_ois_boot",
        provenance={
            "source": "unit",
            "source_kind": "mixed",
            "bootstrap_inputs": {"discount_curves": {"usd_ois_boot": bundle.to_payload()}},
            "bootstrap_runs": {"discount_curves": {"usd_ois_boot": result.to_payload()}},
        },
    )


class TestPipeline:

    def test_basic_run(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .run()
        )
        assert "base" in results
        assert isinstance(results["base"], BookResult)
        assert results["base"].total_mv > 0

    def test_multiple_scenarios(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios([
                {"name": "base", "shift_bps": 0},
                {"name": "up100", "shift_bps": 100},
            ])
            .run()
        )
        assert "base" in results
        assert "up100" in results
        assert results["up100"].total_mv < results["base"].total_mv

    def test_run_returns_scenario_result_cube_with_aggregation_metadata(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios([
                {"name": "base", "shift_bps": 0},
                {"name": "up100", "shift_bps": 100},
            ])
            .run()
        )

        assert isinstance(results, ScenarioResultCube)
        assert results.scenario_specs["up100"]["shift_bps"] == pytest.approx(100.0)
        ladder = results.book_ladder("total_mv")
        assert ladder["up100"] < ladder["base"]
        assert ladder.metadata["scenario_provenance"]["up100"]["data_source"] == "treasury_gov"

    def test_named_scenario_pack_expands_into_twist_templates(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios(
                [
                    {
                        "scenario_pack": "twist",
                        "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                        "amplitude_bps": 25.0,
                    }
                ]
            )
            .run()
        )

        assert "twist_steepener_25bp" in results
        assert "twist_flattener_25bp" in results
        assert (
            results.scenario_provenance["twist_steepener_25bp"]["expanded_from"]["scenario_pack"]
            == "twist"
        )

    def test_named_scenario_pack_supports_off_grid_bucket_workflow(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=YieldCurve([1.0, 5.0, 10.0, 30.0], [0.04, 0.042, 0.045, 0.047]))
            .scenarios(
                [
                    {
                        "scenario_pack": "twist",
                        "bucket_tenors": (2.0, 7.0, 10.0, 30.0),
                        "amplitude_bps": 25.0,
                    }
                ]
            )
            .run()
        )

        assert "twist_steepener_25bp" in results
        assert "twist_flattener_25bp" in results
        assert results["twist_steepener_25bp"].total_mv != pytest.approx(
            results["twist_flattener_25bp"].total_mv
        )

    def test_compute_price_only(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .compute(["price"])
            .run()
        )
        br = results["base"]
        assert br["10Y"].greeks == {}

    def test_compute_selective_greeks(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .compute(["price", "dv01"])
            .run()
        )
        br = results["base"]
        assert "dv01" in br["10Y"].greeks
        assert "convexity" not in br["10Y"].greeks

    def test_missing_instruments_raises(self):
        with pytest.raises(ValueError, match="No instruments"):
            Pipeline().market_data(curve=_curve()).run()

    def test_output_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_template = str(Path(tmpdir) / "{scenario}.csv")
            (
                Pipeline()
                .instruments(_book())
                .market_data(curve=_curve())
                .output_csv(path_template)
                .run()
            )
            assert Path(tmpdir, "base.csv").exists()

    def test_output_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_template = str(Path(tmpdir) / "{scenario}.parquet")
            (
                Pipeline()
                .instruments(_book())
                .market_data(curve=_curve())
                .output_parquet(path_template)
                .run()
            )
            assert Path(tmpdir, "base.parquet").exists()

    def test_mock_source_no_mocking_needed(self):
        """Full offline pipeline — no patches, no network."""
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(source="mock", as_of="2024-11-15")
            .run()
        )
        assert results["base"].total_mv > 0

    def test_market_snapshot_input(self):
        snapshot = _snapshot()
        usd_results = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=snapshot, discount_curve="usd_ois")
            .run()
        )
        eur_results = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=snapshot, discount_curve="eur_ois")
            .run()
        )
        assert eur_results["base"].total_mv > usd_results["base"].total_mv

    def test_market_snapshot_accepts_named_vol_surface(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=_snapshot(), vol_surface_name="smile")
            .run()
        )
        assert results["base"].total_mv > 0

    def test_compile_compute_plan_expands_scenarios_and_preserves_market_metadata(self):
        pipeline = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=_snapshot(), discount_curve="eur_ois", vol_surface_name="smile")
            .compute(["price", "dv01"])
            .scenarios(
                [
                    {"name": "base", "shift_bps": 0},
                    {
                        "scenario_pack": "twist",
                        "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                        "amplitude_bps": 25.0,
                    },
                ]
            )
        )

        plan = pipeline.compile_compute_plan()

        assert plan.to_dict()["plan_type"] == "book_scenario_batch"
        assert plan.to_dict()["scenario_count"] == 3
        assert plan.to_dict()["discount_curve_name"] == "eur_ois"
        assert plan.to_dict()["vol_surface_name"] == "smile"
        assert plan.to_dict()["measures"] == ["price", "dv01"]
        assert plan.to_dict()["scenarios"][1]["expanded_from"]["scenario_pack"] == "twist"

    def test_run_attaches_reusable_compute_plan_to_scenario_result_cube(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios([
                {"name": "base", "shift_bps": 0},
                {"name": "up100", "shift_bps": 100},
            ])
            .run()
        )

        assert results.compute_plan["plan_type"] == "book_scenario_batch"
        assert results.compute_plan["scenario_count"] == 2
        assert results.compute_plan["scenarios"][1]["name"] == "up100"

    def test_saved_scenario_template_expands_into_named_batch_plan(self):
        snapshot = _snapshot()
        snapshot = MarketSnapshot(
            as_of=snapshot.as_of,
            source=snapshot.source,
            discount_curves=snapshot.discount_curves,
            vol_surfaces=snapshot.vol_surfaces,
            default_discount_curve=snapshot.default_discount_curve,
            default_vol_surface=snapshot.default_vol_surface,
            metadata={
                "scenario_templates": {
                    "desk_twist": {
                        "scenario_pack": "twist",
                        "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                        "amplitude_bps": 25.0,
                    }
                }
            },
        )
        plan = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=snapshot)
            .scenarios([{"scenario_template": "desk_twist"}])
            .compile_compute_plan()
        )

        assert plan.to_dict()["scenario_count"] == 2
        assert plan.to_dict()["scenarios"][0]["expanded_from"]["scenario_template"] == "desk_twist"
        assert "twist_steepener_25bp" in {
            scenario["name"] for scenario in plan.to_dict()["scenarios"]
        }

    def test_saved_rebuild_scenario_template_replays_quote_space_methodology(self):
        snapshot = _bootstrapped_snapshot()
        snapshot = MarketSnapshot(
            as_of=snapshot.as_of,
            source=snapshot.source,
            discount_curves=snapshot.discount_curves,
            default_discount_curve=snapshot.default_discount_curve,
            provenance=snapshot.provenance,
            metadata={
                "scenario_templates": {
                    "desk_twist_steepener": {
                        "name": "twist_steepener_25bp",
                        "scenario_pack": "twist",
                        "description": "Short-end down and long-end up in quote space.",
                        "methodology": "curve_rebuild",
                        "bucket_convention": "bootstrap_quote",
                        "selected_curve_name": "usd_ois_boot",
                        "bucket_tenors": (1.0, 2.0, 5.0, 10.0),
                        "tenor_bumps": {1.0: -25.0, 2.0: -12.5, 5.0: 8.3333333333, 10.0: 25.0},
                        "quote_bucket_bumps": {
                            "DEP1Y": -25.0,
                            "SWAP2Y": -12.5,
                            "SWAP5Y": 8.3333333333,
                            "SWAP10Y": 25.0,
                        },
                    }
                }
            },
        )

        plan = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=snapshot)
            .scenarios([{"scenario_template": "desk_twist_steepener"}])
            .compile_compute_plan()
        )

        scenario = plan.to_dict()["scenarios"][0]
        assert plan.to_dict()["scenario_count"] == 1
        assert scenario["name"] == "twist_steepener_25bp"
        assert scenario["methodology"] == "curve_rebuild"
        assert scenario["bucket_convention"] == "bootstrap_quote"
        assert scenario["quote_bucket_bumps"]["DEP1Y"] == pytest.approx(-25.0)

        results = plan.execute()

        assert results["twist_steepener_25bp"].total_mv > 0.0
        assert (
            results.scenario_provenance["twist_steepener_25bp"]["scenario_spec"]["methodology"]
            == "curve_rebuild"
        )

    def test_compile_compute_plan_accepts_float_shift_names_and_stable_tenor_bump_names(self):
        plan = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios(
                [
                    {"shift_bps": 25.0},
                    {"tenor_bumps": {2.0: 10.0, 10.0: -5.5}},
                ]
            )
            .compile_compute_plan()
        )

        scenarios = plan.to_dict()["scenarios"]
        assert scenarios[0]["name"] == "shift_p25bp"
        assert scenarios[1]["name"] == "tenor_bump_2y_p10bp_10y_m5p5bp"

    def test_run_projects_executor_results_without_calling_session_price(self):
        from trellis.platform.results import ExecutionResult

        pipeline = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios([
                {"name": "base", "shift_bps": 0},
                {"name": "up100", "shift_bps": 100},
            ])
        )
        executor_results = iter(
            (
                BookResult(
                    {
                        "10Y": PricingResult(
                            clean_price=100.0,
                            dirty_price=100.0,
                            accrued_interest=0.0,
                            greeks={"dv01": 0.10},
                            curve_sensitivities={},
                        )
                    },
                    _book(),
                ),
                BookResult(
                    {
                        "10Y": PricingResult(
                            clean_price=95.0,
                            dirty_price=95.0,
                            accrued_interest=0.0,
                            greeks={"dv01": 0.09},
                            curve_sensitivities={},
                        )
                    },
                    _book(),
                ),
            )
        )
        calls = []

        def fake_execute(compiled_request, execution_context, *, handlers=None):
            calls.append(compiled_request.execution_plan.action)
            result = next(executor_results)
            return ExecutionResult(
                run_id=f"run_pipeline_{len(calls)}",
                request_id=compiled_request.request.request_id,
                status="succeeded",
                action=compiled_request.execution_plan.action,
                output_mode=execution_context.default_output_mode,
                result_payload={"result": result},
                provenance={"run_mode": execution_context.run_mode.value},
                policy_outcome={"allowed": True},
            )

        with patch("trellis.platform.executor.execute_compiled_request", side_effect=fake_execute):
            with patch("trellis.session.Session.price", side_effect=AssertionError("pipeline should execute through the platform core")):
                results = pipeline.run()

        assert calls == ["price_book", "price_book"]
        assert results["base"].total_mv == pytest.approx(100.0)
        assert results["up100"].total_mv == pytest.approx(95.0)

    def test_run_failure_records_platform_trace(self):
        from trellis.platform.results import ExecutionResult

        pipeline = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
        )
        recorded = []

        def fake_execute(compiled_request, execution_context, *, handlers=None):
            return ExecutionResult(
                run_id="run_pipeline_failed",
                request_id=compiled_request.request.request_id,
                status="failed",
                action=compiled_request.execution_plan.action,
                output_mode=execution_context.default_output_mode,
                result_payload={
                    "error_type": "ValueError",
                    "error": "boom",
                },
                provenance={"run_mode": execution_context.run_mode.value},
                policy_outcome={"allowed": True},
            )

        def fake_record(compiled_request, *, success, outcome, details=None, root=None):
            recorded.append(
                {
                    "request_id": compiled_request.request.request_id,
                    "success": success,
                    "outcome": outcome,
                    "details": details,
                }
            )

        with patch("trellis.platform.executor.execute_compiled_request", side_effect=fake_execute):
            with patch("trellis.agent.platform_traces.record_platform_trace", side_effect=fake_record):
                with pytest.raises(ValueError, match="Governed pipeline execution failed."):
                    pipeline.run()

        assert recorded
        assert recorded[-1]["success"] is False
        assert recorded[-1]["outcome"] == "pipeline_failed"
        assert recorded[-1]["details"]["run_id"] == "run_pipeline_failed"

    def test_run_delegates_scenarios_to_session_shared_governed_runner(self):
        pipeline = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios([
                {"name": "base", "shift_bps": 0},
                {"name": "up100", "shift_bps": 100},
            ])
        )
        projected_results = iter(
            (
                BookResult(
                    {
                        "10Y": PricingResult(
                            clean_price=100.0,
                            dirty_price=100.0,
                            accrued_interest=0.0,
                            greeks={},
                            curve_sensitivities={},
                        )
                    },
                    _book(),
                ),
                BookResult(
                    {
                        "10Y": PricingResult(
                            clean_price=95.0,
                            dirty_price=95.0,
                            accrued_interest=0.0,
                            greeks={},
                            curve_sensitivities={},
                        )
                    },
                    _book(),
                ),
            )
        )
        calls = []

        def fake_run(session, **kwargs):
            calls.append(kwargs)
            return next(projected_results)

        with patch("trellis.pipeline.Session._run_governed_request", autospec=True, side_effect=fake_run):
            with patch("trellis.pipeline.Session._compile_platform_request", side_effect=AssertionError("pipeline should delegate through Session._run_governed_request")):
                results = pipeline.run()

        assert results["base"].total_mv == pytest.approx(100.0)
        assert results["up100"].total_mv == pytest.approx(95.0)
        assert [call["request_type"] for call in calls] == ["price", "price"]
        assert [call["success_outcome"] for call in calls] == ["pipeline_priced", "pipeline_priced"]
        assert [call["failure_outcome"] for call in calls] == ["pipeline_failed", "pipeline_failed"]
        assert all(call["book"] == pipeline._book for call in calls)

    def test_pipeline_suite_is_marked_global_workflow(self, request):
        assert request.node.get_closest_marker("global_workflow") is not None
