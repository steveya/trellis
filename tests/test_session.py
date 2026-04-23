"""Tests for trellis.session — Session (immutable market snapshot)."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from trellis.book import Book, BookResult
from trellis.conventions.day_count import DayCountConvention
from trellis.core.types import Frequency
from trellis.curves.bootstrap import (
    BootstrapConventionBundle,
    BootstrapCurveInputBundle,
    BootstrapInstrument,
    bootstrap_curve_result,
)
from trellis.data.schema import MarketSnapshot
from trellis.core.types import PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.session import Session


def _curve():
    return YieldCurve.flat(0.045)


def _bond():
    return Bond(face=100, coupon=0.045, maturity_date=date(2034, 11, 15),
                maturity=10, frequency=2)


SETTLE = date(2024, 11, 15)


class _VolPointPayoff:
    requirements = {"black_vol_surface"}

    def __init__(self, expiry: float, strike: float):
        self.expiry = float(expiry)
        self.strike = float(strike)

    def evaluate(self, market_state):
        return float(market_state.vol_surface.black_vol(self.expiry, self.strike))


class _TraceSafeVolPointPayoff:
    requirements = {"black_vol_surface"}

    def __init__(self, expiry: float, strike: float):
        self.expiry = float(expiry)
        self.strike = float(strike)

    def evaluate(self, market_state):
        return market_state.vol_surface.black_vol(self.expiry, self.strike)


class _SpotQuadraticPayoff:
    requirements = {"spot"}

    def evaluate(self, market_state):
        spot = float(market_state.spot)
        return spot**2 + 0.5 * spot


class _LinearTimeDecayPayoff:
    requirements = set()

    def __init__(self, expiry_date: date):
        self.expiry_date = expiry_date

    def evaluate(self, market_state):
        return float((self.expiry_date - market_state.settlement).days)


class _ShiftableFlatCurve:
    def __init__(self, rate: float):
        self.rate = float(rate)

    def zero_rate(self, t: float) -> float:
        return self.rate

    def discount(self, t: float) -> float:
        return float(np.exp(-self.rate * float(t)))

    def shift(self, bps: float):
        return _ShiftableFlatCurve(self.rate + float(bps) / 10_000.0)

    def bump(self, tenor_bumps: dict[float, float]):
        parallel_bps = next(iter((tenor_bumps or {}).values()), 0.0)
        return self.shift(parallel_bps)


class _RepresentativeVolSurface:
    def __init__(self, base_vol: float):
        self.base_vol = float(base_vol)

    def black_vol(self, expiry: float, strike: float) -> float:
        return self.base_vol + 0.01 * float(expiry) + 0.0001 * float(strike)


def _snapshot():
    usd = YieldCurve.flat(0.045)
    eur = YieldCurve.flat(0.025)
    return MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": usd, "eur_ois": eur},
        forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.05)},
        vol_surfaces={
            "usd_atm": FlatVol(0.20),
            "usd_smile": GridVolSurface(
                expiries=(1.0, 2.0),
                strikes=(90.0, 110.0),
                vols=((0.25, 0.22), (0.27, 0.24)),
            ),
        },
        default_discount_curve="usd_ois",
        default_vol_surface="usd_atm",
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
        as_of=SETTLE,
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


def _spot_snapshot():
    return MarketSnapshot(
        as_of=SETTLE,
        source="unit",
        discount_curves={"usd_ois": YieldCurve.flat(0.045)},
        underlier_spots={"AAPL": 100.0},
        default_discount_curve="usd_ois",
        default_underlier_spot="AAPL",
    )


def _callable_bond_spec():
    from trellis.instruments.callable_bond import CallableBondSpec

    return CallableBondSpec(
        notional=100.0,
        coupon=0.05,
        start_date=date(2025, 1, 15),
        end_date=date(2035, 1, 15),
        call_dates=[date(2028, 1, 15), date(2030, 1, 15), date(2032, 1, 15)],
        call_price=100.0,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )


class TestSessionPricing:

    def test_price_bond(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        r = s.price(_bond())
        assert isinstance(r, PricingResult)
        assert 80 < r.clean_price < 120
        assert r.curve_sensitivities

    def test_greeks_none_returns_empty(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        r = s.price(_bond(), greeks=None)
        assert r.greeks == {}

    def test_greeks_selective(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        r = s.price(_bond(), greeks=["dv01"])
        assert "dv01" in r.greeks
        assert "convexity" not in r.greeks

    def test_price_book(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        book = Book({"10Y": _bond()})
        br = s.price(book)
        assert isinstance(br, BookResult)
        assert br.book_dv01 > 0

    def test_greeks_method(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        g = s.greeks(_bond())
        assert "dv01" in g
        assert g["dv01"] > 0

    def test_price_krd_exposes_zero_curve_methodology_metadata(self):
        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)

        result = s.price(_bond(), greeks="all")

        metadata = result.greeks["key_rate_durations"].metadata
        assert metadata["resolved_methodology"] == "zero_curve"
        assert metadata["resolved_derivative_method"] == "autodiff_public_curve"
        assert metadata["bucket_convention"] == "curve_tenor"
        assert metadata["bucket_tenors"] == [1.0, 2.0, 5.0, 10.0]

    def test_analyze_uses_autodiff_curve_sensitivities(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=["price", "dv01", "duration", "convexity", "key_rate_durations"],
        )
        assert result.price > 0
        assert result.dv01 > 0
        assert result.convexity > 0
        assert result.dv01.metadata["resolved_derivative_method"] == "autodiff_public_curve"
        assert result.duration.metadata["resolved_derivative_method"] == "autodiff_public_curve"
        assert result.convexity.metadata["resolved_derivative_method"] == "autodiff_public_curve"
        assert result.key_rate_durations.metadata["resolved_derivative_method"] == "autodiff_public_curve"
        assert sum(result.key_rate_durations.values()) == pytest.approx(result.duration, rel=0.05)

    def test_analyze_rate_sensitivities_record_parallel_bump_fallback_provenance(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        s = Session(curve=_ShiftableFlatCurve(0.045), settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=["dv01", "duration", "convexity"],
        )

        assert result.dv01 > 0.0
        assert result.dv01.metadata["resolved_derivative_method"] == "parallel_curve_bump"
        assert result.duration.metadata["resolved_derivative_method"] == "parallel_curve_bump"
        assert result.convexity.metadata["resolved_derivative_method"] == "parallel_curve_bump"
        assert result.dv01.metadata["fallback_reason"]["code"] == "autodiff_public_curve_unavailable"

    def test_analyze_uses_interpolation_aware_krd_for_off_grid_buckets(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=[{"key_rate_durations": {"tenors": (7.0,), "bump_bps": 25.0}}],
        )

        assert result.key_rate_durations[7.0] > 0.0

    def test_analyze_rebuild_krd_uses_bootstrap_quote_buckets(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        s = Session(market_snapshot=_bootstrapped_snapshot(), settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=[{"key_rate_durations": {"methodology": "curve_rebuild", "bump_bps": 25.0}}],
        )

        assert set(result.key_rate_durations) == {"DEP1Y", "SWAP2Y", "SWAP5Y", "SWAP10Y"}
        metadata = result.key_rate_durations.metadata
        assert metadata["resolved_methodology"] == "curve_rebuild"
        assert metadata["bucket_convention"] == "bootstrap_quote"
        assert metadata["selected_curve_name"] == "usd_ois_boot"

    def test_analyze_rebuild_krd_falls_back_without_bootstrap_provenance(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=[
                {
                    "key_rate_durations": {
                        "methodology": "curve_rebuild",
                        "tenors": (2.0, 5.0, 10.0),
                    }
                }
            ],
        )

        assert set(result.key_rate_durations) == {2.0, 5.0, 10.0}
        metadata = result.key_rate_durations.metadata
        assert metadata["requested_methodology"] == "curve_rebuild"
        assert metadata["resolved_methodology"] == "zero_curve"
        assert metadata["fallback_reason"]["code"] == "bootstrap_discount_curve_unavailable"

    def test_analyze_custom_krd_buckets_still_sum_like_duration(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=[
                "price",
                "duration",
                {"key_rate_durations": {"tenors": (1.0, 3.0, 7.0, 10.0), "bump_bps": 25.0}},
            ],
        )

        assert sum(result.key_rate_durations.values()) == pytest.approx(result.duration, rel=0.10)

    def test_analyze_supports_named_twist_and_butterfly_scenario_packs(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        curve = YieldCurve([1.0, 2.0, 5.0, 10.0, 30.0], [0.04, 0.041, 0.043, 0.046, 0.048])
        s = Session(curve=curve, settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=[
                {
                    "scenario_pnl": {
                        "scenario_packs": ("twist", "butterfly"),
                        "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                        "pack_amplitude_bps": 25.0,
                    }
                }
            ],
        )

        assert "twist_steepener_25bp" in result.scenario_pnl
        assert "twist_flattener_25bp" in result.scenario_pnl
        assert "butterfly_belly_up_25bp" in result.scenario_pnl
        assert "butterfly_belly_down_25bp" in result.scenario_pnl

    def test_analyze_rebuild_scenario_pnl_uses_bootstrap_quote_methodology(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        s = Session(market_snapshot=_bootstrapped_snapshot(), settlement=SETTLE)
        result = s.analyze(
            DeterministicCashflowPayoff(_bond()),
            measures=[
                {
                    "scenario_pnl": {
                        "methodology": "curve_rebuild",
                        "scenario_packs": ("twist", "butterfly"),
                        "bucket_tenors": (1.0, 2.0, 5.0, 10.0),
                    }
                }
            ],
        )

        assert "twist_steepener_25bp" in result.scenario_pnl
        assert "butterfly_belly_up_25bp" in result.scenario_pnl
        assert not any(isinstance(key, int) for key in result.scenario_pnl)
        metadata = result.scenario_pnl.metadata
        assert metadata["resolved_methodology"] == "curve_rebuild"
        assert metadata["bucket_convention"] == "bootstrap_quote"
        assert metadata["scenario_templates"][0]["methodology"] == "curve_rebuild"
        assert metadata["scenario_templates"][0]["bucket_convention"] == "bootstrap_quote"
        assert metadata["scenario_templates"][0]["quote_bucket_bumps"]["DEP1Y"] == pytest.approx(
            -25.0
        )

    def test_analyze_bucketed_vega_returns_expiry_strike_surface(self):
        s = Session(market_snapshot=_snapshot(), settlement=SETTLE).with_vol_surface_name("usd_smile")
        result = s.analyze(
            _VolPointPayoff(1.5, 100.0),
            measures=[
                {
                    "vega": {
                        "expiries": (1.0, 1.5, 2.0),
                        "strikes": (90.0, 100.0, 110.0),
                        "bump_pct": 1.0,
                    }
                }
            ],
        )

        assert set(result.vega) == {1.0, 1.5, 2.0}
        assert set(result.vega[1.5]) == {90.0, 100.0, 110.0}
        assert result.vega[1.5][100.0] == pytest.approx(0.01, abs=1e-12)
        assert result.vega[1.0][90.0] == pytest.approx(0.0, abs=1e-12)
        metadata = result.vega.metadata
        assert metadata["bucket_convention"] == "expiry_strike"
        assert metadata["resolved_derivative_method"] == "surface_bucket_bump"
        assert metadata["bucket_expiries"] == [1.0, 1.5, 2.0]
        assert metadata["bucket_strikes"] == [90.0, 100.0, 110.0]
        warnings = {warning["code"] for warning in metadata["warnings"]}
        assert "interpolated_surface_bucket" in warnings

    def test_analyze_scalar_flat_vega_records_autodiff_provenance(self):
        s = Session(curve=_curve(), settlement=SETTLE, vol_surface=FlatVol(0.20))
        result = s.analyze(
            _TraceSafeVolPointPayoff(1.0, 90.0),
            measures=[{"vega": {"bump_pct": 1.0}}],
        )

        assert result.vega == pytest.approx(0.01, abs=1e-12)
        assert result.vega.metadata["resolved_derivative_method"] == "autodiff_flat_vol"
        assert result.vega.metadata["resolved_surface_type"] == "flat"

    def test_analyze_scalar_vega_records_representative_surface_fallback_provenance(self):
        s = Session(
            curve=_curve(),
            settlement=SETTLE,
            vol_surface=_RepresentativeVolSurface(0.20),
        )
        result = s.analyze(
            _TraceSafeVolPointPayoff(1.0, 90.0),
            measures=[{"vega": {"bump_pct": 1.0}}],
        )

        assert result.vega.metadata["resolved_derivative_method"] == "representative_flat_vol_bump"
        assert result.vega.metadata["fallback_reason"]["code"] == "representative_surface_reduction"
        warnings = {warning["code"] for warning in result.vega.metadata["warnings"]}
        assert "representative_surface_reduction" in warnings

    def test_analyze_bucketed_vega_records_flat_surface_expansion_warning(self):
        s = Session(curve=_curve(), settlement=SETTLE, vol_surface=FlatVol(0.20))
        result = s.analyze(
            _VolPointPayoff(1.0, 90.0),
            measures=[
                {
                    "vega": {
                        "expiries": (1.0, 2.0),
                        "strikes": (90.0, 110.0),
                        "bump_pct": 1.0,
                    }
                }
            ],
        )

        assert result.vega[1.0][90.0] == pytest.approx(0.01, abs=1e-12)
        warnings = {warning["code"] for warning in result.vega.metadata["warnings"]}
        assert "flat_surface_expanded" in warnings

    def test_analyze_delta_and_gamma_use_selected_spot_binding(self):
        s = Session(market_snapshot=_spot_snapshot(), settlement=SETTLE)
        result = s.analyze(
            _SpotQuadraticPayoff(),
            measures=[
                {"delta": {"bump_pct": 1.0}},
                {"gamma": {"bump_pct": 1.0}},
            ],
        )

        assert result.delta == pytest.approx(200.5, abs=1e-10)
        assert result.gamma == pytest.approx(2.0, abs=1e-10)
        assert result.delta.metadata["resolved_derivative_method"] == "spot_central_bump"
        assert result.gamma.metadata["resolved_derivative_method"] == "spot_central_bump"
        assert result.delta.metadata["resolved_spot_binding"] == "spot"

    def test_analyze_theta_rolls_one_day_forward(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        result = s.analyze(
            _LinearTimeDecayPayoff(date(2024, 11, 25)),
            measures=[{"theta": {"day_step": 1}}],
        )

        assert result.theta == pytest.approx(-1.0, abs=1e-12)
        assert result.theta.metadata["resolved_derivative_method"] == "calendar_roll_down_bump"
        assert result.theta.metadata["day_step"] == 1

    def test_analyze_delta_fails_when_no_spot_binding_is_available(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        s = Session(curve=_curve(), settlement=SETTLE)
        with pytest.raises(ValueError, match="spot"):
            s.analyze(
                DeterministicCashflowPayoff(_bond()),
                measures=["delta"],
            )

    def test_analyze_callable_oas_duration_is_positive(self):
        from trellis.instruments.callable_bond import CallableBondPayoff

        payoff = CallableBondPayoff(_callable_bond_spec())
        s = Session(curve=YieldCurve.flat(0.05), settlement=SETTLE, vol_surface=FlatVol(0.20))
        result = s.analyze(
            payoff,
            measures=["price", {"oas_duration": {"bump_bps": 25.0}}],
        )

        assert result.price > 0.0
        assert result.oas_duration > 0.0

    def test_analyze_callable_scenario_explain_tracks_optionality_under_rate_shifts(self):
        from trellis.instruments.callable_bond import CallableBondPayoff

        payoff = CallableBondPayoff(_callable_bond_spec())
        s = Session(curve=YieldCurve.flat(0.05), settlement=SETTLE, vol_surface=FlatVol(0.20))
        result = s.analyze(
            payoff,
            measures=[{"callable_scenario_explain": {"shifts_bps": (-100, 100)}}],
        )

        assert set(result.callable_scenario_explain) == {-100.0, 100.0}
        assert (
            result.callable_scenario_explain[-100.0]["call_option_value"]
            > result.callable_scenario_explain[100.0]["call_option_value"]
        )
        metadata = result.callable_scenario_explain.metadata
        assert metadata["controller_role"] == "issuer"
        assert metadata["exercise_dates"] == ["2028-01-15", "2030-01-15", "2032-01-15"]

    def test_analyze_callable_specific_measures_fail_for_non_callable_payoff(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        s = Session(curve=_curve(), settlement=SETTLE, vol_surface=FlatVol(0.20))
        with pytest.raises(ValueError, match="callable"):
            s.analyze(
                DeterministicCashflowPayoff(_bond()),
                measures=["oas_duration"],
            )

    def test_price_projects_executor_result(self):
        from trellis.platform.results import ExecutionResult

        projected = PricingResult(
            clean_price=101.25,
            dirty_price=102.00,
            accrued_interest=0.75,
            ytm=0.041,
            greeks={"dv01": 0.08},
            curve_sensitivities={10.0: -0.8},
        )
        s = Session(curve=_curve(), settlement=SETTLE)

        def fake_execute(compiled_request, execution_context, *, handlers=None):
            assert compiled_request.execution_plan.action == "price_existing_instrument"
            assert execution_context.session_id == s.session_id
            return ExecutionResult(
                run_id="run_session_price",
                request_id=compiled_request.request.request_id,
                status="succeeded",
                action=compiled_request.execution_plan.action,
                output_mode=execution_context.default_output_mode,
                result_payload={"result": projected},
                provenance={"run_mode": execution_context.run_mode.value},
                policy_outcome={"allowed": True},
            )

        with patch("trellis.platform.executor.execute_compiled_request", side_effect=fake_execute):
            with patch("trellis.session.price_instrument", side_effect=AssertionError("direct pricer should not run")):
                result = s.price(_bond(), greeks=["dv01"])

        assert result is projected

    def test_greeks_projects_executor_result(self):
        from trellis.platform.results import ExecutionResult

        projected = {"dv01": 0.12, "duration": 4.8}
        s = Session(curve=_curve(), settlement=SETTLE)

        def fake_execute(compiled_request, execution_context, *, handlers=None):
            assert compiled_request.execution_plan.action == "compute_greeks"
            return ExecutionResult(
                run_id="run_session_greeks",
                request_id=compiled_request.request.request_id,
                status="succeeded",
                action=compiled_request.execution_plan.action,
                output_mode=execution_context.default_output_mode,
                result_payload={"result": projected},
                provenance={"run_mode": execution_context.run_mode.value},
                policy_outcome={"allowed": True},
            )

        with patch("trellis.platform.executor.execute_compiled_request", side_effect=fake_execute):
            with patch("trellis.session.price_instrument", side_effect=AssertionError("direct greeks path should not run")):
                result = s.greeks(_bond(), measures=["dv01", "duration"])

        assert result == projected

    def test_analyze_projects_executor_result(self):
        from trellis.analytics.result import AnalyticsResult
        from trellis.core.payoff import DeterministicCashflowPayoff
        from trellis.platform.results import ExecutionResult

        projected = AnalyticsResult({"price": 98.2, "duration": 5.4})
        s = Session(curve=_curve(), settlement=SETTLE)
        payoff = DeterministicCashflowPayoff(_bond())

        def fake_execute(compiled_request, execution_context, *, handlers=None):
            assert compiled_request.execution_plan.action == "analyze_existing_instrument"
            return ExecutionResult(
                run_id="run_session_analyze",
                request_id=compiled_request.request.request_id,
                status="succeeded",
                action=compiled_request.execution_plan.action,
                output_mode=execution_context.default_output_mode,
                result_payload={"result": projected},
                provenance={"run_mode": execution_context.run_mode.value},
                policy_outcome={"allowed": True},
            )

        with patch("trellis.platform.executor.execute_compiled_request", side_effect=fake_execute):
            result = s.analyze(payoff, measures=["price", "duration"])

        assert result is projected

    def test_analyze_forwards_shared_measure_kwargs(self):
        from trellis.core.payoff import DeterministicCashflowPayoff

        class CaptureMarketPriceMeasure:

            name = "capture_market_price"
            requires = set()

            def compute(self, payoff, market_state, **context):
                return context["market_price"]

        s = Session(curve=_curve(), settlement=SETTLE)
        payoff = DeterministicCashflowPayoff(_bond())
        actual = s.analyze(
            payoff,
            measures=[CaptureMarketPriceMeasure()],
            market_price=92.0,
        )

        assert actual.capture_market_price == pytest.approx(92.0)

    def test_public_governed_entrypoints_delegate_to_shared_runner(self):
        from trellis.analytics.result import AnalyticsResult
        from trellis.core.payoff import DeterministicCashflowPayoff

        projected_price = PricingResult(
            clean_price=100.0,
            dirty_price=100.0,
            accrued_interest=0.0,
            greeks={"dv01": 0.1},
            curve_sensitivities={},
        )
        projected_analytics = AnalyticsResult({"price": 100.0, "duration": 4.5})
        s = Session(curve=_curve(), settlement=SETTLE)
        payoff = DeterministicCashflowPayoff(_bond())
        calls = []

        def fake_run(session, **kwargs):
            calls.append(kwargs)
            if kwargs["request_type"] == "price":
                return projected_price
            if kwargs["request_type"] == "greeks":
                return {"dv01": 0.12}
            return projected_analytics

        with patch.object(Session, "_run_governed_request", autospec=True, side_effect=fake_run):
            price_result = s.price(_bond(), greeks=["dv01"])
            greeks_result = s.greeks(_bond(), measures=["dv01"])
            analytics_result = s.analyze(payoff, measures=["price", "duration"])

        assert price_result is projected_price
        assert greeks_result == {"dv01": 0.12}
        assert analytics_result is projected_analytics
        assert [call["request_type"] for call in calls] == ["price", "greeks", "analytics"]
        assert [call["success_outcome"] for call in calls] == [
            "priced",
            "greeks_computed",
            "analytics_computed",
        ]
        assert [call["failure_outcome"] for call in calls] == [
            "price_failed",
            "greeks_failed",
            "analytics_failed",
        ]

    def test_price_failure_records_platform_trace(self):
        s = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
        recorded = []

        def fake_record(compiled_request, *, success, outcome, details=None, root=None):
            recorded.append(
                {
                    "request_id": compiled_request.request.request_id,
                    "success": success,
                    "outcome": outcome,
                    "details": details,
                }
            )

        with patch("trellis.engine.pricer.price_instrument", side_effect=ValueError("boom")):
            with patch("trellis.agent.platform_traces.record_platform_trace", side_effect=fake_record):
                with pytest.raises(ValueError, match="boom"):
                    s.price(_bond())

        assert recorded
        assert recorded[-1]["success"] is False
        assert recorded[-1]["outcome"] == "price_failed"
        assert recorded[-1]["details"]["error"] == "boom"


class TestSessionScenarios:

    def test_with_curve_shift_returns_new_session(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        s2 = s.with_curve_shift(+100)
        assert s2 is not s
        # Original unchanged
        assert np.allclose(s.curve.rates, 0.045)
        # Shifted curve
        assert np.allclose(s2.curve.rates, 0.045 + 0.01)

    def test_shift_lowers_price(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        p_base = s.price(_bond()).clean_price
        s2 = s.with_curve_shift(+100)
        p_shifted = s2.price(_bond()).clean_price
        assert p_shifted < p_base

    def test_with_tenor_bumps(self):
        tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.04, 0.042, 0.045, 0.047, 0.05]
        curve = YieldCurve(tenors, rates)
        s = Session(curve=curve, settlement=SETTLE)
        s2 = s.with_tenor_bumps({10.0: +50})
        # 10Y rate bumped by 50bps
        idx = 3  # 10.0 is at index 3
        assert float(s2.curve.rates[idx]) == pytest.approx(0.047 + 0.005, abs=1e-9)
        # Other rates unchanged
        assert float(s2.curve.rates[0]) == pytest.approx(0.04, abs=1e-9)

    def test_with_tenor_bumps_supports_off_grid_bucket_shocks(self):
        tenors = [1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.04, 0.042, 0.045, 0.047, 0.05]
        curve = YieldCurve(tenors, rates)
        s = Session(curve=curve, settlement=SETTLE)

        base_price = s.price(_bond(), greeks=None).clean_price
        s2 = s.with_tenor_bumps({7.0: +25})
        shocked_price = s2.price(_bond(), greeks=None).clean_price

        assert tuple(float(tenor) for tenor in s2.curve.tenors) == pytest.approx((1.0, 2.0, 5.0, 7.0, 10.0, 30.0))
        assert s2.curve.zero_rate(7.0) == pytest.approx(curve.zero_rate(7.0) + 0.0025)
        assert shocked_price < base_price

    def test_with_curve_replaces(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        new_curve = YieldCurve.flat(0.06)
        s2 = s.with_curve(new_curve)
        assert np.allclose(s2.curve.rates, 0.06)


class TestSessionImmutability:

    def test_setattr_raises(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        with pytest.raises(AttributeError, match="immutable"):
            s.foo = "bar"


class TestSessionAgent:

    def test_agent_false_raises(self):
        """With agent=False (default), unsupported types propagate errors."""
        s = Session(curve=_curve(), settlement=SETTLE)
        with pytest.raises((NotImplementedError, TypeError, ValueError, AttributeError)):
            # Pass something that isn't an instrument
            s.price("not_a_bond")


class TestSessionMarketData:

    @patch("trellis.data.resolver.resolve_market_snapshot")
    def test_auto_resolution(self, mock_resolve):
        mock_resolve.return_value = _snapshot()
        s = Session(as_of="2024-11-15", data_source="treasury_gov")
        assert s.curve is not None
        mock_resolve.assert_called_once_with(
            as_of="2024-11-15",
            source="treasury_gov",
            vol_surface=None,
            vol_surfaces=None,
            default_vol_surface=None,
            forecast_curves=None,
            credit_curve=None,
            fx_rates=None,
            metadata=None,
        )

    def test_mock_source_no_mocking_needed(self):
        """Full offline path — no patches, no network."""
        s = Session(as_of="2024-11-15", data_source="mock")
        result = s.price(_bond())
        assert result.clean_price > 0

    def test_mock_source_loads_full_snapshot_context(self):
        s = Session(as_of="2024-11-15", data_source="mock")
        assert s.market_snapshot is not None
        assert s.market_snapshot.provenance["source_kind"] == "synthetic_snapshot"
        assert s.vol_surface is not None
        assert s.credit_curve is not None
        assert "USD-SOFR-3M" in s.forecast_curves
        assert "EURUSD" in s.fx_rates

    def test_explicit_curve_session_records_market_provenance(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        assert s.market_snapshot is not None
        assert s.market_snapshot.provenance["source_kind"] == "explicit_input"
        assert s.market_snapshot.provenance["source_ref"] == "Session(curve=...)"

    def test_mock_source_can_price_cap_without_manual_vol_surface(self):
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency

        s = Session(as_of="2024-11-15", data_source="mock", settlement=SETTLE)
        spec = CapFloorSpec(
            notional=1_000_000,
            strike=0.05,
            start_date=date(2025, 2, 15),
            end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
            rate_index="USD-SOFR-3M",
        )
        pv = s.price_payoff(CapPayoff(spec))
        assert pv > 0


class TestSpreadToCurve:

    def test_spread_to_curve(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        bond = _bond()
        # Price at base curve
        base_price = s.price(bond, greeks=None).clean_price
        # Now shift curve +50bp and get that price
        s2 = s.with_curve_shift(50)
        target_price = s2.price(bond, greeks=None).clean_price
        # spread_to_curve should find ~50 bps
        spread = s.spread_to_curve(bond, target_price)
        assert spread == pytest.approx(50.0, abs=0.5)


class TestRiskReport:

    def test_risk_report_structure(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        book = Book({"10Y": _bond()})
        report = s.risk_report(book)
        assert "total_mv" in report
        assert "book_dv01" in report
        assert "book_duration" in report
        assert "book_krd" in report
        assert "positions" in report
        assert "portfolio_aad" in report
        assert "10Y" in report["positions"]

    def test_risk_report_uses_canonical_numeric_krd_keys(self):
        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        book = Book({"10Y": _bond()})

        report = s.risk_report(book)

        assert set(report["book_krd"]) == {1.0, 2.0, 5.0, 10.0}

    def test_risk_report_excludes_unsupported_positions_explicitly(self):
        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        book = Book({"10Y": _bond(), "unsupported": object()})

        report = s.risk_report(book)

        assert "unsupported" not in report["positions"]
        assert report["unsupported_positions"][0]["position_name"] == "unsupported"
        assert report["unsupported_positions"][0]["reason"] == "unsupported_instrument_type"
        assert report["portfolio_aad"]["metadata"]["support_status"] == "partial"
        assert report["portfolio_aad"]["metadata"]["unsupported_position_count"] == 1

    def test_risk_report_portfolio_aad_records_vjp_provenance(self):
        curve = YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047])
        s = Session(curve=curve, settlement=SETTLE)
        book = Book({"10Y": _bond()})

        report = s.risk_report(book)

        assert report["portfolio_aad"]["metadata"]["resolved_derivative_method"] == "portfolio_aad_vjp"
        assert report["portfolio_aad"]["metadata"]["backend_operator"] == "vjp"
        assert report["portfolio_aad"]["metadata"]["unsupported_position_count"] == 0
        assert set(report["portfolio_aad"]["values"]) == {1.0, 2.0, 5.0, 10.0}


class TestSessionPricePayoff:

    def test_price_payoff_basic(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        s = Session(curve=_curve(), settlement=SETTLE)
        adapter = DeterministicCashflowPayoff(_bond())
        pv = s.price_payoff(adapter)
        assert 80 < pv < 120

    def test_price_payoff_matches_price_instrument(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        s = Session(curve=_curve(), settlement=SETTLE)
        bond = _bond()
        result = s.price(bond, greeks=None)
        adapter = DeterministicCashflowPayoff(bond)
        pv = s.price_payoff(DeterministicCashflowPayoff(bond, day_count=bond.day_count))
        assert pv == pytest.approx(result.dirty_price, rel=1e-12)

    def test_price_payoff_with_scenario(self):
        from trellis.core.payoff import DeterministicCashflowPayoff
        s = Session(curve=_curve(), settlement=SETTLE)
        adapter = DeterministicCashflowPayoff(_bond())
        pv_base = s.price_payoff(adapter)
        s2 = s.with_curve_shift(+100)
        pv_shifted = s2.price_payoff(adapter)
        assert pv_shifted < pv_base

    def test_price_cap_payoff(self):
        from trellis.instruments.cap import CapFloorSpec, CapPayoff
        from trellis.core.types import Frequency
        from trellis.models.vol_surface import FlatVol

        s = Session(curve=_curve(), settlement=SETTLE, vol_surface=FlatVol(0.20))
        spec = CapFloorSpec(
            notional=1_000_000, strike=0.05,
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )
        pv = s.price_payoff(CapPayoff(spec))
        assert pv > 0

    def test_with_vol_surface(self):
        s = Session(curve=_curve(), settlement=SETTLE)
        s2 = s.with_vol_surface(FlatVol(0.30))
        assert s2 is not s
        assert s2.vol_surface is not None
        assert s.vol_surface is None


class TestSessionMarketSnapshot:

    def test_session_uses_market_snapshot_defaults(self):
        snapshot = _snapshot()
        s = Session(market_snapshot=snapshot, settlement=SETTLE)
        assert s.market_snapshot is snapshot
        assert s.curve is snapshot.discount_curve()
        assert "USD-SOFR-3M" in s.forecast_curves
        assert s.vol_surface is snapshot.vol_surface()

    def test_with_discount_curve_switches_named_curve(self):
        snapshot = _snapshot()
        s = Session(market_snapshot=snapshot, settlement=SETTLE)
        s2 = s.with_discount_curve("eur_ois")
        assert s2 is not s
        assert float(s2.curve.rates[0]) == pytest.approx(0.025, abs=1e-12)
        assert s2.market_snapshot is snapshot

    def test_with_curve_shift_preserves_snapshot_context(self):
        snapshot = _snapshot()
        s = Session(market_snapshot=snapshot, settlement=SETTLE)
        s2 = s.with_curve_shift(+100)
        assert s2.market_snapshot is not None
        assert "USD-SOFR-3M" in s2.forecast_curves
        assert float(s2.curve.rates[0]) == pytest.approx(0.045 + 0.01, abs=1e-12)

    def test_with_vol_surface_name_switches_named_surface(self):
        snapshot = _snapshot()
        s = Session(market_snapshot=snapshot, settlement=SETTLE)
        s2 = s.with_vol_surface_name("usd_smile")
        assert s2 is not s
        assert s2.vol_surface is snapshot.vol_surface("usd_smile")
        assert s2.vol_surface.black_vol(1.5, 100.0) == pytest.approx(0.245, abs=1e-12)
