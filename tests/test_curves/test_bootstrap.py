"""Tests for differentiable curve bootstrapping."""

from datetime import date

import numpy as np
import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.types import Frequency
from trellis.curves.bootstrap import (
    BootstrapCalibrationDiagnostics,
    BootstrapCalibrationResult,
    BootstrapConventionBundle,
    BootstrapCurveInputBundle,
    BootstrapInstrument,
    bootstrap,
    bootstrap_curve_result,
    bootstrap_named_yield_curves,
    bootstrap_yield_curve,
    build_bootstrap_solve_request,
)
from trellis.curves.yield_curve import YieldCurve


def _par_swap_rate(curve: YieldCurve, tenor: float, *, fixed_step: float, float_step: float) -> float:
    """Return the par swap rate implied by ``curve`` for explicit accrual steps."""
    float_pv = 0.0
    n_float = int(round(tenor / float_step))
    for k in range(n_float):
        t0 = k * float_step
        t1 = (k + 1) * float_step
        df0 = float(curve.discount(max(t0, 0.001)))
        df1 = float(curve.discount(t1))
        fwd = (df0 / df1 - 1.0) / float_step
        float_pv += fwd * float_step * df1

    annuity = 0.0
    n_fixed = int(round(tenor / fixed_step))
    for k in range(n_fixed):
        t_pay = (k + 1) * fixed_step
        annuity += fixed_step * float(curve.discount(t_pay))
    return float_pv / annuity


class TestBootstrapDeposits:

    def test_curve_input_bundle_serializes_conventions_and_instruments(self):
        bundle = BootstrapCurveInputBundle(
            curve_name="usd_ois_boot",
            currency="USD",
            rate_index="USD-SOFR-3M",
            conventions=BootstrapConventionBundle(
                deposit_day_count=DayCountConvention.ACT_360,
                future_day_count=DayCountConvention.ACT_360,
                swap_fixed_frequency=Frequency.ANNUAL,
                swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                swap_float_frequency=Frequency.QUARTERLY,
                swap_float_day_count=DayCountConvention.ACT_360,
            ),
            instruments=(
                BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit", label="DEP3M"),
                BootstrapInstrument(tenor=2.0, quote=0.045, instrument_type="swap", label="SWAP2Y"),
            ),
        )

        payload = bundle.to_payload()

        assert payload["curve_name"] == "usd_ois_boot"
        assert payload["currency"] == "USD"
        assert payload["rate_index"] == "USD-SOFR-3M"
        assert payload["conventions"]["swap_fixed_frequency"] == "ANNUAL"
        assert payload["conventions"]["swap_fixed_day_count"] == "THIRTY_360"
        assert payload["instruments"][0]["label"] == "DEP3M"
        assert payload["instruments"][1]["instrument_type"] == "swap"

    def test_single_deposit(self):
        """Single deposit should recover the quoted rate as a zero rate."""
        instruments = [BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit")]
        tenors, rates = bootstrap(instruments)
        # For a deposit: quote = (1/df - 1) / t, df = exp(-r*t)
        # So: 0.05 = (exp(r) - 1) / 1 => r = ln(1.05) = 0.04879
        expected_r = np.log(1.05)
        assert float(rates[0]) == pytest.approx(expected_r, rel=1e-8)

    def test_two_deposits(self):
        """Two deposits at different tenors."""
        instruments = [
            BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit"),
            BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit"),
        ]
        tenors, rates = bootstrap(instruments)

        # Verify repricing: each deposit should reprice to its quote
        for inst, r in zip(instruments, rates):
            df = np.exp(-float(r) * inst.tenor)
            model_rate = (1.0 / df - 1.0) / inst.tenor
            assert model_rate == pytest.approx(inst.quote, rel=1e-8)

    def test_flat_deposits(self):
        """All deposits at the same rate → flat zero curve."""
        instruments = [
            BootstrapInstrument(tenor=0.25, quote=0.05, instrument_type="deposit"),
            BootstrapInstrument(tenor=0.5, quote=0.05, instrument_type="deposit"),
            BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit"),
        ]
        tenors, rates = bootstrap(instruments)
        # All CC rates should be similar (not exactly equal due to simple→CC conversion)
        for r in rates:
            assert 0.04 < float(r) < 0.06


class TestBootstrapSwaps:

    def test_bootstrap_curve_result_exposes_solver_artifacts_and_jacobian(self):
        bundle = BootstrapCurveInputBundle(
            curve_name="usd_ois_boot",
            currency="USD",
            rate_index="USD-SOFR-3M",
            conventions=BootstrapConventionBundle(
                swap_fixed_frequency=Frequency.ANNUAL,
                swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                swap_float_frequency=Frequency.QUARTERLY,
                swap_float_day_count=DayCountConvention.ACT_360,
            ),
            instruments=(
                BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit", label="DEP3M"),
                BootstrapInstrument(tenor=2.0, quote=0.045, instrument_type="swap", label="SWAP2Y"),
                BootstrapInstrument(tenor=5.0, quote=0.048, instrument_type="swap", label="SWAP5Y"),
            ),
        )

        solve_request = build_bootstrap_solve_request(bundle, max_iter=75, tol=1e-12)
        result = bootstrap_curve_result(bundle, max_iter=75, tol=1e-12)

        assert solve_request.problem_kind == "least_squares"
        assert solve_request.objective.labels == ("DEP3M", "SWAP2Y", "SWAP5Y")
        assert solve_request.objective.target_values == pytest.approx((0.04, 0.045, 0.048))
        assert isinstance(result, BootstrapCalibrationResult)
        assert isinstance(result.diagnostics, BootstrapCalibrationDiagnostics)
        assert result.solve_result.metadata["backend_id"] == "scipy"
        assert result.solve_result.metadata["derivative_method"] == "autodiff_vector_jacobian"
        assert result.solver_provenance.backend["backend_id"] == "scipy"
        assert result.solver_provenance.backend["derivative_method"] == "autodiff_vector_jacobian"
        assert result.solver_replay_artifact.request["request_id"] == "rates_bootstrap_least_squares"
        assert result.diagnostics.max_abs_residual < 1e-8
        assert result.diagnostics.jacobian_rank == 3
        assert len(result.diagnostics.jacobian_matrix) == 3
        assert len(result.diagnostics.jacobian_matrix[0]) == 3
        assert result.to_payload()["solver_provenance"]["backend"]["backend_id"] == "scipy"
        assert result.to_payload()["solver_provenance"]["backend"]["derivative_method"] == (
            "autodiff_vector_jacobian"
        )

    def test_bootstrap_respects_explicit_bundle_conventions(self):
        """Bundles should drive repricing instead of the old hard-coded swap glue."""
        bundle = BootstrapCurveInputBundle(
            curve_name="usd_ois_boot",
            currency="USD",
            rate_index="USD-SOFR-3M",
            conventions=BootstrapConventionBundle(
                swap_fixed_frequency=Frequency.ANNUAL,
                swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                swap_float_frequency=Frequency.QUARTERLY,
                swap_float_day_count=DayCountConvention.ACT_360,
            ),
            instruments=(
                BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit", label="DEP3M"),
                BootstrapInstrument(tenor=2.0, quote=0.045, instrument_type="swap", label="SWAP2Y"),
                BootstrapInstrument(tenor=5.0, quote=0.048, instrument_type="swap", label="SWAP5Y"),
            ),
        )

        curve = bootstrap_yield_curve(bundle)

        for inst in bundle.instruments:
            if inst.instrument_type != "swap":
                continue
            par_rate = _par_swap_rate(curve, inst.tenor, fixed_step=1.0, float_step=0.25)
            assert par_rate == pytest.approx(inst.quote, abs=1e-6)

    def test_deposit_plus_swap(self):
        """Combine a deposit and a swap."""
        instruments = [
            BootstrapInstrument(tenor=0.5, quote=0.04, instrument_type="deposit"),
            BootstrapInstrument(tenor=2.0, quote=0.045, instrument_type="swap"),
            BootstrapInstrument(tenor=5.0, quote=0.05, instrument_type="swap"),
        ]
        tenors, rates = bootstrap(instruments)
        assert len(rates) == 3
        # Rates should be positive and reasonable
        for r in rates:
            assert 0.03 < float(r) < 0.07

    def test_swap_repricing(self):
        """Bootstrapped curve should reprice calibration swaps to par."""
        instruments = [
            BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit"),
            BootstrapInstrument(tenor=1.0, quote=0.042, instrument_type="deposit"),
            BootstrapInstrument(tenor=2.0, quote=0.045, instrument_type="swap"),
            BootstrapInstrument(tenor=5.0, quote=0.048, instrument_type="swap"),
        ]
        tenors, rates = bootstrap(instruments)
        curve = YieldCurve(tenors, rates)

        # Verify swap repricing: compute par swap rate from the curve
        # For the 2Y swap with quarterly float / semi-annual fixed:
        for inst in instruments:
            if inst.instrument_type == "swap":
                t = inst.tenor
                # Float PV
                n_float = int(t * 4)
                float_pv = 0.0
                for k in range(n_float):
                    t0 = max(k * 0.25, 0.001)
                    t1 = (k + 1) * 0.25
                    df0 = float(curve.discount(t0))
                    df1 = float(curve.discount(t1))
                    fwd = (df0 / df1 - 1.0) / 0.25
                    float_pv += fwd * 0.25 * df1

                # Fixed annuity
                n_fixed = int(t * 2)
                annuity = sum(
                    0.5 * float(curve.discount((k + 1) * 0.5))
                    for k in range(n_fixed)
                )
                par_rate = float_pv / annuity
                assert par_rate == pytest.approx(inst.quote, abs=1e-6), \
                    f"Swap at {t}Y: par_rate={par_rate:.6f}, quote={inst.quote:.6f}"


class TestBootstrapYieldCurve:

    def test_returns_yield_curve(self):
        instruments = [
            BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit"),
        ]
        curve = bootstrap_yield_curve(instruments)
        assert isinstance(curve, YieldCurve)
        assert len(curve.tenors) == 1

    def test_curve_discounts(self):
        instruments = [
            BootstrapInstrument(tenor=0.5, quote=0.04, instrument_type="deposit"),
            BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit"),
        ]
        curve = bootstrap_yield_curve(instruments)
        # Discount factors should be decreasing
        assert float(curve.discount(0.5)) > float(curve.discount(1.0))
        # And positive
        assert float(curve.discount(1.0)) > 0


class TestBootstrapNamedYieldCurves:

    def test_bootstraps_multiple_named_curve_sets(self):
        curve_sets = {
            "usd_ois_boot": [
                BootstrapInstrument(tenor=1.0, quote=0.05, instrument_type="deposit"),
            ],
            "eur_ois_boot": [
                BootstrapInstrument(tenor=1.0, quote=0.03, instrument_type="deposit"),
            ],
        }

        curves = bootstrap_named_yield_curves(curve_sets)

        assert set(curves) == {"usd_ois_boot", "eur_ois_boot"}
        assert float(curves["usd_ois_boot"].discount(1.0)) == pytest.approx(1.0 / 1.05)
        assert float(curves["eur_ois_boot"].discount(1.0)) == pytest.approx(1.0 / 1.03)


class TestBootstrapFutures:

    def test_deposit_plus_future(self):
        """Combine deposit with futures."""
        instruments = [
            BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit"),
            BootstrapInstrument(tenor=0.5, quote=95.5, instrument_type="future"),  # 4.5% rate
        ]
        tenors, rates = bootstrap(instruments)
        assert len(rates) == 2
        for r in rates:
            assert 0.03 < float(r) < 0.06


class TestConvergence:

    def test_converges(self):
        """Bootstrap should converge within reasonable iterations."""
        instruments = [
            BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit"),
            BootstrapInstrument(tenor=0.5, quote=0.042, instrument_type="deposit"),
            BootstrapInstrument(tenor=1.0, quote=0.045, instrument_type="deposit"),
            BootstrapInstrument(tenor=2.0, quote=0.047, instrument_type="swap"),
            BootstrapInstrument(tenor=5.0, quote=0.050, instrument_type="swap"),
            BootstrapInstrument(tenor=10.0, quote=0.048, instrument_type="swap"),
        ]
        tenors, rates = bootstrap(instruments, max_iter=50, tol=1e-10)
        # Should have converged — verify deposit repricing
        for inst in instruments:
            if inst.instrument_type == "deposit":
                r = float(rates[list(tenors).index(inst.tenor)])
                df = np.exp(-r * inst.tenor)
                model_rate = (1.0 / df - 1.0) / inst.tenor
                assert model_rate == pytest.approx(inst.quote, rel=1e-6)
