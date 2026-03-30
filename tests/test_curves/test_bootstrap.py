"""Tests for differentiable curve bootstrapping."""

from datetime import date

import numpy as np
import pytest

from trellis.curves.bootstrap import (
    BootstrapInstrument,
    bootstrap,
    bootstrap_named_yield_curves,
    bootstrap_yield_curve,
)
from trellis.curves.yield_curve import YieldCurve


class TestBootstrapDeposits:

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
