"""Tests for calibration: implied vol, SABR fit, rates vol, Dupire local vol."""

from dataclasses import replace
from datetime import date

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.instruments._agent.swaption import SwaptionPayoff, SwaptionSpec
from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
from trellis.models.calibration.rates import (
    RatesCalibrationResult,
    calibrate_cap_floor_black_vol,
    calibrate_swaption_black_vol,
    swaption_terms,
)
from trellis.models.calibration.implied_vol import implied_vol, implied_vol_jaeckel, _bs_price
from trellis.models.calibration.sabr_fit import calibrate_sabr
from trellis.models.calibration.local_vol import dupire_local_vol
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


# ---------------------------------------------------------------------------
# implied_vol round-trip
# ---------------------------------------------------------------------------


class TestImpliedVol:
    def test_round_trip_call(self):
        """Compute BS call price, then recover vol."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        price = _bs_price(S, K, T, r, sigma, "call")
        recovered = implied_vol(price, S, K, T, r, option_type="call")
        assert recovered == pytest.approx(sigma, abs=1e-6)

    def test_round_trip_put(self):
        S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.05, 0.30
        price = _bs_price(S, K, T, r, sigma, "put")
        recovered = implied_vol(price, S, K, T, r, option_type="put")
        assert recovered == pytest.approx(sigma, abs=1e-6)

    def test_round_trip_otm_call(self):
        S, K, T, r, sigma = 100.0, 120.0, 0.5, 0.03, 0.25
        price = _bs_price(S, K, T, r, sigma, "call")
        recovered = implied_vol(price, S, K, T, r, option_type="call")
        assert recovered == pytest.approx(sigma, abs=1e-6)


# ---------------------------------------------------------------------------
# implied_vol_jaeckel round-trip
# ---------------------------------------------------------------------------


class TestImpliedVolJaeckel:
    def test_round_trip_call(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        price = _bs_price(S, K, T, r, sigma, "call")
        recovered = implied_vol_jaeckel(price, S, K, T, r, option_type="call")
        assert recovered == pytest.approx(sigma, abs=1e-4)

    def test_matches_brent_method(self):
        """Jaeckel and Brent should give the same result."""
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
        price = _bs_price(S, K, T, r, sigma, "call")
        vol_brent = implied_vol(price, S, K, T, r, option_type="call")
        vol_jaeckel = implied_vol_jaeckel(price, S, K, T, r, option_type="call")
        assert vol_jaeckel == pytest.approx(vol_brent, abs=1e-4)


# ---------------------------------------------------------------------------
# SABR calibration
# ---------------------------------------------------------------------------


class TestSABRCalibration:
    def test_calibrated_vols_match_market(self):
        """Generate market vols from known SABR params, then calibrate back."""
        from trellis.models.processes.sabr import SABRProcess

        F, T = 100.0, 1.0
        alpha_true, beta, rho_true, nu_true = 0.20, 0.5, -0.3, 0.4
        sabr_true = SABRProcess(alpha_true, beta, rho_true, nu_true)

        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
        market_vols = [sabr_true.implied_vol(F, K, T) for K in strikes]

        sabr_fit = calibrate_sabr(F, T, strikes, market_vols, beta=beta)

        for K, mv in zip(strikes, market_vols):
            fitted_vol = sabr_fit.implied_vol(F, K, T)
            assert fitted_vol == pytest.approx(mv, abs=0.005)
        assert sabr_fit.calibration_provenance["source_kind"] == "calibrated_surface"
        assert sabr_fit.calibration_provenance["calibration_target"]["strike_count"] == len(strikes)
        assert sabr_fit.calibration_provenance["calibration_target"]["beta"] == pytest.approx(beta)
        assert sabr_fit.calibration_summary["optimizer_success"] is True

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match="same length"):
            calibrate_sabr(100.0, 1.0, [90.0, 100.0, 110.0], [0.2, 0.21], beta=0.5)


# ---------------------------------------------------------------------------
# Dupire local vol
# ---------------------------------------------------------------------------


class TestDupireLocalVol:
    def test_flat_vol_surface_gives_constant_local_vol(self):
        """If implied vol is flat (constant sigma), local vol = sigma everywhere."""
        sigma_flat = 0.20
        S0, r = 100.0, 0.05

        strikes = raw_np.linspace(60, 150, 30)
        expiries = raw_np.linspace(0.1, 3.0, 15)
        # Flat surface
        implied_vols = raw_np.full((len(expiries), len(strikes)), sigma_flat)

        local_vol_fn = dupire_local_vol(strikes, expiries, implied_vols, S0, r)

        # Check local vol at several (S, t) points near the center
        for S in [90.0, 100.0, 110.0]:
            for t in [0.5, 1.0, 2.0]:
                lv = local_vol_fn(S, t)
                assert lv == pytest.approx(sigma_flat, abs=0.02)
        assert local_vol_fn.calibration_provenance["source_kind"] == "calibrated_surface"
        assert local_vol_fn.calibration_target["surface_shape"] == (len(expiries), len(strikes))
        assert local_vol_fn.calibration_summary["spot"] == pytest.approx(S0)

    def test_rejects_mismatched_surface_shape(self):
        with pytest.raises(ValueError, match="shape"):
            dupire_local_vol(
                raw_np.array([60.0, 70.0, 80.0, 90.0]),
                raw_np.array([0.25, 0.5, 1.0, 2.0]),
                raw_np.ones((3, 4)),
                100.0,
                0.05,
            )


# ---------------------------------------------------------------------------
# Rates Black-vol calibration
# ---------------------------------------------------------------------------


def _multi_curve_state() -> tuple[MarketSnapshot, object]:
    """Return a multi-curve snapshot and its compiled MarketState."""
    snapshot = MarketSnapshot(
        as_of=SETTLE,
        source="test",
        discount_curves={
            "usd_ois": YieldCurve.flat(0.050),
            "usd_ois_alt": YieldCurve.flat(0.045),
        },
        forecast_curves={
            "USD-SOFR-3M": YieldCurve.flat(0.052),
            "USD-LIBOR-3M": YieldCurve.flat(0.054),
        },
        provenance={
            "source": "test",
            "source_kind": "explicit_input",
            "source_ref": "_multi_curve_state",
        },
    )
    market_state = snapshot.to_market_state(
        settlement=SETTLE,
        discount_curve="usd_ois",
        forecast_curve="USD-SOFR-3M",
    )
    return snapshot, market_state


class TestRatesCalibration:
    @pytest.mark.parametrize("kind,payoff_cls", [("cap", CapPayoff), ("floor", FloorPayoff)])
    def test_cap_floor_round_trip_preserves_curve_provenance(self, kind, payoff_cls):
        """Cap/floor implied-vol calibration should round-trip under multi-curve inputs."""
        _snapshot, market_state = _multi_curve_state()
        true_vol = 0.215
        spec = CapFloorSpec(
            notional=1_000_000.0,
            strike=0.05,
            start_date=date(2025, 2, 15),
            end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
        )
        target_state = replace(market_state, vol_surface=FlatVol(true_vol))
        target_price = payoff_cls(spec).evaluate(target_state)

        result = calibrate_cap_floor_black_vol(
            spec,
            market_state,
            target_price,
            kind=kind,
            vol_surface_name="rates_cap_surface",
            correlation_source="not_used",
        )

        assert isinstance(result, RatesCalibrationResult)
        assert result.calibrated_vol == pytest.approx(true_vol, abs=1e-6)
        assert result.repriced_price == pytest.approx(target_price, abs=1e-5)
        assert result.residual == pytest.approx(0.0, abs=1e-5)
        assert result.provenance["selected_curve_names"] == {
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        }
        assert result.provenance["rate_index"] == "USD-SOFR-3M"
        assert result.provenance["vol_surface_name"] == "rates_cap_surface"
        assert result.provenance["correlation_source"] == "not_used"
        assert result.provenance["market_provenance"]["source_kind"] == "explicit_input"
        assert result.summary["period_count"] > 0

    def test_swaption_round_trip_preserves_curve_provenance(self):
        """Swaption implied-vol calibration should round-trip under multi-curve inputs."""
        _snapshot, market_state = _multi_curve_state()
        true_vol = 0.180
        spec = SwaptionSpec(
            notional=5_000_000.0,
            strike=0.05,
            expiry_date=date(2026, 2, 15),
            swap_start=date(2026, 2, 15),
            swap_end=date(2031, 2, 15),
            swap_frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            is_payer=True,
        )
        target_state = replace(market_state, vol_surface=FlatVol(true_vol))
        target_price = SwaptionPayoff(spec).evaluate(target_state)
        T, annuity, forward_swap_rate, payment_count = swaption_terms(spec, market_state)
        assert T > 0.0
        assert annuity > 0.0
        assert forward_swap_rate > 0.0
        assert payment_count > 0

        result = calibrate_swaption_black_vol(
            spec,
            market_state,
            target_price,
            vol_surface_name="rates_swaption_surface",
            correlation_source="corr_pack_A",
        )

        assert isinstance(result, RatesCalibrationResult)
        assert result.calibrated_vol == pytest.approx(true_vol, abs=1e-6)
        assert result.repriced_price == pytest.approx(target_price, abs=1e-5)
        assert result.residual == pytest.approx(0.0, abs=1e-5)
        assert result.provenance["selected_curve_names"] == {
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        }
        assert result.provenance["rate_index"] == "USD-SOFR-3M"
        assert result.provenance["vol_surface_name"] == "rates_swaption_surface"
        assert result.provenance["correlation_source"] == "corr_pack_A"
        assert result.provenance["market_provenance"]["source_ref"] == "_multi_curve_state"
        assert result.summary["annuity"] > 0.0
        assert result.summary["forward_swap_rate"] > 0.0
