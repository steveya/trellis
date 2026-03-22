"""Tests for calibration: implied vol, SABR fit, Dupire local vol."""

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.models.calibration.implied_vol import implied_vol, implied_vol_jaeckel, _bs_price
from trellis.models.calibration.sabr_fit import calibrate_sabr
from trellis.models.calibration.local_vol import dupire_local_vol


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
