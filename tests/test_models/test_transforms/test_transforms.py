"""Tests for transform methods: FFT pricer, COS method, Heston characteristic function."""

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.models.transforms.fft_pricer import fft_price
from trellis.models.transforms.cos_method import cos_price
from trellis.models.processes.heston import Heston


def bs_call(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)


def gbm_char_fn(S0, r, sigma, T):
    """GBM characteristic function of log(S_T).

    phi(u) = exp(i*u*(log(S0) + (r - 0.5*sigma^2)*T) - 0.5*sigma^2*T*u^2)
    """
    def phi(u):
        return raw_np.exp(
            1j * u * (raw_np.log(S0) + (r - 0.5 * sigma**2) * T)
            - 0.5 * sigma**2 * T * u**2
        )
    return phi


def gbm_char_fn_log_ratio(r, sigma, T):
    """GBM characteristic function of log(S_T / S0) for COS method.

    phi(u) = exp(i*u*(r - 0.5*sigma^2)*T - 0.5*sigma^2*T*u^2)
    """
    def phi(u):
        return raw_np.exp(
            1j * u * (r - 0.5 * sigma**2) * T
            - 0.5 * sigma**2 * T * u**2
        )
    return phi


# ---------------------------------------------------------------------------
# FFT pricer with GBM
# ---------------------------------------------------------------------------


class TestFFTPricer:
    def test_gbm_call_matches_bs(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        char_fn = gbm_char_fn(S0, r, sigma, T)
        fft_call = fft_price(char_fn, S0, K, T, r)
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert fft_call == pytest.approx(bs_ref, rel=0.01)

    def test_gbm_itm_call(self):
        S0, K, r, sigma, T = 100.0, 90.0, 0.05, 0.20, 1.0
        char_fn = gbm_char_fn(S0, r, sigma, T)
        fft_call = fft_price(char_fn, S0, K, T, r)
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert fft_call == pytest.approx(bs_ref, rel=0.01)

    def test_gbm_otm_call(self):
        S0, K, r, sigma, T = 100.0, 110.0, 0.05, 0.20, 1.0
        char_fn = gbm_char_fn(S0, r, sigma, T)
        fft_call = fft_price(char_fn, S0, K, T, r)
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert fft_call == pytest.approx(bs_ref, rel=0.01)


# ---------------------------------------------------------------------------
# COS method with GBM
# ---------------------------------------------------------------------------


class TestCOSMethod:
    def test_gbm_call_matches_bs(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        char_fn = gbm_char_fn_log_ratio(r, sigma, T)
        cos_call = cos_price(char_fn, S0, K, T, r, option_type="call")
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert cos_call == pytest.approx(bs_ref, rel=0.02)

    def test_gbm_itm_call(self):
        S0, K, r, sigma, T = 100.0, 90.0, 0.05, 0.20, 1.0
        char_fn = gbm_char_fn_log_ratio(r, sigma, T)
        cos_call = cos_price(char_fn, S0, K, T, r, option_type="call")
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert cos_call == pytest.approx(bs_ref, rel=0.02)


# ---------------------------------------------------------------------------
# Heston characteristic function via FFT
# ---------------------------------------------------------------------------


class TestHestonFFT:
    def test_heston_price_reasonable(self):
        """Heston call price with known parameters should be within a reasonable range.

        Using benchmark params from literature:
        kappa=2, theta=0.04, xi=0.3, rho=-0.7, v0=0.04, mu=0.05
        For S0=100, K=100, T=1: the call price should be roughly 8-12
        (close to BS with sigma=sqrt(v0)=0.20).
        """
        S0, K, r, T = 100.0, 100.0, 0.05, 1.0
        heston = Heston(mu=r, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

        def char_fn(u):
            return heston.characteristic_function(u, T, log_spot=raw_np.log(S0))

        heston_call = fft_price(char_fn, S0, K, T, r)
        # With v0=theta=0.04 (equiv sigma=0.20) and skew from rho=-0.7,
        # ATM call should be close to BS value (~10.45) but slightly different
        bs_flat = bs_call(S0, K, T, r, 0.20)
        assert 5.0 < heston_call < 20.0
        # Should be in the neighborhood of BS
        assert abs(heston_call - bs_flat) < 3.0

    def test_heston_char_fn_at_zero(self):
        """phi(0) = 1 for any properly normalized characteristic function."""
        heston = Heston(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        cf_val = heston.characteristic_function(0.0, 1.0)
        # phi(0) = exp(0) = 1 (up to the mu*T drift term which gives exp(i*0*...) = 1)
        assert abs(cf_val) == pytest.approx(1.0, abs=1e-10)
