"""Tests for Black76 option pricing formulas."""

import pytest
from scipy.stats import norm

from trellis.models.black import black76_call, black76_put


class TestBlack76Call:

    def test_atm_call(self):
        """ATM caplet: F=K=0.05, sigma=0.20, T=1.0."""
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0
        d1 = (0 + 0.5 * 0.04) / 0.20  # = 0.1
        expected = F * norm.cdf(d1) - K * norm.cdf(d1 - 0.20)
        result = black76_call(F, K, sigma, T)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ F - K."""
        result = black76_call(0.10, 0.01, 0.20, 1.0)
        assert result == pytest.approx(0.09, abs=0.001)

    def test_deep_otm_call(self):
        """Deep OTM call ≈ 0."""
        result = black76_call(0.01, 0.10, 0.20, 1.0)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_zero_vol_itm(self):
        """Zero vol, ITM: call = max(F-K, 0) = F-K."""
        result = black76_call(0.06, 0.05, 0.0, 1.0)
        assert result == pytest.approx(0.01, abs=1e-10)

    def test_zero_vol_otm(self):
        """Zero vol, OTM: call = 0."""
        result = black76_call(0.04, 0.05, 0.0, 1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_monotonic_in_vol(self):
        """Call price increases with vol (ATM)."""
        F, K, T = 0.05, 0.05, 1.0
        p1 = black76_call(F, K, 0.10, T)
        p2 = black76_call(F, K, 0.20, T)
        p3 = black76_call(F, K, 0.40, T)
        assert p1 < p2 < p3

    def test_monotonic_in_T(self):
        """Call price increases with T (ATM)."""
        F, K, sigma = 0.05, 0.05, 0.20
        p1 = black76_call(F, K, sigma, 0.25)
        p2 = black76_call(F, K, sigma, 1.0)
        p3 = black76_call(F, K, sigma, 4.0)
        assert p1 < p2 < p3

    def test_non_negative(self):
        """Call price is always non-negative."""
        for F in [0.01, 0.05, 0.10]:
            for K in [0.01, 0.05, 0.10]:
                assert black76_call(F, K, 0.20, 1.0) >= 0


class TestBlack76Put:

    def test_non_negative(self):
        for F in [0.01, 0.05, 0.10]:
            for K in [0.01, 0.05, 0.10]:
                assert black76_put(F, K, 0.20, 1.0) >= 0

    def test_zero_vol_itm(self):
        result = black76_put(0.04, 0.05, 0.0, 1.0)
        assert result == pytest.approx(0.01, abs=1e-10)

    def test_zero_vol_otm(self):
        result = black76_put(0.06, 0.05, 0.0, 1.0)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestPutCallParity:

    def test_parity(self):
        """call - put = F - K for all parameter combinations."""
        params = [
            (0.06, 0.05, 0.25, 0.5),
            (0.04, 0.05, 0.20, 1.0),
            (0.05, 0.05, 0.30, 2.0),
            (0.10, 0.03, 0.15, 0.25),
        ]
        for F, K, sigma, T in params:
            call = black76_call(F, K, sigma, T)
            put = black76_put(F, K, sigma, T)
            assert call - put == pytest.approx(F - K, abs=1e-10), (
                f"Parity failed for F={F}, K={K}, σ={sigma}, T={T}"
            )
