"""Tests for Black76 and Garman-Kohlhagen option pricing formulas."""

import math

import pytest
from scipy.stats import norm

from trellis.core.differentiable import gradient
from trellis.models.black import (
    black76_call,
    black76_asset_or_nothing_call,
    black76_asset_or_nothing_put,
    black76_cash_or_nothing_call,
    black76_cash_or_nothing_put,
    black76_put,
    garman_kohlhagen_call,
    garman_kohlhagen_put,
)
from trellis.models.analytical import terminal_vanilla_from_basis


def _finite_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


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

    def test_autodiff_vega_matches_closed_form(self):
        """Autograd on Black76 call recovers the analytical vega."""
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0
        d1 = (0.0 + 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5)
        expected = F * T ** 0.5 * norm.pdf(d1)

        autodiff_vega = gradient(lambda vol: black76_call(F, K, vol, T))(sigma)

        assert autodiff_vega == pytest.approx(expected, rel=1e-10)

    def test_autodiff_vega_matches_finite_difference(self):
        """Autograd on Black76 call matches finite-difference vega."""
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0

        autodiff_vega = gradient(lambda vol: black76_call(F, K, vol, T))(sigma)
        fd_vega = _finite_difference(lambda vol: black76_call(F, K, vol, T), sigma)

        assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)


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


class TestBlack76CashOrNothing:

    def test_atm_call_matches_closed_form(self):
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0
        d1 = (0 + 0.5 * 0.04) / 0.20
        d2 = d1 - 0.20
        expected = norm.cdf(d2)
        result = black76_cash_or_nothing_call(F, K, sigma, T)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_zero_vol_intrinsic(self):
        assert black76_cash_or_nothing_call(0.06, 0.05, 0.0, 1.0) == pytest.approx(1.0)
        assert black76_cash_or_nothing_call(0.04, 0.05, 0.0, 1.0) == pytest.approx(0.0)
        assert black76_cash_or_nothing_put(0.04, 0.05, 0.0, 1.0) == pytest.approx(1.0)
        assert black76_cash_or_nothing_put(0.06, 0.05, 0.0, 1.0) == pytest.approx(0.0)

    def test_autodiff_vega_matches_finite_difference(self):
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0

        autodiff_vega = gradient(lambda vol: black76_cash_or_nothing_call(F, K, vol, T))(sigma)
        fd_vega = _finite_difference(
            lambda vol: black76_cash_or_nothing_call(F, K, vol, T),
            sigma,
        )

        assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)
        assert abs(autodiff_vega) > 0.0


class TestBlack76AssetOrNothing:

    def test_atm_call_matches_closed_form(self):
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0
        d1 = (0 + 0.5 * 0.04) / 0.20
        expected = F * norm.cdf(d1)
        result = black76_asset_or_nothing_call(F, K, sigma, T)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_zero_vol_intrinsic(self):
        assert black76_asset_or_nothing_call(0.06, 0.05, 0.0, 1.0) == pytest.approx(0.06)
        assert black76_asset_or_nothing_call(0.04, 0.05, 0.0, 1.0) == pytest.approx(0.0)
        assert black76_asset_or_nothing_put(0.04, 0.05, 0.0, 1.0) == pytest.approx(0.04)
        assert black76_asset_or_nothing_put(0.06, 0.05, 0.0, 1.0) == pytest.approx(0.0)

    def test_autodiff_vega_matches_finite_difference(self):
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0

        autodiff_vega = gradient(
            lambda vol: black76_asset_or_nothing_call(F, K, vol, T)
        )(sigma)
        fd_vega = _finite_difference(
            lambda vol: black76_asset_or_nothing_call(F, K, vol, T),
            sigma,
        )

        assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)


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

    def test_call_assembles_from_basis_claims(self):
        F, K, sigma, T = 0.05, 0.05, 0.20, 1.0
        expected_call = (
            black76_asset_or_nothing_call(F, K, sigma, T)
            - K * black76_cash_or_nothing_call(F, K, sigma, T)
        )
        expected_put = (
            K * black76_cash_or_nothing_put(F, K, sigma, T)
            - black76_asset_or_nothing_put(F, K, sigma, T)
        )

        assert black76_call(F, K, sigma, T) == pytest.approx(expected_call, abs=1e-10)
        assert black76_put(F, K, sigma, T) == pytest.approx(expected_put, abs=1e-10)


class TestGarmanKohlhagen:

    def test_matches_black76_on_fx_forward(self):
        spot = 1.10
        strike = 1.05
        sigma = 0.18
        T = 1.25
        df_domestic = 0.94
        df_foreign = 0.97

        forward = spot * df_foreign / df_domestic
        expected = df_domestic * black76_call(forward, strike, sigma, T)

        result = garman_kohlhagen_call(
            spot,
            strike,
            sigma,
            T,
            df_domestic,
            df_foreign,
        )

        assert result == pytest.approx(expected, abs=1e-10)

    def test_explicit_basis_assembly_matches_wrapper(self):
        spot = 1.10
        strike = 1.05
        sigma = 0.18
        T = 1.25
        df_domestic = 0.94
        df_foreign = 0.97

        forward = spot * df_foreign / df_domestic
        call_asset = black76_asset_or_nothing_call(forward, strike, sigma, T)
        call_cash = black76_cash_or_nothing_call(forward, strike, sigma, T)
        put_asset = black76_asset_or_nothing_put(forward, strike, sigma, T)
        put_cash = black76_cash_or_nothing_put(forward, strike, sigma, T)

        expected_call = df_domestic * terminal_vanilla_from_basis(
            "call",
            asset_value=call_asset,
            cash_value=call_cash,
            strike=strike,
        )
        expected_put = df_domestic * terminal_vanilla_from_basis(
            "put",
            asset_value=put_asset,
            cash_value=put_cash,
            strike=strike,
        )

        assert garman_kohlhagen_call(
            spot, strike, sigma, T, df_domestic, df_foreign
        ) == pytest.approx(expected_call, abs=1e-10)
        assert garman_kohlhagen_put(
            spot, strike, sigma, T, df_domestic, df_foreign
        ) == pytest.approx(expected_put, abs=1e-10)

    def test_put_call_parity(self):
        spot = 1.10
        strike = 1.05
        sigma = 0.22
        T = 2.0
        df_domestic = 0.91
        df_foreign = 0.96

        call = garman_kohlhagen_call(spot, strike, sigma, T, df_domestic, df_foreign)
        put = garman_kohlhagen_put(spot, strike, sigma, T, df_domestic, df_foreign)

        assert call - put == pytest.approx(
            spot * df_foreign - strike * df_domestic,
            abs=1e-10,
        )

    def test_zero_vol_reduces_to_discounted_intrinsic(self):
        spot = 1.10
        strike = 1.05
        df_domestic = 0.95
        df_foreign = 0.98

        call = garman_kohlhagen_call(spot, strike, 0.0, 1.0, df_domestic, df_foreign)
        put = garman_kohlhagen_put(spot, strike, 0.0, 1.0, df_domestic, df_foreign)

        assert call == pytest.approx(max(spot * df_foreign - strike * df_domestic, 0.0))
        assert put == pytest.approx(max(strike * df_domestic - spot * df_foreign, 0.0))

    def test_autodiff_delta_matches_closed_form(self):
        """Autograd on Garman-Kohlhagen call recovers the spot delta."""
        spot = 1.10
        strike = 1.05
        sigma = 0.22
        T = 2.0
        df_domestic = 0.91
        df_foreign = 0.96

        forward = spot * df_foreign / df_domestic
        d1 = (math.log(forward / strike) + 0.5 * sigma ** 2 * T) / (sigma * T ** 0.5)
        expected = df_foreign * norm.cdf(d1)

        autodiff_delta = gradient(
            lambda s: garman_kohlhagen_call(
                s,
                strike,
                sigma,
                T,
                df_domestic,
                df_foreign,
            )
        )(spot)

        assert autodiff_delta == pytest.approx(expected, rel=1e-10)

    def test_autodiff_delta_matches_finite_difference(self):
        """Autograd on Garman-Kohlhagen call matches finite-difference delta."""
        spot = 1.10
        strike = 1.05
        sigma = 0.22
        T = 2.0
        df_domestic = 0.91
        df_foreign = 0.96

        autodiff_delta = gradient(
            lambda s: garman_kohlhagen_call(
                s,
                strike,
                sigma,
                T,
                df_domestic,
                df_foreign,
            )
        )(spot)
        fd_delta = _finite_difference(
            lambda s: garman_kohlhagen_call(
                s,
                strike,
                sigma,
                T,
                df_domestic,
                df_foreign,
            ),
            spot,
        )

        assert autodiff_delta == pytest.approx(fd_delta, rel=1e-6, abs=1e-8)
