"""Tests for stochastic processes: GBM, Vasicek, CIR, HullWhite, MertonJumpDiffusion, SABR, LocalVol."""

import numpy as raw_np
import pytest

from trellis.models.processes.gbm import GBM
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.processes.vasicek import Vasicek
from trellis.models.processes.cir import CIR
from trellis.models.processes.hull_white import HullWhite
from trellis.models.processes.jump_diffusion import MertonJumpDiffusion
from trellis.models.processes.sabr import SABRProcess
from trellis.models.processes.local_vol import LocalVol


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------


class TestGBM:
    def test_exact_mean_matches_formula(self):
        """E[S_T] = S0 * exp(mu * T)."""
        mu, sigma = 0.05, 0.20
        gbm = GBM(mu, sigma)
        S0, T = 100.0, 1.0
        expected = S0 * raw_np.exp(mu * T)
        assert gbm.exact_mean(S0, 0, T) == pytest.approx(expected, rel=1e-12)

    def test_exact_variance_matches_formula(self):
        """Var[S_T] = S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)."""
        mu, sigma = 0.05, 0.20
        gbm = GBM(mu, sigma)
        S0, T = 100.0, 1.0
        expected = S0**2 * raw_np.exp(2 * mu * T) * (raw_np.exp(sigma**2 * T) - 1)
        assert gbm.exact_variance(S0, 0, T) == pytest.approx(expected, rel=1e-12)

    def test_exact_sample_positive(self):
        """GBM exact samples must be positive for positive S0."""
        gbm = GBM(mu=0.05, sigma=0.20)
        rng = raw_np.random.default_rng(42)
        S0, dt = 100.0, 0.01
        for _ in range(1000):
            dw = rng.standard_normal()
            s = gbm.exact_sample(S0, 0, dt, dw)
            assert s > 0

    def test_drift_and_diffusion(self):
        gbm = GBM(mu=0.05, sigma=0.20)
        assert gbm.drift(100.0, 0) == pytest.approx(5.0)
        assert gbm.diffusion(100.0, 0) == pytest.approx(20.0)


class TestCorrelatedGBM:
    def test_two_asset_shorthand_constructor_matches_array_form(self):
        shorthand = CorrelatedGBM(
            mu1=0.05,
            sigma1=0.20,
            mu2=0.03,
            sigma2=0.15,
            rho=0.35,
        )
        explicit = CorrelatedGBM(
            mu=[0.05, 0.03],
            sigma=[0.20, 0.15],
            corr=[[1.0, 0.35], [0.35, 1.0]],
        )

        raw_np.testing.assert_allclose(shorthand.mu, explicit.mu, atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(shorthand.sigma, explicit.sigma, atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(shorthand.corr, explicit.corr, atol=0.0, rtol=0.0)

    def test_generated_alias_constructor_matches_explicit_form(self):
        alias = CorrelatedGBM(
            spots=[100.0, 80.0],
            rates=[0.05, 0.04],
            vols=[0.20, 0.25],
            correlation=[[1.0, 0.3], [0.3, 1.0]],
            div_yields=[0.01, 0.02],
        )
        explicit = CorrelatedGBM(
            mu=[0.05, 0.04],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.3], [0.3, 1.0]],
            dividend_yield=[0.01, 0.02],
        )

        raw_np.testing.assert_allclose(alias.mu, explicit.mu, atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(alias.sigma, explicit.sigma, atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(alias.corr, explicit.corr, atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(alias.dividend_yield, explicit.dividend_yield, atol=0.0, rtol=0.0)

    def test_spot_prices_alias_is_accepted(self):
        process = CorrelatedGBM(
            spot_prices=[100.0, 80.0],
            rates=[0.05, 0.04],
            vols=[0.20, 0.25],
            corr_matrix=[[1.0, 0.3], [0.3, 1.0]],
            dividends=[0.01, 0.02],
        )

        raw_np.testing.assert_allclose(process.mu, raw_np.array([0.05, 0.04]), atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(process.sigma, raw_np.array([0.20, 0.25]), atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(process.corr, raw_np.array([[1.0, 0.3], [0.3, 1.0]]), atol=0.0, rtol=0.0)
        raw_np.testing.assert_allclose(process.dividend_yield, raw_np.array([0.01, 0.02]), atol=0.0, rtol=0.0)

    def test_drift_uses_dividend_adjusted_mu(self):
        process = CorrelatedGBM(
            mu=[0.05, 0.04],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.3], [0.3, 1.0]],
            dividend_yield=[0.01, 0.02],
        )
        x = raw_np.array([[100.0, 80.0], [110.0, 90.0]])

        drift = process.drift(x, 0.0)

        raw_np.testing.assert_allclose(
            drift,
            x * raw_np.array([0.04, 0.02]),
            atol=0.0,
            rtol=0.0,
        )

    def test_diffusion_returns_factor_loadings(self):
        process = CorrelatedGBM(
            mu=[0.05, 0.04],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.5], [0.5, 1.0]],
        )
        x = raw_np.array([[100.0, 80.0]])

        diffusion = process.diffusion(x, 0.0)

        assert diffusion.shape == (1, 2, 2)
        raw_np.testing.assert_allclose(
            diffusion[0, :, 0],
            raw_np.array([20.0, 10.0]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_exact_sample_zero_shocks_matches_deterministic_growth(self):
        process = CorrelatedGBM(
            mu=[0.05, 0.04],
            sigma=[0.20, 0.25],
            corr=[[1.0, 0.2], [0.2, 1.0]],
            dividend_yield=[0.01, 0.02],
        )
        x0 = raw_np.array([[100.0, 80.0]])
        dt = 0.25
        dw = raw_np.zeros((1, 2))

        sampled = process.exact_sample(x0, 0.0, dt, dw)
        expected = x0 * raw_np.exp((raw_np.array([0.04, 0.02]) - 0.5 * raw_np.array([0.20, 0.25]) ** 2) * dt)

        raw_np.testing.assert_allclose(sampled, expected, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# Vasicek
# ---------------------------------------------------------------------------


class TestVasicek:
    def test_exact_mean_converges_to_long_term(self):
        """As dt -> inf, mean -> b."""
        a, b, sigma = 0.5, 0.05, 0.01
        v = Vasicek(a, b, sigma)
        r0 = 0.10
        mean_long = v.exact_mean(r0, 0, 100.0)
        assert mean_long == pytest.approx(b, abs=1e-6)

    def test_exact_variance_formula(self):
        """Var = sigma^2 / (2a) * (1 - exp(-2a*dt))."""
        a, b, sigma = 0.5, 0.05, 0.01
        v = Vasicek(a, b, sigma)
        dt = 1.0
        expected = (sigma**2 / (2 * a)) * (1 - raw_np.exp(-2 * a * dt))
        assert v.exact_variance(0.03, 0, dt) == pytest.approx(expected, rel=1e-12)

    def test_exact_mean_at_t0(self):
        """At dt=0, mean should equal current value."""
        v = Vasicek(0.5, 0.05, 0.01)
        assert v.exact_mean(0.08, 0, 0.0) == pytest.approx(0.08, abs=1e-14)


# ---------------------------------------------------------------------------
# CIR
# ---------------------------------------------------------------------------


class TestCIR:
    def test_drift_correct(self):
        cir = CIR(a=0.5, b=0.05, sigma=0.1)
        # drift = a*(b - x) at x=0.03: 0.5*(0.05 - 0.03) = 0.01
        assert cir.drift(0.03, 0) == pytest.approx(0.01, rel=1e-12)

    def test_diffusion_correct(self):
        cir = CIR(a=0.5, b=0.05, sigma=0.1)
        # diffusion = sigma * sqrt(x) at x=0.04 => 0.1 * 0.2 = 0.02
        assert cir.diffusion(0.04, 0) == pytest.approx(0.02, rel=1e-12)

    def test_feller_condition_positivity(self):
        """When 2*a*b > sigma^2 (Feller condition), exact_mean stays positive."""
        a, b, sigma = 1.0, 0.05, 0.1
        assert 2 * a * b > sigma**2, "Feller condition not met"
        cir = CIR(a, b, sigma)
        r0 = 0.001  # start near zero
        for dt in [0.1, 0.5, 1.0, 5.0]:
            assert cir.exact_mean(r0, 0, dt) > 0

    def test_diffusion_zero_at_zero(self):
        """Diffusion vanishes at x=0, preventing negative values."""
        cir = CIR(a=0.5, b=0.05, sigma=0.1)
        assert cir.diffusion(0.0, 0) == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# HullWhite
# ---------------------------------------------------------------------------


class TestHullWhite:
    def test_drift_equals_theta_minus_a_times_r(self):
        hw = HullWhite(a=0.1, sigma=0.01, theta=0.05)
        r = 0.03
        assert hw.drift(r, 0) == pytest.approx(0.05 - 0.1 * 0.03, rel=1e-12)

    def test_exact_mean_reverts_to_theta_over_a(self):
        """As dt -> inf, mean -> theta / a."""
        a, theta = 0.1, 0.05
        hw = HullWhite(a=a, sigma=0.01, theta=theta)
        r0 = 0.10
        mean_long = hw.exact_mean(r0, 0, 200.0)
        assert mean_long == pytest.approx(theta / a, abs=1e-4)

    def test_theta_fn_used_when_provided(self):
        hw = HullWhite(a=0.1, sigma=0.01, theta_fn=lambda t: 0.05 + 0.01 * t)
        assert hw.drift(0.03, 1.0) == pytest.approx(0.06 - 0.1 * 0.03, rel=1e-12)


# ---------------------------------------------------------------------------
# MertonJumpDiffusion
# ---------------------------------------------------------------------------


class TestMertonJumpDiffusion:
    def test_drift_includes_jump_compensator(self):
        """drift = (mu - lam*k) * x, where k = E[e^J - 1]."""
        mu, sigma, lam = 0.05, 0.20, 1.0
        jump_mean, jump_vol = -0.05, 0.10
        mjd = MertonJumpDiffusion(mu, sigma, lam, jump_mean, jump_vol)
        k = raw_np.exp(jump_mean + 0.5 * jump_vol**2) - 1
        x = 100.0
        expected_drift = (mu - lam * k) * x
        assert mjd.drift(x, 0) == pytest.approx(expected_drift, rel=1e-10)

    def test_sample_jump_returns_one_when_no_jumps(self):
        """With lam=0, no jumps occur and multiplicative factor is 1.0."""
        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=0.0,
                                  jump_mean=0.0, jump_vol=0.1)
        rng = raw_np.random.default_rng(42)
        for _ in range(100):
            assert mjd.sample_jump(0.01, rng) == 1.0

    def test_compensator_formula(self):
        jump_mean, jump_vol = -0.05, 0.10
        mjd = MertonJumpDiffusion(0.05, 0.20, 1.0, jump_mean, jump_vol)
        expected_k = raw_np.exp(jump_mean + 0.5 * jump_vol**2) - 1
        assert mjd.k == pytest.approx(expected_k, rel=1e-12)


# ---------------------------------------------------------------------------
# SABRProcess
# ---------------------------------------------------------------------------


class TestSABRProcess:
    def test_atm_implied_vol_approx_alpha_times_f_beta_minus_1(self):
        """ATM implied vol ~ alpha * F^(beta-1) for small T and nu."""
        alpha, beta, rho, nu = 0.20, 0.5, 0.0, 0.01  # small nu
        sabr = SABRProcess(alpha, beta, rho, nu)
        F = 100.0
        T = 0.01  # very small T to minimize higher-order terms
        iv = sabr.implied_vol(F, F, T)
        approx = alpha * F ** (beta - 1)
        assert iv == pytest.approx(approx, rel=0.05)

    def test_implied_vol_positive_for_reasonable_params(self):
        sabr = SABRProcess(alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        F = 100.0
        for K in [80, 90, 100, 110, 120]:
            iv = sabr.implied_vol(F, float(K), 1.0)
            assert iv > 0

    def test_implied_vol_smile_shape(self):
        """With negative rho, OTM puts have higher vol (skew)."""
        sabr = SABRProcess(alpha=0.20, beta=0.5, rho=-0.5, nu=0.4)
        F = 100.0
        T = 1.0
        vol_low = sabr.implied_vol(F, 80.0, T)
        vol_atm = sabr.implied_vol(F, 100.0, T)
        assert vol_low > vol_atm


# ---------------------------------------------------------------------------
# LocalVol
# ---------------------------------------------------------------------------


class TestLocalVol:
    def test_from_flat_equivalent_to_gbm(self):
        """Flat local vol: diffusion = sigma * x, same as GBM."""
        sigma = 0.20
        lv = LocalVol.from_flat(mu=0.05, sigma=sigma)
        x = 100.0
        assert lv.diffusion(x, 0) == pytest.approx(sigma * x, rel=1e-12)
        assert lv.drift(x, 0) == pytest.approx(0.05 * x, rel=1e-12)

    def test_custom_vol_fn(self):
        vol_fn = lambda s, t: 0.20 + 0.001 * (s - 100)
        lv = LocalVol(mu=0.05, vol_fn=vol_fn)
        assert lv.diffusion(110.0, 0) == pytest.approx(0.21 * 110.0, rel=1e-10)
