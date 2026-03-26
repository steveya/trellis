"""T12: Variance reduction — plain MC vs antithetic vs control variate.

Measure variance reduction ratios. Cross-validate against BS analytical.

Parameters:
  S0=100, K=100, r=0.05, sigma=0.20, T=1.0, n_paths=50K, n_steps=100

Methods:
  1. Plain MC (Exact scheme)
  2. Antithetic variates (Antithetic(Exact()) wrapper)
  3. Control variate (stock price as control, E[S_T] = S0*exp(rT) known)

Tolerances: all <= 1%.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.schemes import Antithetic, Exact
from trellis.models.monte_carlo.variance_reduction import control_variate
from trellis.models.processes.gbm import GBM

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0

N_PATHS = 50_000
N_STEPS = 100
SEED = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_call_price(S, K, r, sigma, T):
    """Black-Scholes European call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


BS_PRICE = bs_call_price(S0, K, R, SIGMA, T)


def _run_plain_mc(seed=SEED):
    """Plain MC with Exact scheme. Returns (price, std_error, payoffs, terminal_S)."""
    process = GBM(mu=R, sigma=SIGMA)
    engine = MonteCarloEngine(
        process, n_paths=N_PATHS, n_steps=N_STEPS, seed=seed, scheme=Exact(),
    )
    paths = engine.simulate(S0, T)
    terminal_S = paths[:, -1]
    payoffs = np.maximum(terminal_S - K, 0.0)
    df = np.exp(-R * T)
    discounted = df * payoffs

    price = float(np.mean(discounted))
    std_error = float(np.std(discounted) / np.sqrt(N_PATHS))
    return price, std_error, discounted, terminal_S


def _run_antithetic_mc(seed=SEED):
    """Antithetic MC with Antithetic(Exact()) wrapper."""
    process = GBM(mu=R, sigma=SIGMA)
    engine = MonteCarloEngine(
        process, n_paths=N_PATHS, n_steps=N_STEPS, seed=seed,
        scheme=Antithetic(Exact()),
    )
    paths = engine.simulate(S0, T)
    terminal_S = paths[:, -1]

    # Antithetic averaging: pair up the first and second halves
    half = N_PATHS // 2
    payoffs_pos = np.maximum(terminal_S[:half] - K, 0.0)
    payoffs_neg = np.maximum(terminal_S[half:] - K, 0.0)
    avg_payoffs = 0.5 * (payoffs_pos + payoffs_neg)

    df = np.exp(-R * T)
    discounted = df * avg_payoffs

    price = float(np.mean(discounted))
    std_error = float(np.std(discounted) / np.sqrt(half))
    return price, std_error


def _run_cv_mc(seed=SEED):
    """Control variate MC using stock price as control."""
    process = GBM(mu=R, sigma=SIGMA)
    engine = MonteCarloEngine(
        process, n_paths=N_PATHS, n_steps=N_STEPS, seed=seed, scheme=Exact(),
    )
    paths = engine.simulate(S0, T)
    terminal_S = paths[:, -1]
    payoffs = np.maximum(terminal_S - K, 0.0)
    df = np.exp(-R * T)
    discounted = df * payoffs

    # Control variate: S_T with known expectation E[S_T] = S0 * exp(r*T)
    control_values = terminal_S
    control_expected = S0 * np.exp(R * T)

    result = control_variate(discounted, control_values, control_expected)
    return result["price"], result["std_error"]


# ---------------------------------------------------------------------------
# Cache results to avoid re-running expensive simulations
# ---------------------------------------------------------------------------
_cache: dict[str, tuple] = {}


def _get_plain():
    if "plain" not in _cache:
        _cache["plain"] = _run_plain_mc()
    return _cache["plain"]


def _get_antithetic():
    if "anti" not in _cache:
        _cache["anti"] = _run_antithetic_mc()
    return _cache["anti"]


def _get_cv():
    if "cv" not in _cache:
        _cache["cv"] = _run_cv_mc()
    return _cache["cv"]


# ---------------------------------------------------------------------------
# Test 1: Plain MC within 1% of BS
# ---------------------------------------------------------------------------

class TestPlainMC:
    """Plain MC (Exact scheme) should produce a price within 1% of BS."""

    def test_plain_mc_vs_bs(self):
        price, std_error, _, _ = _get_plain()
        rel_err = abs(price - BS_PRICE) / BS_PRICE
        assert rel_err < 0.01, (
            f"Plain MC={price:.4f} vs BS={BS_PRICE:.4f}, "
            f"rel_err={rel_err:.4f}, std_error={std_error:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Antithetic MC within 1% of BS
# ---------------------------------------------------------------------------

class TestAntitheticMC:
    """Antithetic MC should produce a price within 1% of BS."""

    def test_antithetic_mc_vs_bs(self):
        price, std_error = _get_antithetic()
        rel_err = abs(price - BS_PRICE) / BS_PRICE
        assert rel_err < 0.01, (
            f"Antithetic MC={price:.4f} vs BS={BS_PRICE:.4f}, "
            f"rel_err={rel_err:.4f}, std_error={std_error:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Control variate MC within 1% of BS
# ---------------------------------------------------------------------------

class TestControlVariateMC:
    """Control variate MC should produce a price within 1% of BS."""

    def test_cv_mc_vs_bs(self):
        price, std_error = _get_cv()
        rel_err = abs(price - BS_PRICE) / BS_PRICE
        assert rel_err < 0.01, (
            f"CV MC={price:.4f} vs BS={BS_PRICE:.4f}, "
            f"rel_err={rel_err:.4f}, std_error={std_error:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Antithetic reduces variance vs plain
# ---------------------------------------------------------------------------

class TestAntitheticReducesVariance:
    """Standard error of antithetic should be < standard error of plain."""

    def test_antithetic_se_less_than_plain(self):
        _, se_plain, _, _ = _get_plain()
        _, se_anti = _get_antithetic()
        assert se_anti < se_plain, (
            f"Antithetic SE={se_anti:.6f} should be < Plain SE={se_plain:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: Control variate reduces variance vs plain
# ---------------------------------------------------------------------------

class TestCVReducesVariance:
    """Standard error of CV should be < standard error of plain."""

    def test_cv_se_less_than_plain(self):
        _, se_plain, _, _ = _get_plain()
        _, se_cv = _get_cv()
        assert se_cv < se_plain, (
            f"CV SE={se_cv:.6f} should be < Plain SE={se_plain:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 6: Control variate reduces variance vs antithetic
# ---------------------------------------------------------------------------

class TestCVBetterThanAntithetic:
    """For European call under GBM, CV with stock control should beat antithetic."""

    def test_cv_se_less_than_antithetic(self):
        _, se_anti = _get_antithetic()
        _, se_cv = _get_cv()
        assert se_cv < se_anti, (
            f"CV SE={se_cv:.6f} should be < Antithetic SE={se_anti:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Variance reduction ratios
# ---------------------------------------------------------------------------

class TestVarianceReductionRatios:
    """Measure VR ratios: antithetic ~2, CV >2."""

    def test_antithetic_vr_ratio(self):
        _, se_plain, _, _ = _get_plain()
        _, se_anti = _get_antithetic()
        # VR ratio = Var_plain / Var_anti = (SE_plain / SE_anti)^2
        vr_anti = (se_plain / se_anti) ** 2
        assert vr_anti > 1.0, (
            f"VR_antithetic={vr_anti:.2f}, expected > 1.0"
        )
        # Print for diagnostics
        print(f"\nVR_antithetic = {vr_anti:.2f} (expected ~2)")

    def test_cv_vr_ratio(self):
        _, se_plain, _, _ = _get_plain()
        _, se_cv = _get_cv()
        vr_cv = (se_plain / se_cv) ** 2
        assert vr_cv > 2.0, (
            f"VR_cv={vr_cv:.2f}, expected > 2.0"
        )
        print(f"\nVR_cv = {vr_cv:.2f} (expected >2)")


# ---------------------------------------------------------------------------
# Test 8: All methods agree on price
# ---------------------------------------------------------------------------

class TestAllMethodsAgree:
    """Plain, antithetic, and CV all produce prices within 1% of each other."""

    def test_all_agree(self):
        price_plain, _, _, _ = _get_plain()
        price_anti, _ = _get_antithetic()
        price_cv, _ = _get_cv()

        prices = {"plain": price_plain, "antithetic": price_anti, "cv": price_cv}

        for name_a, p_a in prices.items():
            for name_b, p_b in prices.items():
                if name_a >= name_b:
                    continue
                rel_diff = abs(p_a - p_b) / abs(0.5 * (p_a + p_b))
                assert rel_diff < 0.01, (
                    f"{name_a}={p_a:.4f} vs {name_b}={p_b:.4f}, "
                    f"rel_diff={rel_diff:.4f}"
                )
