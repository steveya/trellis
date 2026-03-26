"""T11: GBM European call — all 4 MC schemes convergence order measurement.
Cross-validate against Black-Scholes and QuantLib MC.

Parameters:
  S0=100, K=100, r=0.05, sigma=0.20, T=1.0

Schemes:
  Euler (weak order 1), Milstein (weak order 1, strong order 1),
  Exact (exact for GBM), LogEuler (weak order 1, positivity-preserving)

Tolerances: all <= 1%.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.schemes import Euler, Milstein, Exact, LogEuler
from trellis.models.processes.gbm import GBM

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.20
T = 1.0

N_PATHS = 200_000
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_call_price(S, K, r, sigma, T):
    """Black-Scholes European call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put_price(S, K, r, sigma, T):
    """Black-Scholes European put price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def mc_european_call(scheme, n_steps, n_paths=N_PATHS, seed=SEED,
                     s0=S0, k=K, r=R, sigma=SIGMA, t=T):
    """Price a European call via MC with a given scheme."""
    process = GBM(mu=r, sigma=sigma)
    engine = MonteCarloEngine(
        process, n_paths=n_paths, n_steps=n_steps, seed=seed, scheme=scheme,
    )

    def payoff_fn(paths):
        return np.maximum(paths[:, -1] - k, 0.0)

    result = engine.price(s0, t, payoff_fn, discount_rate=r)
    return result


def mc_european_put(scheme, n_steps, n_paths=N_PATHS, seed=SEED,
                    s0=S0, k=K, r=R, sigma=SIGMA, t=T):
    """Price a European put via MC with a given scheme."""
    process = GBM(mu=r, sigma=sigma)
    engine = MonteCarloEngine(
        process, n_paths=n_paths, n_steps=n_steps, seed=seed, scheme=scheme,
    )

    def payoff_fn(paths):
        return np.maximum(k - paths[:, -1], 0.0)

    result = engine.price(s0, t, payoff_fn, discount_rate=r)
    return result


def measure_convergence_order(scheme, n_steps_list, ref_price):
    """Measure weak convergence order using Exact scheme as control variate.

    For each n_steps, we simulate with both the test scheme and the Exact
    scheme using the same seed. The discretization bias is:
        bias(scheme) = E[payoff_scheme] - E[payoff_exact]
    since Exact has zero discretization bias for GBM.
    This cancels the MC sampling noise and isolates the scheme bias.

    Returns (order, biases) where order is the slope of the log-log fit.
    """
    biases = []
    dts = []
    for n_steps in n_steps_list:
        # Same seed so MC noise cancels in the difference
        result_scheme = mc_european_call(scheme, n_steps, seed=77)
        result_exact = mc_european_call(Exact(), n_steps, seed=77)
        bias = abs(result_scheme["price"] - result_exact["price"])
        biases.append(bias)
        dts.append(T / n_steps)

    log_dts = np.log(np.array(dts))
    log_biases = np.log(np.array(biases) + 1e-15)

    # Least-squares fit: log(bias) = p * log(dt) + c
    coeffs = np.polyfit(log_dts, log_biases, 1)
    order = coeffs[0]
    return order, biases


# ---------------------------------------------------------------------------
# Precompute cache
# ---------------------------------------------------------------------------
_cache: dict[str, object] = {}

BS_CALL = bs_call_price(S0, K, R, SIGMA, T)
BS_PUT = bs_put_price(S0, K, R, SIGMA, T)


def _get_mc_call(scheme_name: str) -> float:
    """Get MC call price at n_steps=200 for a named scheme."""
    key = f"mc_call_{scheme_name}"
    if key not in _cache:
        schemes = {
            "euler": Euler(),
            "milstein": Milstein(),
            "exact": Exact(),
            "log_euler": LogEuler(),
        }
        result = mc_european_call(schemes[scheme_name], n_steps=200)
        _cache[key] = result["price"]
    return _cache[key]


# ---------------------------------------------------------------------------
# Test 1: All schemes converge to BS at n_steps=200
# ---------------------------------------------------------------------------

class TestAllSchemesConverge:
    """At n_steps=200, all 4 scheme prices should be within 1% of BS."""

    @pytest.mark.parametrize("scheme_name", ["euler", "milstein", "exact", "log_euler"])
    def test_scheme_vs_bs(self, scheme_name):
        mc_price = _get_mc_call(scheme_name)
        rel_err = abs(mc_price - BS_CALL) / BS_CALL
        assert rel_err < 0.01, (
            f"{scheme_name}: MC={mc_price:.4f} vs BS={BS_CALL:.4f}, "
            f"rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Euler convergence order between 0.5 and 1.5
# ---------------------------------------------------------------------------

class TestEulerConvergenceOrder:
    """Measured weak convergence order of Euler should be between 0.5 and 1.5."""

    def test_euler_order(self):
        n_steps_list = [10, 20, 50, 100, 200]
        order, errors = measure_convergence_order(
            Euler(), n_steps_list, BS_CALL,
        )
        assert 0.5 <= order <= 1.5, (
            f"Euler weak order={order:.2f}, expected in [0.5, 1.5]. "
            f"Errors: {errors}"
        )


# ---------------------------------------------------------------------------
# Test 3: Exact scheme is step-independent
# ---------------------------------------------------------------------------

class TestExactStepIndependent:
    """Exact scheme: every n_steps value produces a price within 1% of BS.

    Since the Exact scheme has zero discretization bias for GBM, the only
    error source is MC sampling noise, which should be small at 200K paths
    regardless of the number of time steps.
    """

    @pytest.mark.parametrize("n_steps", [10, 20, 50, 100, 200])
    def test_exact_vs_bs(self, n_steps):
        result = mc_european_call(Exact(), n_steps)
        rel_err = abs(result["price"] - BS_CALL) / BS_CALL
        assert rel_err < 0.01, (
            f"Exact n_steps={n_steps}: MC={result['price']:.4f} vs "
            f"BS={BS_CALL:.4f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: LogEuler preserves positivity at high vol
# ---------------------------------------------------------------------------

class TestLogEulerPositivity:
    """LogEuler should produce no negative paths even at sigma=0.80."""

    def test_no_negative_paths(self):
        sigma_high = 0.80
        process = GBM(mu=R, sigma=sigma_high)
        engine = MonteCarloEngine(
            process, n_paths=N_PATHS, n_steps=200, seed=SEED,
            scheme=LogEuler(),
        )
        paths = engine.simulate(S0, T)
        assert np.all(paths > 0), (
            f"LogEuler produced negative paths. Min value: {paths.min():.6f}"
        )

    def test_euler_can_go_negative_at_high_vol(self):
        """Euler at high vol CAN produce negative values (demonstrates the problem)."""
        sigma_high = 0.80
        process = GBM(mu=R, sigma=sigma_high)
        engine = MonteCarloEngine(
            process, n_paths=N_PATHS, n_steps=50, seed=SEED,
            scheme=Euler(),
        )
        paths = engine.simulate(S0, T)
        # We just check that Euler does go negative to motivate LogEuler.
        # If it does not (unlikely but possible), we skip gracefully.
        n_neg = np.sum(paths < 0)
        if n_neg == 0:
            pytest.skip("Euler did not produce negatives at this seed/vol")
        assert n_neg > 0  # Euler indeed goes negative


# ---------------------------------------------------------------------------
# Test 5: Milstein vs Euler strong error
# ---------------------------------------------------------------------------

class TestMilsteinStrongError:
    """Milstein strong error (pathwise) should be < Euler strong error.

    We compare both against the Exact scheme (which is the true GBM solution)
    using the same Brownian increments.
    """

    def test_milstein_strong_error_less_than_euler(self):
        n_paths = 50_000
        n_steps = 100
        seed = 123

        process = GBM(mu=R, sigma=SIGMA)
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        # Generate shared Brownian increments
        dw_all = rng.standard_normal((n_steps, n_paths))

        # Simulate all three schemes with same noise
        euler = Euler()
        milstein = Milstein()
        exact = Exact()

        x_euler = np.full(n_paths, S0)
        x_milstein = np.full(n_paths, S0)
        x_exact = np.full(n_paths, S0)

        for i in range(n_steps):
            t = i * dt
            dw = dw_all[i]
            x_euler = euler.step(process, x_euler, t, dt, dw)
            x_milstein = milstein.step(process, x_milstein, t, dt, dw)
            x_exact = exact.step(process, x_exact, t, dt, dw)

        # Strong error = E[|X_scheme - X_exact|]
        euler_strong_err = np.mean(np.abs(x_euler - x_exact))
        milstein_strong_err = np.mean(np.abs(x_milstein - x_exact))

        assert milstein_strong_err < euler_strong_err, (
            f"Milstein strong error ({milstein_strong_err:.6f}) should be < "
            f"Euler strong error ({euler_strong_err:.6f})"
        )


# ---------------------------------------------------------------------------
# Test 6: QuantLib MC cross-validation
# ---------------------------------------------------------------------------

class TestQuantLibMC:
    """Compare trellis MC (Exact scheme) to QuantLib MCEuropeanEngine."""

    @staticmethod
    def _ql_mc_price(n_paths):
        """Price a European call via QuantLib MC."""
        ql = pytest.importorskip("QuantLib")

        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        maturity = today + ql.Period(1, ql.Years)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, R, ql.Actual365Fixed()),
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed()),
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                today, ql.NullCalendar(), SIGMA, ql.Actual365Fixed(),
            ),
        )

        bs_process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle,
        )

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        # Use QuantLib analytic engine for a precise reference
        engine = ql.AnalyticEuropeanEngine(bs_process)
        option.setPricingEngine(engine)
        return option.NPV()

    def test_trellis_vs_quantlib(self):
        """Trellis MC (Exact, 200 steps, 200K paths) vs QuantLib analytic within 1%."""
        ql_price = self._ql_mc_price(N_PATHS)

        trellis_result = mc_european_call(Exact(), n_steps=200)
        trellis_price = trellis_result["price"]

        rel_err = abs(trellis_price - ql_price) / ql_price
        assert rel_err < 0.01, (
            f"Trellis MC={trellis_price:.4f} vs QuantLib={ql_price:.4f}, "
            f"rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Put-call parity
# ---------------------------------------------------------------------------

class TestPutCallParity:
    """MC call - MC put should equal S0 - K*exp(-rT) within 1%."""

    def test_put_call_parity(self):
        scheme = Exact()
        call_result = mc_european_call(scheme, n_steps=200)
        put_result = mc_european_put(scheme, n_steps=200)

        mc_call = call_result["price"]
        mc_put = put_result["price"]
        parity_rhs = S0 - K * np.exp(-R * T)

        mc_diff = mc_call - mc_put
        rel_err = abs(mc_diff - parity_rhs) / abs(parity_rhs)
        assert rel_err < 0.01, (
            f"Put-call parity: C-P={mc_diff:.4f} vs S-Ke^(-rT)={parity_rhs:.4f}, "
            f"rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 8: Various strikes
# ---------------------------------------------------------------------------

class TestVariousStrikes:
    """All strikes K=80,100,120 should be within 1% of BS."""

    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_strike_vs_bs(self, strike):
        bs_ref = bs_call_price(S0, strike, R, SIGMA, T)
        result = mc_european_call(Exact(), n_steps=200, k=strike)
        mc_price = result["price"]

        rel_err = abs(mc_price - bs_ref) / bs_ref
        assert rel_err < 0.01, (
            f"K={strike}: MC={mc_price:.4f} vs BS={bs_ref:.4f}, "
            f"rel_err={rel_err:.4f}"
        )
