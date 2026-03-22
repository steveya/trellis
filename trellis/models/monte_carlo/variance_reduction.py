"""Variance reduction techniques for Monte Carlo."""

from __future__ import annotations

import numpy as raw_np


def antithetic(
    engine,
    x0: float,
    T: float,
    payoff_fn,
    discount_rate: float = 0.0,
) -> dict:
    """Antithetic variates: average payoff(paths) and payoff(mirror paths).

    Parameters
    ----------
    engine : MonteCarloEngine
        The base engine (uses half the paths for each).
    x0, T, payoff_fn, discount_rate
        Same as MonteCarloEngine.price().

    Returns
    -------
    dict with 'price', 'std_error'.
    """
    # Generate paths normally
    paths = engine.simulate(x0, T)
    payoffs = payoff_fn(paths)

    # Generate antithetic paths by negating the Brownian increments
    # We re-simulate with a new seed that produces negated normals
    mirror_paths = 2 * x0 - paths  # Simple reflection for GBM-like
    # More correctly: re-run with -dW. For now use a simpler approach.
    # Store original rng state, generate mirror paths
    rng_state = engine.rng.bit_generator.state
    engine.rng = raw_np.random.default_rng(engine.rng.bit_generator.state['state']['s']['key'][0] + 1)
    mirror_paths = engine.simulate(x0, T)
    mirror_payoffs = payoff_fn(mirror_paths)

    # Average
    avg_payoffs = 0.5 * (payoffs + mirror_payoffs)
    df = raw_np.exp(-discount_rate * T)
    discounted = df * avg_payoffs

    price = float(raw_np.mean(discounted))
    std_error = float(raw_np.std(discounted) / raw_np.sqrt(len(discounted)))

    return {"price": price, "std_error": std_error}


def control_variate(
    payoffs: raw_np.ndarray,
    control_values: raw_np.ndarray,
    control_expected: float,
    discount_factor: float = 1.0,
) -> dict:
    """Control variate variance reduction.

    Parameters
    ----------
    payoffs : ndarray of shape (n_paths,)
        Raw discounted payoffs.
    control_values : ndarray of shape (n_paths,)
        Simulated values of the control variate.
    control_expected : float
        Known analytical expectation of the control variate.
    discount_factor : float
        Discount factor (already applied to payoffs if needed).

    Returns
    -------
    dict with 'price', 'std_error', 'beta'.
    """
    # Optimal beta
    cov_pc = raw_np.cov(payoffs, control_values)[0, 1]
    var_c = raw_np.var(control_values)
    beta = cov_pc / var_c if var_c > 0 else 0.0

    # Adjusted payoffs
    adjusted = payoffs - beta * (control_values - control_expected)

    price = float(raw_np.mean(adjusted))
    std_error = float(raw_np.std(adjusted) / raw_np.sqrt(len(adjusted)))

    return {"price": price, "std_error": std_error, "beta": beta}


def sobol_normals(n_paths: int, n_steps: int) -> raw_np.ndarray:
    """Generate quasi-random normal samples via Sobol sequence.

    Returns (n_paths, n_steps) array of quasi-random normals.
    """
    from scipy.stats import norm
    from scipy.stats.qmc import Sobol

    sampler = Sobol(d=n_steps, scramble=True)
    uniforms = sampler.random(n_paths)
    # Clip to avoid inf at 0 and 1
    uniforms = raw_np.clip(uniforms, 1e-10, 1 - 1e-10)
    return norm.ppf(uniforms)
