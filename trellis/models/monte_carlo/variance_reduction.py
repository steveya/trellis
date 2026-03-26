"""Variance reduction techniques for Monte Carlo."""

from __future__ import annotations

import math

import numpy as raw_np

from trellis.models.monte_carlo.brownian_bridge import brownian_bridge


def antithetic_normals(
    n_paths: int,
    n_steps: int,
    n_factors: int = 1,
    *,
    rng=None,
) -> raw_np.ndarray:
    """Generate paired standard-normal shocks with antithetic symmetry."""
    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic sampling")
    if rng is None:
        rng = raw_np.random.default_rng()

    half = n_paths // 2
    if n_factors == 1:
        base = rng.standard_normal((half, n_steps))
        return raw_np.concatenate([base, -base], axis=0)

    base = rng.standard_normal((half, n_steps, n_factors))
    return raw_np.concatenate([base, -base], axis=0)


def brownian_bridge_increments(normals: raw_np.ndarray, T: float) -> raw_np.ndarray:
    """Map bridge-order normals to per-step standard Brownian increments."""
    normals_arr = raw_np.asarray(normals, dtype=float)
    if normals_arr.ndim not in {2, 3}:
        raise ValueError("normals must have shape (n_paths, n_steps) or (n_paths, n_steps, n_factors)")

    n_paths, n_steps = normals_arr.shape[:2]
    dt = T / n_steps
    inv_sqrt_dt = 1.0 / math.sqrt(dt)

    if normals_arr.ndim == 2:
        bridge = brownian_bridge(T, n_steps, n_paths, bridge_shocks=normals_arr)
        return raw_np.diff(bridge, axis=1) * inv_sqrt_dt

    n_factors = normals_arr.shape[2]
    increments = raw_np.empty_like(normals_arr)
    for factor in range(n_factors):
        bridge = brownian_bridge(T, n_steps, n_paths, bridge_shocks=normals_arr[:, :, factor])
        increments[:, :, factor] = raw_np.diff(bridge, axis=1) * inv_sqrt_dt
    return increments


def antithetic(
    engine,
    x0,
    T: float,
    payoff_fn,
    discount_rate: float = 0.0,
    *,
    use_brownian_bridge: bool = False,
) -> dict:
    """Antithetic variates using paired factor shocks."""
    factor_dim = int(getattr(engine.process, "factor_dim", getattr(engine.process, "state_dim", 1)))
    shocks = antithetic_normals(
        engine.n_paths,
        engine.n_steps,
        n_factors=factor_dim,
        rng=engine.rng,
    )
    if use_brownian_bridge:
        shocks = brownian_bridge_increments(shocks, T)

    paths = engine.simulate_with_shocks(x0, T, shocks)
    payoffs = raw_np.asarray(payoff_fn(paths), dtype=float)

    half = engine.n_paths // 2
    avg_payoffs = 0.5 * (payoffs[:half] + payoffs[half:])
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
    """Control variate variance reduction."""
    cov_pc = raw_np.cov(payoffs, control_values)[0, 1]
    var_c = raw_np.var(control_values)
    beta = cov_pc / var_c if var_c > 0 else 0.0

    adjusted = payoffs - beta * (control_values - control_expected)

    price = float(raw_np.mean(discounted := discount_factor * adjusted))
    std_error = float(raw_np.std(discounted) / raw_np.sqrt(len(discounted)))

    return {"price": price, "std_error": std_error, "beta": beta}


def sobol_normals(
    n_paths: int,
    n_steps: int,
    n_factors: int = 1,
) -> raw_np.ndarray:
    """Generate quasi-random normal samples via a factor-space Sobol sequence."""
    from scipy.stats import norm
    from scipy.stats.qmc import Sobol

    dimension = n_steps * n_factors
    sampler = Sobol(d=dimension, scramble=True)
    uniforms = sampler.random(n_paths)
    uniforms = raw_np.clip(uniforms, 1e-10, 1 - 1e-10)
    normals = norm.ppf(uniforms)

    if n_factors == 1:
        return normals.reshape(n_paths, n_steps)
    return normals.reshape(n_paths, n_steps, n_factors)

