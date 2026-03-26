"""SDE discretization schemes."""

from __future__ import annotations

import math

import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit


@maybe_njit(cache=False)
def _gbm_euler_step_numba(
    x: raw_np.ndarray, mu: float, sigma: float, dt: float, sqrt_dt: float, dw: raw_np.ndarray,
) -> raw_np.ndarray:
    """Advance one Euler GBM step with a compiled loop."""
    out = raw_np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[i] + mu * x[i] * dt + sigma * x[i] * sqrt_dt * dw[i]
    return out


@maybe_njit(cache=False)
def _gbm_milstein_step_numba(
    x: raw_np.ndarray, mu: float, sigma: float, dt: float, sqrt_dt: float, dw: raw_np.ndarray,
) -> raw_np.ndarray:
    """Advance one Milstein GBM step with a compiled loop."""
    out = raw_np.empty_like(x)
    sigma2 = sigma * sigma
    correction_scale = 0.5 * sigma2 * dt
    for i in range(len(x)):
        xi = x[i]
        zi = dw[i]
        out[i] = (
            xi
            + mu * xi * dt
            + sigma * xi * sqrt_dt * zi
            + correction_scale * xi * (zi * zi - 1.0)
        )
    return out


def _state_dim(process) -> int:
    """Return the process state dimension."""
    return int(getattr(process, "state_dim", 1))


def _factor_dim(process) -> int:
    """Return the number of independent Brownian factors."""
    return int(getattr(process, "factor_dim", _state_dim(process)))


def _coerce_initial_paths(x0, n_paths: int, state_dim: int) -> raw_np.ndarray:
    """Broadcast the initial state across the path set."""
    if state_dim == 1:
        return raw_np.full(n_paths, float(x0), dtype=float)

    initial = raw_np.asarray(x0, dtype=float)
    if initial.ndim == 0:
        initial = raw_np.full(state_dim, float(initial), dtype=float)
    if initial.shape != (state_dim,):
        raise ValueError(f"x0 must be scalar or shape ({state_dim},) for a {state_dim}D process")
    return raw_np.broadcast_to(initial, (n_paths, state_dim)).copy()


def _allocate_paths(n_paths: int, n_steps: int, initial_values: raw_np.ndarray) -> raw_np.ndarray:
    """Allocate a path tensor matching the state dimensionality."""
    if initial_values.ndim == 1:
        paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
    else:
        paths = raw_np.empty((n_paths, n_steps + 1, initial_values.shape[1]), dtype=float)
    paths[:, 0] = initial_values
    return paths


def _normal_shocks(rng, n_paths: int, factor_dim: int):
    """Return one set of standard-normal shocks."""
    if factor_dim == 1:
        return rng.standard_normal(n_paths)
    return rng.standard_normal((n_paths, factor_dim))


def _broadcast_scalar(value, shape, dtype):
    """Broadcast a scalar value into a full array."""
    return raw_np.full(shape, value, dtype=dtype)


def _apply_diffusion(sig, dw, sqrt_dt: float) -> raw_np.ndarray:
    """Apply a diffusion loading to a batch of factor shocks."""
    sig_arr = raw_np.asarray(sig, dtype=float)
    dw_arr = raw_np.asarray(dw, dtype=float)

    if sig_arr.shape == dw_arr.shape:
        return sig_arr * sqrt_dt * dw_arr

    if sig_arr.ndim == 3 and dw_arr.ndim == 2:
        return sqrt_dt * raw_np.einsum("nij,nj->ni", sig_arr, dw_arr)

    raise ValueError(
        "diffusion output must match the state shape or provide (n_paths, state_dim, factor_dim) loadings",
    )


def _pointwise_from_iter(method, x: raw_np.ndarray, t: float) -> raw_np.ndarray:
    """Evaluate a non-vectorized drift/diffusion function over a batch of states."""
    if x.ndim == 1:
        return raw_np.fromiter(
            (method(float(xi), t) for xi in x),
            dtype=raw_np.result_type(x, raw_np.float64),
            count=len(x),
        )
    return raw_np.stack([raw_np.asarray(method(xi, t), dtype=float) for xi in x], axis=0)


def _exact_from_iter(method, x: raw_np.ndarray, t: float, dt: float, dw: raw_np.ndarray) -> raw_np.ndarray:
    """Evaluate a non-vectorized exact sampler over state/increment batches."""
    if x.ndim == 1:
        return raw_np.fromiter(
            (method(float(xi), t, dt, float(dwi)) for xi, dwi in zip(x, dw)),
            dtype=raw_np.result_type(x, dw, raw_np.float64),
            count=len(x),
        )
    return raw_np.stack(
        [raw_np.asarray(method(xi, t, dt, dwi), dtype=float) for xi, dwi in zip(x, dw)],
        axis=0,
    )


def _is_vectorized_state_output(sample: raw_np.ndarray, x_sample: raw_np.ndarray) -> bool:
    """Return whether a sample output looks vectorized over the path axis."""
    if sample.ndim == 0:
        return False
    if sample.shape == x_sample.shape:
        return True
    return sample.ndim > x_sample.ndim and sample.shape[: x_sample.ndim] == x_sample.shape


def _build_state_time_evaluator(method, x_sample: raw_np.ndarray, t0: float):
    """Return a callable that evaluates ``method(x, t)`` efficiently over vectors."""
    try:
        sample = raw_np.asarray(method(x_sample, t0))
    except Exception:
        sample = None

    if sample is not None:
        if _is_vectorized_state_output(sample, x_sample):
            def evaluate(x, t):
                values = raw_np.asarray(method(x, t))
                if _is_vectorized_state_output(values, raw_np.asarray(x)):
                    return values
                if values.ndim == 0:
                    return _broadcast_scalar(values.item(), raw_np.asarray(x).shape, raw_np.result_type(values, x))
                return _pointwise_from_iter(method, raw_np.asarray(x), t)

            return evaluate, sample

        if sample.ndim == 0:
            first_values = _broadcast_scalar(
                sample.item(), x_sample.shape, raw_np.result_type(sample, x_sample),
            )

            def evaluate(x, t):
                values = raw_np.asarray(method(x, t))
                if values.ndim == 0:
                    return _broadcast_scalar(values.item(), raw_np.asarray(x).shape, raw_np.result_type(values, x))
                if _is_vectorized_state_output(values, raw_np.asarray(x)):
                    return values
                return _pointwise_from_iter(method, raw_np.asarray(x), t)

            return evaluate, first_values

    def evaluate(x, t):
        return _pointwise_from_iter(method, raw_np.asarray(x), t)

    return evaluate, None


def _build_exact_evaluator(method, x_sample: raw_np.ndarray, t0: float, dt: float, dw_sample: raw_np.ndarray):
    """Return a callable that evaluates ``exact_sample(x, t, dt, dw)`` efficiently."""
    try:
        sample = raw_np.asarray(method(x_sample, t0, dt, dw_sample))
    except Exception:
        sample = None

    if sample is not None:
        if _is_vectorized_state_output(sample, x_sample):
            def evaluate(x, t, local_dt, dw):
                values = raw_np.asarray(method(x, t, local_dt, dw))
                if _is_vectorized_state_output(values, raw_np.asarray(x)):
                    return values
                if values.ndim == 0:
                    return _broadcast_scalar(values.item(), raw_np.asarray(x).shape, raw_np.result_type(values, x))
                return _exact_from_iter(method, raw_np.asarray(x), t, local_dt, raw_np.asarray(dw))

            return evaluate, sample

        if sample.ndim == 0:
            first_values = _broadcast_scalar(
                sample.item(), x_sample.shape, raw_np.result_type(sample, x_sample),
            )

            def evaluate(x, t, local_dt, dw):
                values = raw_np.asarray(method(x, t, local_dt, dw))
                if values.ndim == 0:
                    return _broadcast_scalar(values.item(), raw_np.asarray(x).shape, raw_np.result_type(values, x))
                if _is_vectorized_state_output(values, raw_np.asarray(x)):
                    return values
                return _exact_from_iter(method, raw_np.asarray(x), t, local_dt, raw_np.asarray(dw))

            return evaluate, first_values

    def evaluate(x, t, local_dt, dw):
        return _exact_from_iter(method, raw_np.asarray(x), t, local_dt, raw_np.asarray(dw))

    return evaluate, None


def _specialized_process_kind(process) -> str | None:
    """Return a stable built-in process tag eligible for accelerated kernels."""
    module_name = process.__class__.__module__
    class_name = process.__class__.__name__

    if module_name == "trellis.models.processes.gbm" and class_name == "GBM":
        return "gbm"
    if module_name == "trellis.models.processes.vasicek" and class_name == "Vasicek":
        return "vasicek"
    if module_name == "trellis.models.processes.correlated_gbm" and class_name == "CorrelatedGBM":
        return "correlated_gbm"
    return None


def _simulate_gbm_exact(process, x0, T, n_steps, n_paths, rng):
    """Exact GBM simulation using the closed-form vectorized transition."""
    dt = T / n_steps
    drift_term = (process.mu - 0.5 * process.sigma ** 2) * dt
    diffusion_term = process.sigma * math.sqrt(dt)

    paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0

    for i in range(n_steps):
        dw = rng.standard_normal(n_paths)
        x = paths[:, i]
        paths[:, i + 1] = x * raw_np.exp(drift_term + diffusion_term * dw)

    return paths


def _simulate_gbm_euler(process, x0, T, n_steps, n_paths, rng):
    """Euler GBM simulation with an optional Numba kernel."""
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0

    for i in range(n_steps):
        dw = rng.standard_normal(n_paths)
        x = paths[:, i]
        if NUMBA_AVAILABLE:
            paths[:, i + 1] = _gbm_euler_step_numba(x, process.mu, process.sigma, dt, sqrt_dt, dw)
        else:
            paths[:, i + 1] = x + process.mu * x * dt + process.sigma * x * sqrt_dt * dw

    return paths


def _simulate_gbm_milstein(process, x0, T, n_steps, n_paths, rng):
    """Milstein GBM simulation with an optional Numba kernel."""
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0

    for i in range(n_steps):
        dw = rng.standard_normal(n_paths)
        x = paths[:, i]
        if NUMBA_AVAILABLE:
            paths[:, i + 1] = _gbm_milstein_step_numba(
                x, process.mu, process.sigma, dt, sqrt_dt, dw,
            )
        else:
            sigma2 = process.sigma ** 2
            paths[:, i + 1] = (
                x
                + process.mu * x * dt
                + process.sigma * x * sqrt_dt * dw
                + 0.5 * sigma2 * x * (dw ** 2 - 1.0) * dt
            )

    return paths


def _simulate_vasicek_exact(process, x0, T, n_steps, n_paths, rng):
    """Exact Vasicek simulation using the closed-form vectorized transition."""
    dt = T / n_steps
    mean_multiplier = math.exp(-process.a * dt)
    mean_shift = process.b * (1.0 - mean_multiplier)
    std = math.sqrt(process.exact_variance(x0, 0.0, dt))

    paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0

    for i in range(n_steps):
        dw = rng.standard_normal(n_paths)
        x = paths[:, i]
        paths[:, i + 1] = x * mean_multiplier + mean_shift + std * dw

    return paths


def _simulate_correlated_gbm_exact(process, x0, T, n_steps, n_paths, rng):
    """Exact correlated GBM simulation across all assets at once."""
    dt = T / n_steps
    x = _coerce_initial_paths(x0, n_paths, process.state_dim)
    paths = _allocate_paths(n_paths, n_steps, x)

    for i in range(n_steps):
        dw = rng.standard_normal((n_paths, process.factor_dim))
        paths[:, i + 1] = process.exact_sample(paths[:, i], i * dt, dt, dw)

    return paths


def _maybe_specialized_simulation(process, method: str, x0, T, n_steps, n_paths, rng):
    """Return a specialized simulation path when a built-in process matches."""
    kind = _specialized_process_kind(process)
    if kind == "gbm":
        if method == "exact":
            return _simulate_gbm_exact(process, x0, T, n_steps, n_paths, rng)
        if method == "euler":
            return _simulate_gbm_euler(process, x0, T, n_steps, n_paths, rng)
        if method == "milstein":
            return _simulate_gbm_milstein(process, x0, T, n_steps, n_paths, rng)

    if kind == "vasicek" and method == "exact":
        return _simulate_vasicek_exact(process, x0, T, n_steps, n_paths, rng)

    if kind == "correlated_gbm" and method == "exact":
        return _simulate_correlated_gbm_exact(process, x0, T, n_steps, n_paths, rng)

    return None


def _ensure_rng(rng):
    """Create a default RNG when one is not provided."""
    if rng is None:
        return raw_np.random.default_rng()
    return rng


def euler_maruyama(
    process,
    x0,
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
) -> raw_np.ndarray:
    """Euler-Maruyama discretization of an SDE."""
    rng = _ensure_rng(rng)

    specialized = _maybe_specialized_simulation(process, "euler", x0, T, n_steps, n_paths, rng)
    if specialized is not None:
        return specialized

    state_dim = _state_dim(process)
    factor_dim = _factor_dim(process)
    dt = T / n_steps
    sqrt_dt = raw_np.sqrt(dt)
    x = _coerce_initial_paths(x0, n_paths, state_dim)
    paths = _allocate_paths(n_paths, n_steps, x)

    sample_x = paths[:, 0]
    drift_eval, first_mu = _build_state_time_evaluator(process.drift, sample_x, 0.0)
    diffusion_eval, first_sig = _build_state_time_evaluator(process.diffusion, sample_x, 0.0)

    for i in range(n_steps):
        t = i * dt
        x = paths[:, i]
        dw = _normal_shocks(rng, n_paths, factor_dim)
        mu = first_mu if i == 0 and first_mu is not None else drift_eval(x, t)
        sig = first_sig if i == 0 and first_sig is not None else diffusion_eval(x, t)
        if state_dim == 1:
            paths[:, i + 1] = x + mu * dt + sig * sqrt_dt * dw
        else:
            paths[:, i + 1] = x + mu * dt + _apply_diffusion(sig, dw, float(sqrt_dt))

    return paths


def milstein(
    process,
    x0,
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
    *,
    fd_epsilon: float = 1e-6,
) -> raw_np.ndarray:
    """Milstein scheme (higher-order for scalar diffusion)."""
    rng = _ensure_rng(rng)

    if _state_dim(process) != 1:
        raise NotImplementedError("Milstein is only supported for scalar diffusions")

    specialized = _maybe_specialized_simulation(process, "milstein", x0, T, n_steps, n_paths, rng)
    if specialized is not None and fd_epsilon == 1e-6:
        return specialized

    dt = T / n_steps
    sqrt_dt = raw_np.sqrt(dt)
    paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0

    drift_eval, first_mu = _build_state_time_evaluator(process.drift, paths[:, 0], 0.0)
    diffusion_eval, first_sig = _build_state_time_evaluator(process.diffusion, paths[:, 0], 0.0)

    for i in range(n_steps):
        t = i * dt
        x = paths[:, i]
        dw = rng.standard_normal(n_paths)

        if i == 0 and first_mu is not None:
            mu = first_mu
        else:
            mu = drift_eval(x, t)

        if i == 0 and first_sig is not None:
            sig = first_sig
        else:
            sig = diffusion_eval(x, t)

        sig_up = diffusion_eval(x + fd_epsilon, t)
        dsig_dx = (sig_up - sig) / fd_epsilon

        paths[:, i + 1] = (
            x + mu * dt + sig * sqrt_dt * dw
            + 0.5 * sig * dsig_dx * (dw ** 2 - 1.0) * dt
        )

    return paths


def exact_simulation(
    process,
    x0,
    T: float,
    n_steps: int,
    n_paths: int,
    rng=None,
) -> raw_np.ndarray:
    """Exact simulation using process.exact_sample if available."""
    rng = _ensure_rng(rng)

    specialized = _maybe_specialized_simulation(process, "exact", x0, T, n_steps, n_paths, rng)
    if specialized is not None:
        return specialized

    state_dim = _state_dim(process)
    factor_dim = _factor_dim(process)
    dt = T / n_steps
    x = _coerce_initial_paths(x0, n_paths, state_dim)
    paths = _allocate_paths(n_paths, n_steps, x)

    sample_dw = _normal_shocks(rng, n_paths, factor_dim)
    sample_x = paths[:, 0]
    exact_eval, first_values = _build_exact_evaluator(process.exact_sample, sample_x, 0.0, dt, sample_dw)
    if first_values is not None:
        paths[:, 1] = first_values
    else:
        paths[:, 1] = exact_eval(sample_x, 0.0, dt, sample_dw)

    for i in range(1, n_steps):
        t = i * dt
        dw = _normal_shocks(rng, n_paths, factor_dim)
        x = paths[:, i]
        paths[:, i + 1] = exact_eval(x, t, dt, dw)

    return paths

