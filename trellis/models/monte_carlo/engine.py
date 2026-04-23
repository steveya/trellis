"""Monte Carlo simulation engine for option pricing.

Generates random price paths from a stochastic process, then uses those
paths to estimate the present value of a derivative. Supports barrier
monitoring, variance reduction, and multiple discretization schemes.
"""

from __future__ import annotations

import math

import numpy as raw_np

from trellis.core.differentiable import get_numpy
import trellis.models.monte_carlo.discretization as mc_discretization
from trellis.models.monte_carlo.discretization import (
    _build_exact_evaluator,
    _build_state_time_evaluator,
    _specialized_process_kind,
    euler_maruyama,
    exact_simulation,
)
from trellis.models.monte_carlo.path_state import (
    BarrierMonitor,
    MonteCarloPathRequirement,
    MonteCarloPathState,
    _materialize_initial_cross_section,
)

np = get_numpy()

_DISCONTINUOUS_DERIVATIVE_POLICY_VERSION = "mc_discontinuous_derivative_policy_v1"
_DISCONTINUOUS_DERIVATIVE_POLICY = "fail_closed"
_DISCONTINUOUS_DERIVATIVE_FALLBACK = "finite_difference_bump_reprice"


def _to_backend_array(values):
    if isinstance(values, raw_np.ndarray):
        return values
    if hasattr(values, "_value"):
        return values
    return raw_np.asarray(values, dtype=float)


def _state_dim(process) -> int:
    """Number of variables tracked per path (e.g. 1 for GBM, 2 for Heston)."""
    return int(getattr(process, "state_dim", 1))


def _factor_dim(process) -> int:
    """Number of independent random drivers (Brownian motions) per step."""
    return int(getattr(process, "factor_dim", _state_dim(process)))


def _coerce_initial_paths(x0, n_paths: int, state_dim: int) -> raw_np.ndarray:
    """Expand a scalar or vector starting value into an array of shape (n_paths,) or (n_paths, state_dim)."""
    if state_dim == 1:
        return raw_np.full(n_paths, float(x0), dtype=float)

    initial = raw_np.asarray(x0, dtype=float)
    if initial.ndim == 0:
        initial = raw_np.full(state_dim, float(initial), dtype=float)
    if initial.shape != (state_dim,):
        raise ValueError(f"x0 must be scalar or shape ({state_dim},) for a {state_dim}D process")
    return raw_np.broadcast_to(initial, (n_paths, state_dim)).copy()


def _allocate_paths(n_paths: int, n_steps: int, initial_values: raw_np.ndarray) -> raw_np.ndarray:
    """Allocate the output array for simulated paths, pre-filled with initial values at step 0."""
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


def _apply_diffusion(sig, dw, sqrt_dt: float) -> raw_np.ndarray:
    """Apply a diffusion loading to one set of factor shocks."""
    sig_arr = raw_np.asarray(sig, dtype=float)
    dw_arr = raw_np.asarray(dw, dtype=float)

    if sig_arr.shape == dw_arr.shape:
        return sig_arr * sqrt_dt * dw_arr

    if sig_arr.ndim == 3 and dw_arr.ndim == 2:
        return sqrt_dt * raw_np.einsum("nij,nj->ni", sig_arr, dw_arr)

    raise ValueError(
        "diffusion output must match the state shape or provide (n_paths, state_dim, factor_dim) loadings",
    )


def _apply_diffusion_differentiable(sig, dw, sqrt_dt):
    """Apply diffusion loading using autograd-compatible operations (no raw numpy)."""
    sig_arr = sig
    dw_arr = dw

    if sig_arr.shape == dw_arr.shape:
        return sig_arr * sqrt_dt * dw_arr

    if sig_arr.ndim == 3 and dw_arr.ndim == 2:
        return sqrt_dt * np.einsum("nij,nj->ni", sig_arr, dw_arr)

    raise ValueError(
        "diffusion output must match the state shape or provide (n_paths, state_dim, factor_dim) loadings",
    )


def _crossed_barrier(values: raw_np.ndarray, monitor: BarrierMonitor) -> raw_np.ndarray:
    """Return the barrier-hit indicator for one cross-section."""
    values_arr = raw_np.asarray(values)
    crossed = values_arr <= monitor.level if monitor.direction == "down" else values_arr >= monitor.level
    if crossed.ndim == 1:
        return crossed
    axes = tuple(range(1, crossed.ndim))
    return raw_np.any(crossed, axis=axes)


def _barrier_hits_from_paths(paths: raw_np.ndarray, monitor: BarrierMonitor) -> raw_np.ndarray:
    """Evaluate one monitor against a fully materialized path array."""
    if monitor.observation_steps:
        observed = paths[:, monitor.observation_steps]
    else:
        observed = paths
    return _crossed_barrier(observed, monitor)


def _replay_reducers(paths: raw_np.ndarray, requirement: MonteCarloPathRequirement) -> dict[str, raw_np.ndarray]:
    """Compute reducer outputs from a materialized path tensor."""
    if not requirement.reducers:
        return {}

    reducer_values = {
        reducer.name: reducer.init(_to_backend_array(paths[:, 0]), paths.shape[1] - 1)
        for reducer in requirement.reducers
    }
    for step in range(1, paths.shape[1]):
        values = _to_backend_array(paths[:, step])
        for reducer in requirement.reducers:
            reducer_values[reducer.name] = reducer.update(reducer_values[reducer.name], values, step)
    return reducer_values


def _metadata_tuple(values) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _payoff_derivative_metadata(payoff_fn) -> dict[str, object]:
    metadata = getattr(payoff_fn, "derivative_metadata", None)
    if not metadata:
        return {}
    return dict(metadata)


def _discontinuous_features(
    requirement: MonteCarloPathRequirement | None,
    payoff_metadata: dict[str, object] | None = None,
) -> tuple[str, ...]:
    features: list[str] = []
    if requirement is not None and requirement.barrier_monitors:
        features.append("barrier_monitor")
    features.extend(_metadata_tuple((payoff_metadata or {}).get("discontinuous_features")))
    return tuple(dict.fromkeys(features))


def _unsupported_reason(features: tuple[str, ...], payoff_metadata: dict[str, object] | None = None) -> str:
    explicit_reason = (payoff_metadata or {}).get("unsupported_reason")
    if explicit_reason:
        return str(explicit_reason)
    if "barrier_monitor" in features:
        return "barrier_monitor_discontinuity"
    if "barrier_event" in features:
        return "barrier_event_discontinuity"
    if "exercise_event" in features:
        return "exercise_event_discontinuity"
    return "discontinuous_payoff"


def describe_monte_carlo_derivative_policy(
    requirement: MonteCarloPathRequirement | None = None,
    *,
    differentiable: bool = False,
    payoff_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return derivative-policy metadata for one Monte Carlo pricing request."""
    features = _discontinuous_features(requirement, payoff_metadata)
    if features:
        return {
            "resolved_derivative_method": (
                "unsupported_discontinuous_pathwise"
                if differentiable
                else "forward_price_only"
            ),
            "pathwise_autodiff_supported": False,
            "discontinuous_features": features,
            "discontinuous_derivative_policy": _DISCONTINUOUS_DERIVATIVE_POLICY,
            "fallback_derivative_method": _DISCONTINUOUS_DERIVATIVE_FALLBACK,
            "unsupported_reason": _unsupported_reason(features, payoff_metadata),
            "policy_version": _DISCONTINUOUS_DERIVATIVE_POLICY_VERSION,
        }
    return {
        "resolved_derivative_method": "autodiff_pathwise" if differentiable else "forward_price_only",
        "pathwise_autodiff_supported": bool(differentiable),
        "discontinuous_features": (),
        "policy_version": _DISCONTINUOUS_DERIVATIVE_POLICY_VERSION,
    }


def _validate_differentiable_state_requirement(
    requirement: MonteCarloPathRequirement,
    payoff_metadata: dict[str, object] | None = None,
) -> None:
    metadata = describe_monte_carlo_derivative_policy(
        requirement,
        differentiable=True,
        payoff_metadata=payoff_metadata,
    )
    if metadata["pathwise_autodiff_supported"] is False:
        reason = metadata["unsupported_reason"]
        features = ", ".join(
            str(feature).replace("_", "-")
            for feature in metadata["discontinuous_features"]
        )
        raise NotImplementedError(
            "differentiable Monte Carlo does not support "
            f"{features} state contracts; policy={metadata['discontinuous_derivative_policy']}; "
            f"unsupported_reason={reason}; "
            f"fallback_derivative_method={metadata['fallback_derivative_method']}",
        )


def _differentiable_path_state_from_paths(
    paths,
    *,
    initial_value,
    requirement: MonteCarloPathRequirement,
    n_steps: int,
) -> MonteCarloPathState:
    terminal_values = paths[:, -1]
    snapshots = {
        step: paths[:, step]
        for step in requirement.snapshot_steps
        if 0 < step < n_steps
    }
    full_paths = paths if requirement.full_path else None
    state = MonteCarloPathState(
        initial_value=initial_value,
        n_steps=n_steps,
        terminal_values=terminal_values,
        full_paths=full_paths,
        snapshots=snapshots,
        barrier_hits={},
        reducer_values=_replay_reducers(paths, requirement),
    )
    return state


def _coerce_shock_tensor(shocks, n_paths: int, n_steps: int, factor_dim: int):
    """Normalize external shock tensors to the expected simulation shape."""
    shock_array = raw_np.asarray(shocks, dtype=float)
    if factor_dim == 1:
        if shock_array.shape != (n_paths, n_steps):
            raise ValueError(f"shocks must have shape ({n_paths}, {n_steps}) for a scalar-factor process")
        return shock_array
    if shock_array.shape != (n_paths, n_steps, factor_dim):
        raise ValueError(
            f"shocks must have shape ({n_paths}, {n_steps}, {factor_dim}) for a {factor_dim}-factor process",
        )
    return shock_array


class _ReducedPathAccumulator:
    """Mutable storage for reduced Monte Carlo state."""

    def __init__(
        self,
        requirement: MonteCarloPathRequirement,
        *,
        initial_value,
        n_paths: int,
        n_steps: int,
        terminal_shape: tuple[int, ...],
    ) -> None:
        self._requirement = requirement
        self._n_steps = n_steps
        self._snapshot_steps = set(requirement.snapshot_steps)
        self._barrier_monitors = requirement.barrier_monitors
        self._reducers = requirement.reducers

        initial = _materialize_initial_cross_section(initial_value, n_paths, float)
        if terminal_shape != initial.shape:
            raise ValueError("initial state shape does not match the simulated terminal shape")

        self._full_paths = None
        if requirement.full_path:
            if initial.ndim == 1:
                self._full_paths = raw_np.empty((n_paths, n_steps + 1), dtype=float)
            else:
                self._full_paths = raw_np.empty((n_paths, n_steps + 1, initial.shape[1]), dtype=float)
            self._full_paths[:, 0] = initial

        self._snapshots: dict[int, raw_np.ndarray] = {}
        if 0 in self._snapshot_steps:
            self._snapshots[0] = initial.copy()

        self._barrier_hits: dict[str, raw_np.ndarray] = {}
        for monitor in self._barrier_monitors:
            if monitor.observation_steps and 0 not in monitor.observation_steps:
                self._barrier_hits[monitor.name] = raw_np.zeros(n_paths, dtype=bool)
            else:
                self._barrier_hits[monitor.name] = _crossed_barrier(initial, monitor)

        self._reducer_values = {
            reducer.name: reducer.init(initial, n_steps)
            for reducer in self._reducers
        }

    def observe(self, step: int, values: raw_np.ndarray) -> None:
        """Store one simulated cross-section."""
        if self._full_paths is not None:
            self._full_paths[:, step] = values

        if 0 < step < self._n_steps and step in self._snapshot_steps:
            self._snapshots[step] = raw_np.asarray(values).copy()

        for monitor in self._barrier_monitors:
            if not monitor.observation_steps or step in monitor.observation_steps:
                self._barrier_hits[monitor.name] |= _crossed_barrier(values, monitor)

        for reducer in self._reducers:
            self._reducer_values[reducer.name] = reducer.update(
                self._reducer_values[reducer.name],
                raw_np.asarray(values),
                step,
            )

    def build(self, initial_value, terminal_values: raw_np.ndarray) -> MonteCarloPathState:
        """Freeze the accumulated reduced simulation state."""
        return MonteCarloPathState(
            initial_value=raw_np.asarray(initial_value, dtype=float).copy()
            if raw_np.asarray(initial_value).ndim > 0 else float(initial_value),
            n_steps=self._n_steps,
            terminal_values=raw_np.asarray(terminal_values, dtype=float).copy(),
            full_paths=self._full_paths,
            snapshots=self._snapshots,
            barrier_hits=self._barrier_hits,
            reducer_values=self._reducer_values,
        )


def _coerce_path_requirement(storage_policy) -> MonteCarloPathRequirement | None:
    """Normalize supported storage-policy values."""
    if storage_policy is None or storage_policy == "auto":
        return None
    if storage_policy == "full_path":
        return MonteCarloPathRequirement.full_paths()
    if storage_policy == "terminal_only":
        return MonteCarloPathRequirement.terminal_only()
    if isinstance(storage_policy, MonteCarloPathRequirement):
        return storage_policy
    raise ValueError(
        "storage_policy must be 'auto', 'full_path', 'terminal_only', or a MonteCarloPathRequirement",
    )


def _payoff_path_requirement(payoff_fn) -> MonteCarloPathRequirement | None:
    """Return an explicit path requirement declared by the payoff."""
    requirement = getattr(payoff_fn, "path_requirement", None)
    if callable(requirement):
        requirement = requirement()
    if requirement is None:
        return None
    if not isinstance(requirement, MonteCarloPathRequirement):
        raise TypeError("payoff path_requirement must be a MonteCarloPathRequirement")
    return requirement


class MonteCarloEngine:
    """Generic Monte Carlo pricing engine."""

    def __init__(
        self,
        process,
        n_paths: int = 10000,
        n_steps: int = 100,
        seed: int | None = None,
        method: str = "euler",
        scheme=None,
    ):
        """Store the process, simulation controls, RNG, and optional scheme override."""
        self.process = process
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = raw_np.random.default_rng(seed)
        self.method = method
        self.scheme = scheme

    def _simulate_with_scheme(self, x0, T: float) -> raw_np.ndarray:
        """Simulate paths using a DiscretizationScheme object."""
        if _state_dim(self.process) != 1:
            return self._simulate_vector_with_scheme(x0, T)

        scheme_name = getattr(self.scheme, "name", None)
        if scheme_name == "exact":
            return exact_simulation(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        if scheme_name == "euler":
            return euler_maruyama(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        if scheme_name == "milstein":
            return mc_discretization.milstein(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
                fd_epsilon=getattr(self.scheme, "eps", 1e-6),
            )

        dt = T / self.n_steps
        paths = raw_np.empty((self.n_paths, self.n_steps + 1))
        paths[:, 0] = x0

        for i in range(self.n_steps):
            t = i * dt
            dw = self.rng.standard_normal(self.n_paths)
            paths[:, i + 1] = self.scheme.step(
                self.process, paths[:, i], t, dt, dw,
            )

        return paths

    def _simulate_vector_with_scheme(self, x0, T: float) -> raw_np.ndarray:
        """Simulate vector-state paths using an explicit scheme."""
        scheme_name = getattr(self.scheme, "name", None)
        if scheme_name == "exact":
            return exact_simulation(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        if scheme_name == "euler":
            return euler_maruyama(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        if scheme_name == "milstein":
            raise NotImplementedError("Milstein is only supported for scalar diffusions")

        dt = T / self.n_steps
        factor_dim = _factor_dim(self.process)
        x = _coerce_initial_paths(x0, self.n_paths, _state_dim(self.process))
        paths = _allocate_paths(self.n_paths, self.n_steps, x)

        for i in range(self.n_steps):
            t = i * dt
            dw = _normal_shocks(self.rng, self.n_paths, factor_dim)
            x = raw_np.asarray(self.scheme.step(self.process, x, t, dt, dw), dtype=float)
            paths[:, i + 1] = x

        return paths

    def simulate(self, x0, T: float) -> raw_np.ndarray:
        """Generate paths."""
        if self.scheme is not None:
            return self._simulate_with_scheme(x0, T)

        if self.method == "exact":
            return exact_simulation(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        if self.method == "milstein":
            if _state_dim(self.process) != 1:
                raise NotImplementedError("Milstein is only supported for scalar diffusions")
            return mc_discretization.milstein(
                self.process, x0, T, self.n_steps, self.n_paths, self.rng,
            )
        return euler_maruyama(
            self.process, x0, T, self.n_steps, self.n_paths, self.rng,
        )

    def simulate_with_shocks(self, x0, T: float, shocks, *, differentiable: bool = False) -> raw_np.ndarray:
        """Generate paths from externally supplied standard-normal shocks.

        When ``differentiable=True``, the path tensor stays inside the autograd
        trace and the caller is responsible for supplying fixed shocks.
        """
        factor_dim = _factor_dim(self.process)
        shock_tensor = _coerce_shock_tensor(shocks, self.n_paths, self.n_steps, factor_dim)
        if differentiable:
            if _state_dim(self.process) == 1:
                return self._simulate_scalar_with_shocks_differentiable(x0, T, shock_tensor)
            return self._simulate_vector_with_shocks_differentiable(x0, T, shock_tensor)
        if _state_dim(self.process) == 1:
            return self._simulate_scalar_with_shocks(x0, T, shock_tensor)
        return self._simulate_vector_with_shocks(x0, T, shock_tensor)

    def _simulate_scalar_with_shocks(self, x0, T: float, shocks: raw_np.ndarray) -> raw_np.ndarray:
        """Generate scalar-state paths from an explicit shock matrix."""
        dt = T / self.n_steps
        sqrt_dt = math.sqrt(dt)
        x = raw_np.full(self.n_paths, x0, dtype=float)
        paths = raw_np.empty((self.n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = x

        scheme_name = getattr(self.scheme, "name", None) if self.scheme is not None else None
        method = scheme_name or self.method
        if method == "milstein" and _state_dim(self.process) != 1:
            raise NotImplementedError("Milstein is only supported for scalar diffusions")

        drift_eval, first_mu = _build_state_time_evaluator(self.process.drift, x, 0.0)
        diffusion_eval, first_sig = _build_state_time_evaluator(self.process.diffusion, x, 0.0)
        exact_eval = None

        for i in range(self.n_steps):
            t = i * dt
            dw = shocks[:, i]
            if method == "exact":
                if exact_eval is None:
                    exact_eval, _ = _build_exact_evaluator(self.process.exact_sample, x, t, dt, dw)
                x = raw_np.asarray(exact_eval(x, t, dt, dw), dtype=float)
            elif method == "milstein":
                mu = first_mu if i == 0 and first_mu is not None else drift_eval(x, t)
                sig = first_sig if i == 0 and first_sig is not None else diffusion_eval(x, t)
                sig_up = diffusion_eval(x + 1e-6, t)
                dsig_dx = (sig_up - sig) / 1e-6
                x = raw_np.asarray(
                    x
                    + mu * dt
                    + sig * sqrt_dt * dw
                    + 0.5 * sig * dsig_dx * (dw ** 2 - 1.0) * dt,
                    dtype=float,
                )
            else:
                mu = first_mu if i == 0 and first_mu is not None else drift_eval(x, t)
                sig = first_sig if i == 0 and first_sig is not None else diffusion_eval(x, t)
                x = raw_np.asarray(x + mu * dt + sig * sqrt_dt * dw, dtype=float)
            paths[:, i + 1] = x

        return paths

    def _simulate_vector_with_shocks(self, x0, T: float, shocks: raw_np.ndarray) -> raw_np.ndarray:
        """Generate vector-state paths from an explicit shock tensor."""
        dt = T / self.n_steps
        sqrt_dt = math.sqrt(dt)
        x = _coerce_initial_paths(x0, self.n_paths, _state_dim(self.process))
        paths = _allocate_paths(self.n_paths, self.n_steps, x)

        scheme_name = getattr(self.scheme, "name", None) if self.scheme is not None else None
        method = scheme_name or self.method
        if method == "milstein":
            raise NotImplementedError("Milstein is only supported for scalar diffusions")

        for i in range(self.n_steps):
            t = i * dt
            dw = shocks[:, i]
            if method == "exact":
                x = raw_np.asarray(self.process.exact_sample(x, t, dt, dw), dtype=float)
            else:
                mu = raw_np.asarray(self.process.drift(x, t), dtype=float)
                sig = raw_np.asarray(self.process.diffusion(x, t), dtype=float)
                x = raw_np.asarray(x + mu * dt + _apply_diffusion(sig, dw, sqrt_dt), dtype=float)
            paths[:, i + 1] = x

        return paths

    def _simulate_scalar_with_shocks_differentiable(self, x0, T: float, shocks) -> raw_np.ndarray:
        """Generate scalar-state paths from explicit shocks without scalarizing."""
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        x = np.ones(self.n_paths) * x0
        columns = [x]

        scheme_name = getattr(self.scheme, "name", None) if self.scheme is not None else None
        method = scheme_name or self.method
        if method not in {"exact", "euler", "milstein"}:
            raise NotImplementedError(
                "differentiable Monte Carlo only supports exact, euler, and milstein methods",
            )

        for i in range(self.n_steps):
            t = i * dt
            dw = shocks[:, i]
            if method == "exact":
                x = self.process.exact_sample(x, t, dt, dw)
            elif method == "milstein":
                eps = getattr(self.scheme, "eps", 1e-6)
                mu = self.process.drift(x, t)
                sig = self.process.diffusion(x, t)
                sig_up = self.process.diffusion(x + eps, t)
                dsig_dx = (sig_up - sig) / eps
                x = x + mu * dt + sig * sqrt_dt * dw + 0.5 * sig * dsig_dx * (dw ** 2 - 1.0) * dt
            else:
                mu = self.process.drift(x, t)
                sig = self.process.diffusion(x, t)
                x = x + mu * dt + sig * sqrt_dt * dw
            columns.append(x)

        return np.stack(columns, axis=1)

    def _simulate_vector_with_shocks_differentiable(self, x0, T: float, shocks) -> raw_np.ndarray:
        """Generate vector-state paths from explicit shocks without scalarizing."""
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        state_dim = _state_dim(self.process)
        initial = x0
        if getattr(initial, "ndim", 0) == 0:
            initial = np.ones(state_dim) * initial
        if initial.shape != (state_dim,):
            raise ValueError(f"x0 must be scalar or shape ({state_dim},) for a {state_dim}D process")
        x = np.ones((self.n_paths, 1)) * initial
        columns = [x]

        scheme_name = getattr(self.scheme, "name", None) if self.scheme is not None else None
        method = scheme_name or self.method
        if method not in {"exact", "euler"}:
            raise NotImplementedError(
                "differentiable vector Monte Carlo only supports exact and euler methods",
            )

        for i in range(self.n_steps):
            t = i * dt
            dw = shocks[:, i]
            if method == "exact":
                x = self.process.exact_sample(x, t, dt, dw)
            else:
                mu = self.process.drift(x, t)
                sig = self.process.diffusion(x, t)
                x = x + mu * dt + _apply_diffusion_differentiable(sig, dw, sqrt_dt)
            columns.append(x)

        return np.stack(columns, axis=1)

    def _simulate_state_with_custom_scheme(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Stream a custom scheme into a reduced path-state accumulator."""
        if _state_dim(self.process) != 1:
            return self._simulate_state_vector_with_custom_scheme(x0, T, requirement)

        dt = T / self.n_steps
        x = raw_np.full(self.n_paths, x0, dtype=float)
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=x0,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )

        for i in range(self.n_steps):
            t = i * dt
            dw = self.rng.standard_normal(self.n_paths)
            x = raw_np.asarray(self.scheme.step(self.process, x, t, dt, dw), dtype=float)
            accumulator.observe(i + 1, x)

        return accumulator.build(x0, x)

    def _simulate_state_vector_with_custom_scheme(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Stream a custom vector-state scheme into reduced storage."""
        dt = T / self.n_steps
        factor_dim = _factor_dim(self.process)
        x = _coerce_initial_paths(x0, self.n_paths, _state_dim(self.process))
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=raw_np.asarray(x0, dtype=float),
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )

        for i in range(self.n_steps):
            t = i * dt
            dw = _normal_shocks(self.rng, self.n_paths, factor_dim)
            x = raw_np.asarray(self.scheme.step(self.process, x, t, dt, dw), dtype=float)
            accumulator.observe(i + 1, x)

        return accumulator.build(raw_np.asarray(x0, dtype=float), x)

    def _simulate_state_exact(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Stream exact transitions into reduced storage."""
        if _state_dim(self.process) != 1:
            return self._simulate_state_exact_vector(x0, T, requirement)

        dt = T / self.n_steps
        x = raw_np.full(self.n_paths, x0, dtype=float)
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=x0,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )
        kind = _specialized_process_kind(self.process)

        if kind == "gbm":
            drift_term = (self.process.mu - 0.5 * self.process.sigma ** 2) * dt
            diffusion_term = self.process.sigma * math.sqrt(dt)
            for i in range(self.n_steps):
                dw = self.rng.standard_normal(self.n_paths)
                x = x * raw_np.exp(drift_term + diffusion_term * dw)
                accumulator.observe(i + 1, x)
            return accumulator.build(x0, x)

        if kind == "vasicek":
            mean_multiplier = math.exp(-self.process.a * dt)
            mean_shift = self.process.b * (1.0 - mean_multiplier)
            std = math.sqrt(self.process.exact_variance(x0, 0.0, dt))
            for i in range(self.n_steps):
                dw = self.rng.standard_normal(self.n_paths)
                x = x * mean_multiplier + mean_shift + std * dw
                accumulator.observe(i + 1, x)
            return accumulator.build(x0, x)

        sample_dw = self.rng.standard_normal(self.n_paths)
        exact_eval, first_values = _build_exact_evaluator(
            self.process.exact_sample,
            x,
            0.0,
            dt,
            sample_dw,
        )
        if first_values is not None:
            x = raw_np.asarray(first_values, dtype=float)
        else:
            x = raw_np.asarray(exact_eval(x, 0.0, dt, sample_dw), dtype=float)
        accumulator.observe(1, x)

        for i in range(1, self.n_steps):
            t = i * dt
            dw = self.rng.standard_normal(self.n_paths)
            x = raw_np.asarray(exact_eval(x, t, dt, dw), dtype=float)
            accumulator.observe(i + 1, x)

        return accumulator.build(x0, x)

    def _simulate_state_exact_vector(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Stream exact vector-state transitions into reduced storage."""
        dt = T / self.n_steps
        factor_dim = _factor_dim(self.process)
        x = _coerce_initial_paths(x0, self.n_paths, _state_dim(self.process))
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=raw_np.asarray(x0, dtype=float),
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )

        for i in range(self.n_steps):
            t = i * dt
            dw = _normal_shocks(self.rng, self.n_paths, factor_dim)
            x = raw_np.asarray(self.process.exact_sample(x, t, dt, dw), dtype=float)
            accumulator.observe(i + 1, x)

        return accumulator.build(raw_np.asarray(x0, dtype=float), x)

    def _simulate_state_euler(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Stream Euler-Maruyama steps into reduced storage."""
        if _state_dim(self.process) != 1:
            return self._simulate_state_euler_vector(x0, T, requirement)

        dt = T / self.n_steps
        sqrt_dt = raw_np.sqrt(dt)
        x = raw_np.full(self.n_paths, x0, dtype=float)
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=x0,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )

        drift_eval, first_mu = _build_state_time_evaluator(self.process.drift, x, 0.0)
        diffusion_eval, first_sig = _build_state_time_evaluator(self.process.diffusion, x, 0.0)

        for i in range(self.n_steps):
            t = i * dt
            dw = self.rng.standard_normal(self.n_paths)
            mu = first_mu if i == 0 and first_mu is not None else drift_eval(x, t)
            sig = first_sig if i == 0 and first_sig is not None else diffusion_eval(x, t)
            x = raw_np.asarray(x + mu * dt + sig * sqrt_dt * dw, dtype=float)
            accumulator.observe(i + 1, x)

        return accumulator.build(x0, x)

    def _simulate_state_euler_vector(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Stream Euler steps for a vector-state process into reduced storage."""
        dt = T / self.n_steps
        sqrt_dt = math.sqrt(dt)
        factor_dim = _factor_dim(self.process)
        x = _coerce_initial_paths(x0, self.n_paths, _state_dim(self.process))
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=raw_np.asarray(x0, dtype=float),
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )

        for i in range(self.n_steps):
            t = i * dt
            dw = _normal_shocks(self.rng, self.n_paths, factor_dim)
            mu = raw_np.asarray(self.process.drift(x, t), dtype=float)
            sig = raw_np.asarray(self.process.diffusion(x, t), dtype=float)
            x = raw_np.asarray(x + mu * dt + _apply_diffusion(sig, dw, sqrt_dt), dtype=float)
            accumulator.observe(i + 1, x)

        return accumulator.build(raw_np.asarray(x0, dtype=float), x)

    def _simulate_state_milstein(
        self,
        x0,
        T: float,
        requirement: MonteCarloPathRequirement,
        *,
        fd_epsilon: float = 1e-6,
    ) -> MonteCarloPathState:
        """Stream Milstein steps into reduced storage."""
        if _state_dim(self.process) != 1:
            raise NotImplementedError("Milstein is only supported for scalar diffusions")

        dt = T / self.n_steps
        sqrt_dt = raw_np.sqrt(dt)
        x = raw_np.full(self.n_paths, x0, dtype=float)
        accumulator = _ReducedPathAccumulator(
            requirement,
            initial_value=x0,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            terminal_shape=x.shape,
        )

        kind = _specialized_process_kind(self.process)
        if kind == "gbm" and fd_epsilon == 1e-6:
            sigma2 = self.process.sigma ** 2
            correction_scale = 0.5 * sigma2 * dt
            for i in range(self.n_steps):
                dw = self.rng.standard_normal(self.n_paths)
                x = raw_np.asarray(
                    x
                    + self.process.mu * x * dt
                    + self.process.sigma * x * sqrt_dt * dw
                    + correction_scale * x * (dw ** 2 - 1.0),
                    dtype=float,
                )
                accumulator.observe(i + 1, x)
            return accumulator.build(x0, x)

        drift_eval, first_mu = _build_state_time_evaluator(self.process.drift, x, 0.0)
        diffusion_eval, first_sig = _build_state_time_evaluator(self.process.diffusion, x, 0.0)

        for i in range(self.n_steps):
            t = i * dt
            dw = self.rng.standard_normal(self.n_paths)
            mu = first_mu if i == 0 and first_mu is not None else drift_eval(x, t)
            sig = first_sig if i == 0 and first_sig is not None else diffusion_eval(x, t)
            sig_up = diffusion_eval(x + fd_epsilon, t)
            dsig_dx = (sig_up - sig) / fd_epsilon
            x = raw_np.asarray(
                x
                + mu * dt
                + sig * sqrt_dt * dw
                + 0.5 * sig * dsig_dx * (dw ** 2 - 1.0) * dt,
                dtype=float,
            )
            accumulator.observe(i + 1, x)

        return accumulator.build(x0, x)

    def simulate_state(
        self,
        x0,
        T: float,
        path_requirement: MonteCarloPathRequirement,
    ) -> MonteCarloPathState:
        """Generate reduced path state according to an explicit storage contract."""
        requirement = _coerce_path_requirement(path_requirement)
        if requirement is None:
            requirement = MonteCarloPathRequirement.full_paths()

        if requirement.full_path:
            paths = self.simulate(x0, T)
            barrier_hits = {
                monitor.name: _barrier_hits_from_paths(paths, monitor)
                for monitor in requirement.barrier_monitors
            }
            snapshots = {
                step: raw_np.asarray(paths[:, step]).copy()
                for step in requirement.snapshot_steps
                if 0 < step < self.n_steps
            }
            return MonteCarloPathState(
                initial_value=raw_np.asarray(x0, dtype=float).copy()
                if raw_np.asarray(x0).ndim > 0 else float(x0),
                n_steps=self.n_steps,
                terminal_values=raw_np.asarray(paths[:, -1]).copy(),
                full_paths=paths,
                snapshots=snapshots,
                barrier_hits=barrier_hits,
                reducer_values=_replay_reducers(paths, requirement),
            )

        if self.scheme is not None:
            scheme_name = getattr(self.scheme, "name", None)
            if scheme_name == "exact":
                return self._simulate_state_exact(x0, T, requirement)
            if scheme_name == "euler":
                return self._simulate_state_euler(x0, T, requirement)
            if scheme_name == "milstein":
                return self._simulate_state_milstein(
                    x0,
                    T,
                    requirement,
                    fd_epsilon=getattr(self.scheme, "eps", 1e-6),
                )
            return self._simulate_state_with_custom_scheme(x0, T, requirement)

        if self.method == "exact":
            return self._simulate_state_exact(x0, T, requirement)
        if self.method == "milstein":
            return self._simulate_state_milstein(x0, T, requirement)
        return self._simulate_state_euler(x0, T, requirement)

    def price(
        self,
        x0,
        T: float,
        payoff_fn,
        discount_rate: float = 0.0,
        *,
        storage_policy: str | MonteCarloPathRequirement = "auto",
        return_paths: bool = True,
        shocks=None,
        differentiable: bool = False,
    ) -> dict:
        """Price a derivative via Monte Carlo.

        ``shocks=...`` forces a deterministic pathwise evaluation path.  When
        ``differentiable=True``, both the simulated paths and the sample mean
        remain autograd-traceable.
        """
        path_state = None
        paths = None

        explicit_requirement = _coerce_path_requirement(storage_policy)
        declared_requirement = _payoff_path_requirement(payoff_fn)
        state_evaluator = getattr(payoff_fn, "evaluate_state", None)
        payoff_metadata = _payoff_derivative_metadata(payoff_fn)

        if shocks is not None:
            if (
                differentiable
                and callable(state_evaluator)
                and (explicit_requirement is not None or declared_requirement is not None)
            ):
                requirement = explicit_requirement or declared_requirement
                _validate_differentiable_state_requirement(requirement, payoff_metadata)
                simulated_paths = self.simulate_with_shocks(x0, T, shocks, differentiable=True)
                path_state = _differentiable_path_state_from_paths(
                    simulated_paths,
                    initial_value=x0,
                    requirement=requirement,
                    n_steps=self.n_steps,
                )
                payoffs = state_evaluator(path_state)
                paths = simulated_paths if return_paths else None
            else:
                paths = self.simulate_with_shocks(x0, T, shocks, differentiable=differentiable)
                payoffs = payoff_fn(paths)
        elif (
            not return_paths
            and callable(state_evaluator)
            and (explicit_requirement is not None or declared_requirement is not None)
        ):
            requirement = explicit_requirement or declared_requirement
            path_state = self.simulate_state(x0, T, requirement)
            payoffs = raw_np.asarray(state_evaluator(path_state), dtype=float)
        else:
            if differentiable:
                raise ValueError("differentiable=True requires explicit shocks for deterministic pathwise gradients")
            paths = self.simulate(x0, T)
            payoffs = raw_np.asarray(payoff_fn(paths), dtype=float)

        if differentiable:
            df = np.exp(-discount_rate * T)
            discounted = df * payoffs
            price = np.mean(discounted)
            std_error = np.std(discounted) / np.sqrt(len(discounted))
        else:
            payoffs = raw_np.asarray(payoffs, dtype=float)
            df = raw_np.exp(-discount_rate * T)
            discounted = df * payoffs
            price = float(raw_np.mean(discounted))
            std_error = float(raw_np.std(discounted) / raw_np.sqrt(len(discounted)))

        return {
            "price": price,
            "std_error": std_error,
            "st_err": std_error,
            "n_paths": self.n_paths,
            "paths": paths if return_paths else None,
            "path_state": path_state,
            "derivative_metadata": describe_monte_carlo_derivative_policy(
                explicit_requirement or declared_requirement,
                differentiable=differentiable,
                payoff_metadata=payoff_metadata,
            ),
        }
