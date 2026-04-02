"""Two-factor product lattices for low-dimensional hybrid pricing."""

from __future__ import annotations

from math import exp, sqrt

import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit


class ProductRecombiningLattice2D:
    """Recombining product lattice with two one-factor binomial axes."""

    def __init__(self, n_steps: int, dt: float):
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.branching = 4
        self.state_dim = 2
        max_nodes = (self.n_steps + 1) ** 2
        self._states = raw_np.full((self.n_steps + 1, max_nodes, 2), raw_np.nan)
        self._probs = raw_np.zeros((self.n_steps, max_nodes, 4), dtype=float)
        self._discounts = raw_np.ones((self.n_steps + 1, max_nodes), dtype=float)

    def n_nodes(self, step: int) -> int:
        width = int(step) + 1
        return width * width

    def index(self, step: int, i: int, j: int) -> int:
        width = int(step) + 1
        return int(i) * width + int(j)

    def coordinates(self, step: int, node: int) -> tuple[int, int]:
        width = int(step) + 1
        return divmod(int(node), width)

    def set_state(self, step: int, node: int, state) -> None:
        self._states[step, node, 0] = float(state[0])
        self._states[step, node, 1] = float(state[1])

    def get_state(self, step: int, node: int) -> tuple[float, float]:
        return (
            float(self._states[step, node, 0]),
            float(self._states[step, node, 1]),
        )

    def set_probabilities(self, step: int, node: int, probs: list[float]) -> None:
        for branch, probability in enumerate(probs):
            self._probs[step, node, branch] = probability

    def get_probabilities(self, step: int, node: int) -> list[float]:
        return [float(self._probs[step, node, branch]) for branch in range(4)]

    def set_discount(self, step: int, node: int, df: float) -> None:
        self._discounts[step, node] = float(df)

    def get_discount(self, step: int, node: int) -> float:
        return float(self._discounts[step, node])

    def child_indices(self, step: int, node: int) -> list[int]:
        i, j = self.coordinates(step, node)
        next_step = int(step) + 1
        return [
            self.index(next_step, i, j),
            self.index(next_step, i, j + 1),
            self.index(next_step, i + 1, j),
            self.index(next_step, i + 1, j + 1),
        ]

    def fast_terminal_rollback(self, values: raw_np.ndarray) -> float:
        """Return the root value for a terminal-only linear claim."""
        current = raw_np.asarray(values, dtype=float)
        for step in range(self.n_steps - 1, -1, -1):
            width = step + 1
            n_nodes = width * width
            discounts = self._discounts[step, :n_nodes]
            probs = self._probs[step, :n_nodes, :]
            if NUMBA_AVAILABLE:
                current = _product_binomial_2d_continuation_numba(current, discounts, probs, width)
            else:
                current = _product_binomial_2d_continuation_numpy(current, discounts, probs, width)
        return float(current[0])


@maybe_njit(cache=False)
def _product_binomial_2d_continuation_numba(
    values: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
    width: int,
) -> raw_np.ndarray:
    next_width = width + 1
    out = raw_np.empty(width * width, dtype=values.dtype)
    for i in range(width):
        row = i * width
        next_row = i * next_width
        next_row_up = (i + 1) * next_width
        for j in range(width):
            idx = row + j
            out[idx] = discounts[idx] * (
                probs[idx, 0] * values[next_row + j]
                + probs[idx, 1] * values[next_row + j + 1]
                + probs[idx, 2] * values[next_row_up + j]
                + probs[idx, 3] * values[next_row_up + j + 1]
            )
    return out


def _product_binomial_2d_continuation_numpy(
    values: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
    width: int,
) -> raw_np.ndarray:
    next_width = width + 1
    out = raw_np.empty(width * width, dtype=values.dtype)
    for i in range(width):
        row = i * width
        next_row = i * next_width
        next_row_up = (i + 1) * next_width
        for j in range(width):
            idx = row + j
            out[idx] = discounts[idx] * (
                probs[idx, 0] * values[next_row + j]
                + probs[idx, 1] * values[next_row + j + 1]
                + probs[idx, 2] * values[next_row_up + j]
                + probs[idx, 3] * values[next_row_up + j + 1]
            )
    return out


def build_product_spot_lattice_2d(
    *,
    spots: tuple[float, float],
    rate: float,
    sigmas: tuple[float, float],
    maturity: float,
    n_steps: int,
    correlation: float = 0.0,
) -> tuple[ProductRecombiningLattice2D, dict[str, float]]:
    """Build a two-factor recombining CRR-style product lattice."""
    if len(spots) != 2 or len(sigmas) != 2:
        raise ValueError("Two-factor spot lattices require two spots and two sigmas")

    dt = float(maturity) / max(int(n_steps), 1)
    lattice = ProductRecombiningLattice2D(int(n_steps), dt)
    sigma_1, sigma_2 = float(sigmas[0]), float(sigmas[1])
    u_1 = exp(sigma_1 * sqrt(dt))
    d_1 = 1.0 / max(u_1, 1e-12)
    u_2 = exp(sigma_2 * sqrt(dt))
    d_2 = 1.0 / max(u_2, 1e-12)
    growth = exp(float(rate) * dt)
    p_1 = (growth - d_1) / (u_1 - d_1)
    p_2 = (growth - d_2) / (u_2 - d_2)

    positivity_violations = 0
    correlation_clips = 0
    min_probability = 1.0
    max_probability = 0.0

    for step in range(int(n_steps) + 1):
        width = step + 1
        for i in range(width):
            s1 = float(spots[0]) * (u_1 ** i) * (d_1 ** (step - i))
            for j in range(width):
                s2 = float(spots[1]) * (u_2 ** j) * (d_2 ** (step - j))
                node = lattice.index(step, i, j)
                lattice.set_state(step, node, (s1, s2))
                if step < int(n_steps):
                    lattice.set_discount(step, node, exp(-float(rate) * dt))

    lower = max(0.0, p_1 + p_2 - 1.0)
    upper = min(p_1, p_2)
    covariance_term = float(correlation) * sqrt(max(p_1 * (1.0 - p_1) * p_2 * (1.0 - p_2), 0.0))
    desired_joint_up = p_1 * p_2 + covariance_term
    joint_up = min(max(desired_joint_up, lower), upper)
    if abs(joint_up - desired_joint_up) > 1e-12:
        correlation_clips += 1
    q_uu = joint_up
    q_ud = p_1 - joint_up
    q_du = p_2 - joint_up
    q_dd = 1.0 - p_1 - p_2 + joint_up
    base_probs = [q_dd, q_du, q_ud, q_uu]

    for step in range(int(n_steps)):
        for node in range(lattice.n_nodes(step)):
            local_probs = [max(float(probability), 0.0) for probability in base_probs]
            total = sum(local_probs)
            if total <= 0.0:
                local_probs = [0.25, 0.25, 0.25, 0.25]
                positivity_violations += 1
            else:
                if any(probability < -1e-12 for probability in base_probs):
                    positivity_violations += 1
                local_probs = [probability / total for probability in local_probs]
            min_probability = min(min_probability, min(local_probs))
            max_probability = max(max_probability, max(local_probs))
            lattice.set_probabilities(step, node, local_probs)

    diagnostics = {
        "strategy": "joint_analytical_2f",
        "min_probability": float(min_probability),
        "max_probability": float(max_probability),
        "positivity_violations": float(positivity_violations),
        "correlation_clips": float(correlation_clips),
        "effective_correlation": float(correlation),
    }
    return lattice, diagnostics


__all__ = [
    "ProductRecombiningLattice2D",
    "build_product_spot_lattice_2d",
]
