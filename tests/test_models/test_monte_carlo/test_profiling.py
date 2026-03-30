"""Tests for Monte Carlo path-kernel profiling helpers."""

from __future__ import annotations

import numpy as np
import pytest


def test_benchmark_path_kernel_returns_stable_summary():
    from trellis.models.monte_carlo.profiling import benchmark_path_kernel

    class _Process:
        factor_dim = 2

    class _Engine:
        def __init__(self):
            self.n_paths = 8
            self.n_steps = 4
            self.method = "exact"
            self.process = _Process()

        def simulate(self, x0, T):
            _ = (x0, T)
            return np.zeros((self.n_paths, self.n_steps + 1, 2), dtype=float)

    timer_values = iter([1.0, 1.25, 2.0, 2.2, 3.0, 3.2])

    def fake_timer() -> float:
        return next(timer_values)

    summary = benchmark_path_kernel(
        label="vector_exact",
        engine_factory=_Engine,
        initial_state=np.array([100.0, 95.0], dtype=float),
        T=1.0,
        repeats=3,
        warmups=1,
        timer=fake_timer,
        notes=("vector_state", "exact_transition"),
    )

    assert summary.label == "vector_exact"
    assert summary.method == "exact"
    assert summary.n_paths == 8
    assert summary.n_steps == 4
    assert summary.state_dim == 2
    assert summary.factor_dim == 2
    assert summary.sample_shape == (8, 5, 2)
    assert summary.mean_seconds == pytest.approx(0.216667, rel=1e-6)
    assert summary.paths_per_second == pytest.approx(36.923, rel=1e-6)
    assert summary.to_dict()["notes"] == ["vector_state", "exact_transition"]
