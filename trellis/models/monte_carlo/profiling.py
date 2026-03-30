"""Benchmark helpers for Monte Carlo path kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter
from typing import Callable

from trellis.core.differentiable import get_numpy

np = get_numpy()


@dataclass(frozen=True)
class MonteCarloPathKernelBenchmark:
    """Stable timing summary for one Monte Carlo path-kernel benchmark."""

    label: str
    method: str
    n_paths: int
    n_steps: int
    state_dim: int
    factor_dim: int
    repeats: int
    warmups: int
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    paths_per_second: float
    sample_shape: tuple[int, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the benchmark summary."""
        return {
            "label": self.label,
            "method": self.method,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "state_dim": self.state_dim,
            "factor_dim": self.factor_dim,
            "repeats": self.repeats,
            "warmups": self.warmups,
            "mean_seconds": self.mean_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "paths_per_second": self.paths_per_second,
            "sample_shape": list(self.sample_shape),
            "notes": list(self.notes),
        }


def benchmark_path_kernel(
    *,
    label: str,
    engine_factory: Callable[[], object],
    initial_state,
    T: float,
    repeats: int = 5,
    warmups: int = 1,
    timer: Callable[[], float] = perf_counter,
    notes: tuple[str, ...] | list[str] = (),
) -> MonteCarloPathKernelBenchmark:
    """Benchmark one path-generation kernel with deterministic replay settings."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmups < 0:
        raise ValueError("warmups must be >= 0")

    notes = tuple(str(note) for note in notes if str(note).strip())

    sample_shape: tuple[int, ...] | None = None
    n_paths = 0
    n_steps = 0
    state_dim = 1
    factor_dim = 1
    method = "unknown"

    for _ in range(warmups):
        engine = engine_factory()
        _ = engine.simulate(initial_state, T)

    durations: list[float] = []
    for _ in range(repeats):
        engine = engine_factory()
        start = timer()
        paths = engine.simulate(initial_state, T)
        durations.append(timer() - start)

        normalized = np.asarray(paths)
        if sample_shape is None:
            sample_shape = tuple(int(value) for value in normalized.shape)
            if normalized.ndim == 2:
                state_dim = 1
            elif normalized.ndim == 3:
                state_dim = int(normalized.shape[-1])
            else:
                raise ValueError(
                    f"Expected Monte Carlo paths with rank 2 or 3; received shape {normalized.shape}."
                )

        n_paths = int(getattr(engine, "n_paths", normalized.shape[0]))
        n_steps = int(getattr(engine, "n_steps", max(normalized.shape[1] - 1, 0)))
        method = str(getattr(engine, "method", getattr(getattr(engine, "scheme", None), "name", "unknown")))
        process = getattr(engine, "process", None)
        factor_dim = int(getattr(process, "factor_dim", state_dim))

    mean_seconds = float(mean(durations))
    min_seconds = float(min(durations))
    max_seconds = float(max(durations))
    throughput = float(n_paths) / mean_seconds if mean_seconds > 0.0 else float("inf")

    return MonteCarloPathKernelBenchmark(
        label=label,
        method=method,
        n_paths=n_paths,
        n_steps=n_steps,
        state_dim=state_dim,
        factor_dim=factor_dim,
        repeats=repeats,
        warmups=warmups,
        mean_seconds=round(mean_seconds, 6),
        min_seconds=round(min_seconds, 6),
        max_seconds=round(max_seconds, 6),
        paths_per_second=round(throughput, 3),
        sample_shape=sample_shape or (),
        notes=notes,
    )


__all__ = [
    "MonteCarloPathKernelBenchmark",
    "benchmark_path_kernel",
]
