"""Benchmark representative Monte Carlo path kernels and write a baseline artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.models.monte_carlo import MonteCarloEngine
from trellis.models.monte_carlo.profiling import benchmark_path_kernel
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.processes.gbm import GBM
from trellis.cli_paths import resolve_repo_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs" / "benchmarks" / "monte_carlo_path_kernels.json"),
        help="Output JSON file for the benchmark baseline.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    return parser.parse_args(argv)


def _scalar_cases(path_counts: tuple[int, ...], *, repeats: int, warmups: int):
    process = GBM(mu=0.05, sigma=0.20)
    for n_paths in path_counts:
        yield {
            "label": f"gbm_exact_{n_paths}",
            "initial_state": 100.0,
            "factory": lambda n_paths=n_paths: MonteCarloEngine(
                process,
                n_paths=n_paths,
                n_steps=64,
                seed=42,
                method="exact",
            ),
            "repeats": repeats,
            "warmups": warmups,
            "notes": ("scalar_state", "exact_transition"),
        }


def _vector_cases(path_counts: tuple[int, ...], *, repeats: int, warmups: int):
    process = CorrelatedGBM(
        mu=[0.04, 0.03, 0.05],
        sigma=[0.20, 0.18, 0.22],
        corr=[
            [1.0, 0.20, 0.15],
            [0.20, 1.0, 0.10],
            [0.15, 0.10, 1.0],
        ],
    )
    initial_state = [100.0, 95.0, 110.0]
    for n_paths in path_counts:
        yield {
            "label": f"correlated_gbm_exact_{n_paths}",
            "initial_state": initial_state,
            "factory": lambda n_paths=n_paths: MonteCarloEngine(
                process,
                n_paths=n_paths,
                n_steps=64,
                seed=42,
                method="exact",
            ),
            "repeats": repeats,
            "warmups": warmups,
            "notes": ("vector_state", "exact_transition", "batched_path_kernel"),
        }


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    output_path = resolve_repo_path(
        args.output,
        ROOT / "docs" / "benchmarks" / "monte_carlo_path_kernels.json",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_counts = (1024, 4096, 16384)
    results = []

    print(f"\n{'#' * 72}")
    print("# Monte Carlo path-kernel benchmark")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"# Repeats: {args.repeats}")
    print(f"# Warmups: {args.warmups}")
    print(f"# Output: {output_path}")
    print(f"{'#' * 72}")

    for case in (*_scalar_cases(path_counts, repeats=args.repeats, warmups=args.warmups), *_vector_cases(path_counts, repeats=args.repeats, warmups=args.warmups)):
        print(f"\n[benchmark] {case['label']}")
        summary = benchmark_path_kernel(
            label=case["label"],
            engine_factory=case["factory"],
            initial_state=case["initial_state"],
            T=1.0,
            repeats=case["repeats"],
            warmups=case["warmups"],
            notes=case["notes"],
        )
        payload = summary.to_dict()
        print(
            f"  mean={payload['mean_seconds']:.6f}s "
            f"min={payload['min_seconds']:.6f}s "
            f"max={payload['max_seconds']:.6f}s "
            f"throughput={payload['paths_per_second']:.1f} paths/s"
        )
        results.append(payload)
        output_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nSaved benchmark baseline to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
