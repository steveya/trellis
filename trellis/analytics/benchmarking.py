"""Pod-risk throughput benchmarks and persisted report helpers."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as raw_np


DEFAULT_REPORT_ROOT = Path("docs") / "benchmarks"
DEFAULT_REPORT_STEM = "pod_risk_workflows"
BENCHMARK_SETTLE = date(2024, 11, 15)


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class RiskBenchmarkMeasurement:
    """Stable timing summary for one pod-risk workflow benchmark."""

    label: str
    mode: str
    repeats: int
    warmups: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    runs_per_second: float
    run_seconds: tuple[float, ...]
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repeats", int(self.repeats))
        object.__setattr__(self, "warmups", int(self.warmups))
        object.__setattr__(self, "mean_seconds", float(self.mean_seconds))
        object.__setattr__(self, "median_seconds", float(self.median_seconds))
        object.__setattr__(self, "min_seconds", float(self.min_seconds))
        object.__setattr__(self, "max_seconds", float(self.max_seconds))
        object.__setattr__(self, "runs_per_second", float(self.runs_per_second))
        object.__setattr__(self, "run_seconds", tuple(float(value) for value in self.run_seconds))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly measurement payload."""
        return {
            "label": self.label,
            "mode": self.mode,
            "repeats": self.repeats,
            "warmups": self.warmups,
            "mean_seconds": round(self.mean_seconds, 6),
            "median_seconds": round(self.median_seconds, 6),
            "min_seconds": round(self.min_seconds, 6),
            "max_seconds": round(self.max_seconds, 6),
            "runs_per_second": round(self.runs_per_second, 3),
            "run_seconds": [round(value, 6) for value in self.run_seconds],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RiskBenchmarkScenario:
    """One reproducible pod-risk benchmark scenario."""

    workflow: str
    label: str
    cold_runner: Callable[[], object] = field(repr=False, compare=False)
    steady_runner: Callable[[], object] | None = field(default=None, repr=False, compare=False)
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class RiskBenchmarkArtifacts:
    """Persisted files for one pod-risk benchmark report."""

    report: dict[str, object]
    json_path: Path
    text_path: Path


def benchmark_risk_workflow(
    *,
    label: str,
    mode: str,
    runner: Callable[[], object],
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
    notes: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> RiskBenchmarkMeasurement:
    """Return a timing summary for one risk workflow runner."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")

    timer = timer or time.perf_counter
    for _ in range(warmups):
        runner()

    run_seconds: list[float] = []
    for _ in range(repeats):
        start = float(timer())
        runner()
        stop = float(timer())
        run_seconds.append(stop - start)

    samples = raw_np.asarray(run_seconds, dtype=float)
    mean_seconds = float(raw_np.mean(samples))
    return RiskBenchmarkMeasurement(
        label=label,
        mode=mode,
        repeats=repeats,
        warmups=warmups,
        mean_seconds=mean_seconds,
        median_seconds=float(raw_np.median(samples)),
        min_seconds=float(raw_np.min(samples)),
        max_seconds=float(raw_np.max(samples)),
        runs_per_second=(float(1.0 / mean_seconds) if mean_seconds > 0.0 else float("inf")),
        run_seconds=tuple(run_seconds),
        notes=tuple(notes or ()),
        metadata=metadata or {},
    )


def benchmark_risk_scenario(
    scenario: RiskBenchmarkScenario,
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark one scenario in cold and optional steady-state modes."""
    cold = benchmark_risk_workflow(
        label=scenario.label,
        mode="cold",
        runner=scenario.cold_runner,
        repeats=repeats,
        warmups=warmups,
        timer=timer,
        notes=scenario.notes,
        metadata=scenario.metadata,
    )
    steady = None
    steady_speedup = None
    if scenario.steady_runner is not None:
        steady = benchmark_risk_workflow(
            label=scenario.label,
            mode="steady_state",
            runner=scenario.steady_runner,
            repeats=repeats,
            warmups=warmups,
            timer=timer,
            notes=scenario.notes,
            metadata=scenario.metadata,
        )
        steady_speedup = (
            float(cold.mean_seconds / steady.mean_seconds)
            if steady.mean_seconds > 0.0
            else float("inf")
        )
    return {
        "workflow": scenario.workflow,
        "label": scenario.label,
        "notes": list(scenario.notes),
        "metadata": dict(scenario.metadata),
        "cold": cold.to_dict(),
        "steady": None if steady is None else steady.to_dict(),
        "steady_speedup": None if steady_speedup is None else round(steady_speedup, 3),
    }


def build_risk_benchmark_report(
    *,
    benchmark_name: str,
    cases: Sequence[Mapping[str, object]],
    notes: Sequence[str] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble one persisted pod-risk benchmark report."""
    cases = [dict(case) for case in cases]
    cold_means = [float(case["cold"]["mean_seconds"]) for case in cases]
    steady_means = [
        float(case["steady"]["mean_seconds"])
        for case in cases
        if isinstance(case.get("steady"), Mapping)
    ]
    speedups = [
        float(case["steady_speedup"])
        for case in cases
        if case.get("steady_speedup") is not None
    ]
    return {
        "benchmark_name": benchmark_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": dict(environment or _default_environment()),
        "summary": {
            "workflow_count": len(cases),
            "steady_state_workflow_count": sum(
                1 for case in cases if case.get("steady") is not None
            ),
            "cold_mean_seconds": round(float(raw_np.mean(cold_means)), 6) if cold_means else 0.0,
            "steady_mean_seconds": round(float(raw_np.mean(steady_means)), 6) if steady_means else 0.0,
            "average_steady_speedup": round(float(raw_np.mean(speedups)), 3) if speedups else None,
        },
        "cases": cases,
        "notes": list(notes or ()),
    }


def render_risk_benchmark_report(report: Mapping[str, object]) -> str:
    """Render a markdown report from a pod-risk benchmark payload."""
    summary = report["summary"]
    environment = report["environment"]

    lines = [
        f"# Pod Risk Benchmark: `{report['benchmark_name']}`",
        f"- Created at: `{report.get('created_at', '')}`",
        f"- Workflows: `{summary['workflow_count']}`",
        f"- Steady-state workflows: `{summary['steady_state_workflow_count']}`",
        f"- Avg cold mean seconds: `{summary['cold_mean_seconds']}`",
        f"- Avg steady mean seconds: `{summary['steady_mean_seconds']}`",
    ]
    if summary.get("average_steady_speedup") is not None:
        lines.append(f"- Avg steady speedup: `{summary['average_steady_speedup']}`x")

    lines.extend(
        [
            "",
            "## Environment",
            f"- Python: `{environment['python_version']}`",
            f"- Platform: `{environment['platform']}`",
        ]
    )

    if report.get("notes"):
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in report["notes"])

    lines.extend(["", "## Workflow Results"])
    for case in report["cases"]:
        cold = case["cold"]
        steady = case.get("steady")
        lines.extend(
            [
                "",
                f"### `{case['workflow']}` {case['label']}",
                f"- Cold mean: `{cold['mean_seconds']}` s",
                f"- Cold throughput: `{cold['runs_per_second']}` runs/s",
            ]
        )
        if steady is not None:
            lines.append(f"- Steady mean: `{steady['mean_seconds']}` s")
            lines.append(f"- Steady throughput: `{steady['runs_per_second']}` runs/s")
            lines.append(f"- Steady speedup: `{case['steady_speedup']}`x")
        else:
            lines.append("- Steady state: `n/a`")
        metadata = case.get("metadata") or {}
        if metadata:
            lines.append(
                "- Metadata: "
                + ", ".join(f"`{key}`={value!r}" for key, value in sorted(metadata.items()))
            )
        case_notes = case.get("notes") or []
        if case_notes:
            lines.extend(f"- Note: {note}" for note in case_notes)
    return "\n".join(lines) + "\n"


def save_risk_benchmark_report(
    report: Mapping[str, object],
    *,
    root: Path | None = None,
    stem: str = DEFAULT_REPORT_STEM,
) -> RiskBenchmarkArtifacts:
    """Persist a pod-risk benchmark report as JSON plus Markdown."""
    root = (root or DEFAULT_REPORT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    text_path = root / f"{stem}.md"
    payload = dict(report)
    payload["json_path"] = str(json_path)
    payload["text_path"] = str(text_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_risk_benchmark_report(payload))
    return RiskBenchmarkArtifacts(report=payload, json_path=json_path, text_path=text_path)


def build_supported_pod_risk_benchmark_report(
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark the supported pod-risk workflows and return a report payload."""
    cases = [
        benchmark_risk_scenario(
            scenario,
            repeats=repeats,
            warmups=warmups,
            timer=timer,
        )
        for scenario in supported_pod_risk_benchmark_scenarios()
    ]
    return build_risk_benchmark_report(
        benchmark_name="pod_risk_workflows",
        cases=cases,
        notes=(
            "Cold runs rebuild sessions or pipelines for each sample, while steady-state runs reuse prebuilt runtime objects when the workflow allows it.",
            "Coverage includes shared scenario-result cube execution plus supported rates, volatility, and spot-risk analytics.",
        ),
    )


def supported_pod_risk_benchmark_scenarios() -> tuple[RiskBenchmarkScenario, ...]:
    """Return the checked pod-risk benchmark fixtures."""
    from trellis import Pipeline, Session
    from trellis.book import Book
    from trellis.conventions.day_count import DayCountConvention
    from trellis.core.payoff import DeterministicCashflowPayoff
    from trellis.core.types import Frequency
    from trellis.curves.bootstrap import (
        BootstrapConventionBundle,
        BootstrapCurveInputBundle,
        BootstrapInstrument,
        bootstrap_curve_result,
    )
    from trellis.curves.yield_curve import YieldCurve
    from trellis.data.schema import MarketSnapshot
    from trellis.instruments.bond import Bond
    from trellis.models.vol_surface import GridVolSurface

    class _VolPointPayoff:
        requirements = {"black_vol_surface"}

        def __init__(self, expiry: float, strike: float):
            self.expiry = float(expiry)
            self.strike = float(strike)

        def evaluate(self, market_state):
            return float(market_state.vol_surface.black_vol(self.expiry, self.strike))

    class _SpotQuadraticPayoff:
        requirements = {"spot"}

        def evaluate(self, market_state):
            spot = float(market_state.spot)
            return spot**2 + 0.5 * spot

    def benchmark_curve() -> YieldCurve:
        return YieldCurve(
            [1.0, 2.0, 5.0, 10.0, 30.0],
            [0.04, 0.041, 0.043, 0.046, 0.048],
        )

    def benchmark_bond() -> Bond:
        return Bond(
            face=100.0,
            coupon=0.045,
            maturity_date=date(2034, 11, 15),
            maturity=10,
            frequency=2,
        )

    def benchmark_book() -> Book:
        return Book(
            {
                "5Y": Bond(
                    face=100.0,
                    coupon=0.04,
                    maturity_date=date(2029, 11, 15),
                    maturity=5,
                    frequency=2,
                ),
                "10Y": benchmark_bond(),
            },
            notionals={"5Y": 1_000_000.0, "10Y": 750_000.0},
        )

    def benchmark_snapshot() -> MarketSnapshot:
        return MarketSnapshot(
            as_of=BENCHMARK_SETTLE,
            source="unit",
            discount_curves={"usd_ois": benchmark_curve()},
            vol_surfaces={
                "usd_smile": GridVolSurface(
                    expiries=(1.0, 2.0),
                    strikes=(90.0, 110.0),
                    vols=((0.25, 0.22), (0.27, 0.24)),
                )
            },
            underlier_spots={"SPX": 100.0},
            default_discount_curve="usd_ois",
            default_vol_surface="usd_smile",
            default_underlier_spot="SPX",
        )

    def bootstrapped_snapshot() -> MarketSnapshot:
        bundle = BootstrapCurveInputBundle(
            curve_name="usd_ois_boot",
            currency="USD",
            rate_index="USD-SOFR-3M",
            conventions=BootstrapConventionBundle(
                deposit_day_count=DayCountConvention.ACT_360,
                swap_fixed_frequency=Frequency.ANNUAL,
                swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
                swap_float_frequency=Frequency.QUARTERLY,
                swap_float_day_count=DayCountConvention.ACT_360,
            ),
            instruments=(
                BootstrapInstrument(tenor=1.0, quote=0.045, instrument_type="deposit", label="DEP1Y"),
                BootstrapInstrument(tenor=2.0, quote=0.046, instrument_type="swap", label="SWAP2Y"),
                BootstrapInstrument(tenor=5.0, quote=0.0475, instrument_type="swap", label="SWAP5Y"),
                BootstrapInstrument(tenor=10.0, quote=0.0485, instrument_type="swap", label="SWAP10Y"),
            ),
        )
        result = bootstrap_curve_result(bundle, max_iter=75, tol=1e-12)
        return MarketSnapshot(
            as_of=BENCHMARK_SETTLE,
            source="unit",
            discount_curves={"usd_ois_boot": result.curve},
            default_discount_curve="usd_ois_boot",
            provenance={
                "source": "unit",
                "source_kind": "mixed",
                "bootstrap_inputs": {"discount_curves": {"usd_ois_boot": bundle.to_payload()}},
                "bootstrap_runs": {"discount_curves": {"usd_ois_boot": result.to_payload()}},
            },
        )

    warm_pipeline = (
        Pipeline()
        .instruments(benchmark_book())
        .market_data(curve=benchmark_curve())
        .compute(["price", "dv01"])
        .scenarios(
            [
                {
                    "scenario_pack": "twist",
                    "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                    "amplitude_bps": 25.0,
                }
            ]
        )
    )
    warm_rates_session = Session(
        market_snapshot=bootstrapped_snapshot(),
        settlement=BENCHMARK_SETTLE,
    )
    warm_bond_payoff = DeterministicCashflowPayoff(benchmark_bond())
    warm_vol_session = Session(
        market_snapshot=benchmark_snapshot(),
        settlement=BENCHMARK_SETTLE,
    ).with_vol_surface_name("usd_smile")
    warm_vol_payoff = _VolPointPayoff(1.5, 100.0)
    warm_spot_session = Session(
        market_snapshot=benchmark_snapshot(),
        settlement=BENCHMARK_SETTLE,
    )
    warm_spot_payoff = _SpotQuadraticPayoff()

    return (
        RiskBenchmarkScenario(
            workflow="pipeline_scenarios",
            label="twist_cube",
            cold_runner=lambda: (
                Pipeline()
                .instruments(benchmark_book())
                .market_data(curve=benchmark_curve())
                .compute(["price", "dv01"])
                .scenarios(
                    [
                        {
                            "scenario_pack": "twist",
                            "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                            "amplitude_bps": 25.0,
                        }
                    ]
                )
                .run()
            ),
            steady_runner=lambda: warm_pipeline.run(),
            notes=("scenario_cube", "twist_pack"),
            metadata={"scenario_count": 2, "position_count": 2},
        ),
        RiskBenchmarkScenario(
            workflow="key_rate_durations",
            label="curve_rebuild_buckets",
            cold_runner=lambda: Session(
                market_snapshot=bootstrapped_snapshot(),
                settlement=BENCHMARK_SETTLE,
            ).analyze(
                DeterministicCashflowPayoff(benchmark_bond()),
                measures=[
                    {
                        "key_rate_durations": {
                            "methodology": "curve_rebuild",
                            "tenors": (1.0, 2.0, 5.0, 10.0),
                            "bump_bps": 25.0,
                        }
                    }
                ],
            ).key_rate_durations,
            steady_runner=lambda: warm_rates_session.analyze(
                warm_bond_payoff,
                measures=[
                    {
                        "key_rate_durations": {
                            "methodology": "curve_rebuild",
                            "tenors": (1.0, 2.0, 5.0, 10.0),
                            "bump_bps": 25.0,
                        }
                    }
                ],
            ).key_rate_durations,
            notes=("curve_rebuild", "rates_risk"),
            metadata={"bucket_count": 4, "methodology": "curve_rebuild"},
        ),
        RiskBenchmarkScenario(
            workflow="scenario_pnl",
            label="rebuild_pack",
            cold_runner=lambda: Session(
                market_snapshot=bootstrapped_snapshot(),
                settlement=BENCHMARK_SETTLE,
            ).analyze(
                DeterministicCashflowPayoff(benchmark_bond()),
                measures=[
                    {
                        "scenario_pnl": {
                            "methodology": "curve_rebuild",
                            "scenario_packs": ("twist", "butterfly"),
                            "bucket_tenors": (1.0, 2.0, 5.0, 10.0),
                        }
                    }
                ],
            ).scenario_pnl,
            steady_runner=lambda: warm_rates_session.analyze(
                warm_bond_payoff,
                measures=[
                    {
                        "scenario_pnl": {
                            "methodology": "curve_rebuild",
                            "scenario_packs": ("twist", "butterfly"),
                            "bucket_tenors": (1.0, 2.0, 5.0, 10.0),
                        }
                    }
                ],
            ).scenario_pnl,
            notes=("curve_rebuild", "named_scenarios"),
            metadata={"scenario_count": 4, "methodology": "curve_rebuild"},
        ),
        RiskBenchmarkScenario(
            workflow="vega",
            label="bucketed_surface",
            cold_runner=lambda: Session(
                market_snapshot=benchmark_snapshot(),
                settlement=BENCHMARK_SETTLE,
            ).with_vol_surface_name("usd_smile").analyze(
                _VolPointPayoff(1.5, 100.0),
                measures=[
                    {
                        "vega": {
                            "expiries": (1.0, 1.5, 2.0),
                            "strikes": (90.0, 100.0, 110.0),
                            "bump_pct": 1.0,
                        }
                    }
                ],
            ).vega,
            steady_runner=lambda: warm_vol_session.analyze(
                warm_vol_payoff,
                measures=[
                    {
                        "vega": {
                            "expiries": (1.0, 1.5, 2.0),
                            "strikes": (90.0, 100.0, 110.0),
                            "bump_pct": 1.0,
                        }
                    }
                ],
            ).vega,
            notes=("vol_surface", "bucketed_vega"),
            metadata={"grid_shape": [3, 3]},
        ),
        RiskBenchmarkScenario(
            workflow="spot_greeks",
            label="delta_gamma_theta_bundle",
            cold_runner=lambda: Session(
                market_snapshot=benchmark_snapshot(),
                settlement=BENCHMARK_SETTLE,
            ).analyze(
                _SpotQuadraticPayoff(),
                measures=[
                    {"delta": {"bump_pct": 1.0}},
                    {"gamma": {"bump_pct": 1.0}},
                    {"theta": {"day_step": 1}},
                ],
            ),
            steady_runner=lambda: warm_spot_session.analyze(
                warm_spot_payoff,
                measures=[
                    {"delta": {"bump_pct": 1.0}},
                    {"gamma": {"bump_pct": 1.0}},
                    {"theta": {"day_step": 1}},
                ],
            ),
            notes=("spot_risk", "bundle"),
            metadata={"measure_count": 3},
        ),
    )


def _default_environment() -> dict[str, object]:
    """Return environment metadata for persisted benchmark reports."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


__all__ = [
    "RiskBenchmarkMeasurement",
    "RiskBenchmarkScenario",
    "RiskBenchmarkArtifacts",
    "benchmark_risk_workflow",
    "benchmark_risk_scenario",
    "build_risk_benchmark_report",
    "render_risk_benchmark_report",
    "save_risk_benchmark_report",
    "build_supported_pod_risk_benchmark_report",
    "supported_pod_risk_benchmark_scenarios",
]
