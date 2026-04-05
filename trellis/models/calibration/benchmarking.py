"""Calibration throughput benchmarks and persisted report helpers."""

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
DEFAULT_REPORT_STEM = "calibration_workflows"


def _freeze_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable mapping proxy."""
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class CalibrationBenchmarkMeasurement:
    """Stable timing summary for one calibration workflow benchmark."""

    label: str
    mode: str
    repeats: int
    warmups: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    calibrations_per_second: float
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
        object.__setattr__(self, "calibrations_per_second", float(self.calibrations_per_second))
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
            "calibrations_per_second": round(self.calibrations_per_second, 3),
            "run_seconds": [round(value, 6) for value in self.run_seconds],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CalibrationBenchmarkScenario:
    """One reproducible calibration benchmark scenario."""

    workflow: str
    label: str
    cold_runner: Callable[[], object] = field(repr=False, compare=False)
    warm_runner: Callable[[], object] | None = field(default=None, repr=False, compare=False)
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class CalibrationBenchmarkArtifacts:
    """Persisted files for one calibration benchmark report."""

    report: dict[str, object]
    json_path: Path
    text_path: Path


def benchmark_calibration_workflow(
    *,
    label: str,
    mode: str,
    runner: Callable[[], object],
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
    notes: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CalibrationBenchmarkMeasurement:
    """Return a timing summary for one calibration runner."""
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
    return CalibrationBenchmarkMeasurement(
        label=label,
        mode=mode,
        repeats=repeats,
        warmups=warmups,
        mean_seconds=mean_seconds,
        median_seconds=float(raw_np.median(samples)),
        min_seconds=float(raw_np.min(samples)),
        max_seconds=float(raw_np.max(samples)),
        calibrations_per_second=(float(1.0 / mean_seconds) if mean_seconds > 0.0 else float("inf")),
        run_seconds=tuple(run_seconds),
        notes=tuple(notes or ()),
        metadata=metadata or {},
    )


def benchmark_calibration_scenario(
    scenario: CalibrationBenchmarkScenario,
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark one scenario in cold and optional warm-start modes."""
    cold = benchmark_calibration_workflow(
        label=scenario.label,
        mode="cold",
        runner=scenario.cold_runner,
        repeats=repeats,
        warmups=warmups,
        timer=timer,
        notes=scenario.notes,
        metadata=scenario.metadata,
    )
    warm = None
    warm_speedup = None
    if scenario.warm_runner is not None:
        warm = benchmark_calibration_workflow(
            label=scenario.label,
            mode="warm",
            runner=scenario.warm_runner,
            repeats=repeats,
            warmups=warmups,
            timer=timer,
            notes=scenario.notes,
            metadata=scenario.metadata,
        )
        warm_speedup = (
            float(cold.mean_seconds / warm.mean_seconds)
            if warm.mean_seconds > 0.0 else float("inf")
        )

    return {
        "workflow": scenario.workflow,
        "label": scenario.label,
        "notes": list(scenario.notes),
        "metadata": dict(scenario.metadata),
        "cold": cold.to_dict(),
        "warm": None if warm is None else warm.to_dict(),
        "warm_speedup": None if warm_speedup is None else round(warm_speedup, 3),
    }


def build_calibration_benchmark_report(
    *,
    benchmark_name: str,
    cases: Sequence[Mapping[str, object]],
    notes: Sequence[str] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble one persisted calibration benchmark report."""
    cases = [dict(case) for case in cases]
    cold_means = [float(case["cold"]["mean_seconds"]) for case in cases]
    warm_means = [
        float(case["warm"]["mean_seconds"])
        for case in cases
        if isinstance(case.get("warm"), Mapping)
    ]
    speedups = [
        float(case["warm_speedup"])
        for case in cases
        if case.get("warm_speedup") is not None
    ]
    return {
        "benchmark_name": benchmark_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": dict(environment or _default_environment()),
        "summary": {
            "workflow_count": len(cases),
            "warm_start_workflow_count": sum(1 for case in cases if case.get("warm") is not None),
            "cold_mean_seconds": round(float(raw_np.mean(cold_means)), 6) if cold_means else 0.0,
            "warm_mean_seconds": round(float(raw_np.mean(warm_means)), 6) if warm_means else 0.0,
            "average_warm_speedup": round(float(raw_np.mean(speedups)), 3) if speedups else None,
        },
        "cases": cases,
        "notes": list(notes or ()),
    }


def render_calibration_benchmark_report(report: Mapping[str, object]) -> str:
    """Render a markdown report from a calibration benchmark payload."""
    summary = report["summary"]
    environment = report["environment"]

    lines = [
        f"# Calibration Benchmark: `{report['benchmark_name']}`",
        f"- Created at: `{report.get('created_at', '')}`",
        f"- Workflows: `{summary['workflow_count']}`",
        f"- Warm-start workflows: `{summary['warm_start_workflow_count']}`",
        f"- Avg cold mean seconds: `{summary['cold_mean_seconds']}`",
        f"- Avg warm mean seconds: `{summary['warm_mean_seconds']}`",
    ]
    if summary.get("average_warm_speedup") is not None:
        lines.append(f"- Avg warm speedup: `{summary['average_warm_speedup']}`x")

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
        warm = case.get("warm")
        lines.extend(
            [
                "",
                f"### `{case['workflow']}` {case['label']}",
                f"- Cold mean: `{cold['mean_seconds']}` s",
                f"- Cold throughput: `{cold['calibrations_per_second']}` runs/s",
            ]
        )
        if warm is not None:
            lines.append(f"- Warm mean: `{warm['mean_seconds']}` s")
            lines.append(f"- Warm throughput: `{warm['calibrations_per_second']}` runs/s")
            lines.append(f"- Warm speedup: `{case['warm_speedup']}`x")
        else:
            lines.append("- Warm start: `n/a`")
        metadata = case.get("metadata") or {}
        if metadata:
            lines.append(
                "- Metadata: "
                + ", ".join(f"`{key}`={value!r}" for key, value in sorted(metadata.items()))
            )
        notes = case.get("notes") or []
        if notes:
            lines.extend(f"- Note: {note}" for note in notes)
    return "\n".join(lines) + "\n"


def save_calibration_benchmark_report(
    report: Mapping[str, object],
    *,
    root: Path | None = None,
    stem: str = DEFAULT_REPORT_STEM,
) -> CalibrationBenchmarkArtifacts:
    """Persist a calibration benchmark report as JSON plus Markdown."""
    root = (root or DEFAULT_REPORT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    text_path = root / f"{stem}.md"
    payload = dict(report)
    payload["json_path"] = str(json_path)
    payload["text_path"] = str(text_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_calibration_benchmark_report(payload))
    return CalibrationBenchmarkArtifacts(report=payload, json_path=json_path, text_path=text_path)


def build_supported_calibration_benchmark_report(
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark the supported calibration workflows and return a report payload."""
    cases = [
        benchmark_calibration_scenario(
            scenario,
            repeats=repeats,
            warmups=warmups,
            timer=timer,
        )
        for scenario in supported_calibration_benchmark_scenarios()
    ]
    return build_calibration_benchmark_report(
        benchmark_name="supported_calibration_workflows",
        cases=cases,
        notes=(
            "Cold-start and warm-start runs share the same synthetic benchmark fixtures.",
            "Warm-start baselines use workflow-native seed hooks where the workflow supports them.",
        ),
    )


def supported_calibration_benchmark_scenarios() -> tuple[CalibrationBenchmarkScenario, ...]:
    """Return the checked calibration benchmark fixtures."""
    from trellis.core.market_state import MarketState
    from trellis.core.types import DayCountConvention, Frequency
    from trellis.curves.yield_curve import YieldCurve
    from trellis.data.schema import MarketSnapshot
    from trellis.instruments._agent.swaption import SwaptionSpec
    from trellis.models.bermudan_swaption_tree import BermudanSwaptionTreeSpec, price_bermudan_swaption_tree
    from trellis.models.calibration.heston_fit import calibrate_heston_smile_workflow
    from trellis.models.calibration.local_vol import calibrate_local_vol_surface_workflow
    from trellis.models.calibration.rates import HullWhiteCalibrationInstrument, calibrate_hull_white
    from trellis.models.calibration.sabr_fit import calibrate_sabr_smile_workflow
    from trellis.models.processes.heston import Heston
    from trellis.models.processes.sabr import SABRProcess
    from trellis.models.transforms.fft_pricer import fft_price

    settle = date(2024, 11, 15)

    def market_state() -> MarketState:
        snapshot = MarketSnapshot(
            as_of=settle,
            source="benchmark",
            discount_curves={"usd_ois": YieldCurve.flat(0.04)},
            forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.041)},
            provenance={"source": "benchmark", "source_kind": "synthetic_input"},
        )
        return snapshot.to_market_state(
            settlement=settle,
            discount_curve="usd_ois",
            forecast_curve="USD-SOFR-3M",
        )

    hw_state = market_state()
    hw_mean_reversion = 0.08
    hw_sigma = 0.006
    hw_specs = (
        BermudanSwaptionTreeSpec(
            notional=1_000_000.0,
            strike=0.047,
            exercise_dates=(date(2025, 11, 15),),
            swap_end=date(2030, 11, 15),
            swap_frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            is_payer=True,
        ),
        BermudanSwaptionTreeSpec(
            notional=1_000_000.0,
            strike=0.048,
            exercise_dates=(date(2026, 11, 15),),
            swap_end=date(2031, 11, 15),
            swap_frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            is_payer=True,
        ),
    )
    hw_instruments = tuple(
        HullWhiteCalibrationInstrument(
            notional=spec.notional,
            strike=spec.strike,
            exercise_date=spec.exercise_dates[0],
            swap_end=spec.swap_end,
            quote=price_bermudan_swaption_tree(
                hw_state,
                spec,
                model="hull_white",
                mean_reversion=hw_mean_reversion,
                sigma=hw_sigma,
            ),
            quote_kind="price",
            swap_frequency=spec.swap_frequency,
            day_count=spec.day_count,
            rate_index=spec.rate_index,
            is_payer=spec.is_payer,
        )
        for spec in hw_specs
    )

    sabr_true = SABRProcess(0.20, 0.5, -0.3, 0.4)
    sabr_strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    sabr_forward = 100.0
    sabr_expiry = 1.0
    sabr_beta = 0.5
    sabr_market_vols = [sabr_true.implied_vol(sabr_forward, strike, sabr_expiry) for strike in sabr_strikes]

    heston_spot = 100.0
    heston_rate = 0.02
    heston_expiry = 1.0
    heston_strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    heston_true = Heston(mu=heston_rate, kappa=1.8, theta=0.04, xi=0.35, rho=-0.6, v0=0.05)
    heston_market_vols = []
    for strike in heston_strikes:
        price = fft_price(
            lambda u: heston_true.characteristic_function(u, heston_expiry, log_spot=raw_np.log(heston_spot)),
            heston_spot,
            strike,
            heston_expiry,
            heston_rate,
            N=1024,
            eta=0.1,
        )
        from trellis.models.calibration.implied_vol import implied_vol

        heston_market_vols.append(
            implied_vol(price, heston_spot, strike, heston_expiry, heston_rate, option_type="call")
        )

    local_vol_strikes = raw_np.linspace(60.0, 150.0, 30)
    local_vol_expiries = raw_np.linspace(0.1, 3.0, 15)
    local_vol_surface = raw_np.array(
        [
            [0.24, 0.23, 0.22, 0.21, 0.20, 0.19],
            [0.245, 0.235, 0.225, 0.215, 0.205, 0.195],
            [0.25, 0.24, 0.23, 0.22, 0.21, 0.20],
            [0.255, 0.245, 0.235, 0.225, 0.215, 0.205],
            [0.26, 0.25, 0.24, 0.23, 0.22, 0.21],
            [0.265, 0.255, 0.245, 0.235, 0.225, 0.215],
        ],
        dtype=float,
    )
    local_vol_surface = raw_np.interp(
        raw_np.linspace(0, 5, len(local_vol_expiries) * len(local_vol_strikes)),
        raw_np.arange(local_vol_surface.size),
        local_vol_surface.ravel(),
    ).reshape(len(local_vol_expiries), len(local_vol_strikes))

    return (
        CalibrationBenchmarkScenario(
            workflow="hull_white",
            label="swaption_strip",
            cold_runner=lambda: calibrate_hull_white(hw_instruments, hw_state, initial_guess=(0.20, 0.02)),
            warm_runner=lambda: calibrate_hull_white(
                hw_instruments,
                hw_state,
                initial_guess=(hw_mean_reversion, hw_sigma),
            ),
            notes=("least_squares", "tree_pricing"),
            metadata={"instrument_count": len(hw_instruments), "warm_start": True},
        ),
        CalibrationBenchmarkScenario(
            workflow="sabr",
            label="single_smile",
            cold_runner=lambda: calibrate_sabr_smile_workflow(
                sabr_forward,
                sabr_expiry,
                sabr_strikes,
                sabr_market_vols,
                beta=sabr_beta,
                surface_name="benchmark_sabr",
            ),
            warm_runner=lambda: calibrate_sabr_smile_workflow(
                sabr_forward,
                sabr_expiry,
                sabr_strikes,
                sabr_market_vols,
                beta=sabr_beta,
                surface_name="benchmark_sabr",
                initial_guess=(0.20, -0.3, 0.4),
            ),
            notes=("least_squares", "implied_vol_fit"),
            metadata={"point_count": len(sabr_strikes), "warm_start": True},
        ),
        CalibrationBenchmarkScenario(
            workflow="heston",
            label="single_smile",
            cold_runner=lambda: calibrate_heston_smile_workflow(
                heston_spot,
                heston_expiry,
                heston_strikes,
                heston_market_vols,
                rate=heston_rate,
                surface_name="benchmark_heston",
                parameter_set_name="benchmark_heston",
            ),
            warm_runner=lambda: calibrate_heston_smile_workflow(
                heston_spot,
                heston_expiry,
                heston_strikes,
                heston_market_vols,
                rate=heston_rate,
                surface_name="benchmark_heston",
                parameter_set_name="benchmark_heston",
                warm_start=(1.8, 0.04, 0.35, -0.6, 0.05),
            ),
            notes=("least_squares", "fft_pricing"),
            metadata={"point_count": len(heston_strikes), "warm_start": True},
        ),
        CalibrationBenchmarkScenario(
            workflow="local_vol",
            label="dupire_surface",
            cold_runner=lambda: calibrate_local_vol_surface_workflow(
                local_vol_strikes,
                local_vol_expiries,
                local_vol_surface,
                100.0,
                0.02,
                surface_name="benchmark_local_vol",
            ),
            notes=("dupire", "workflow_surface"),
            metadata={
                "grid_shape": [len(local_vol_expiries), len(local_vol_strikes)],
                "warm_start": False,
            },
        ),
    )


def _default_environment() -> dict[str, object]:
    """Return environment metadata for persisted benchmark reports."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


__all__ = [
    "CalibrationBenchmarkMeasurement",
    "CalibrationBenchmarkScenario",
    "CalibrationBenchmarkArtifacts",
    "benchmark_calibration_workflow",
    "benchmark_calibration_scenario",
    "build_calibration_benchmark_report",
    "render_calibration_benchmark_report",
    "save_calibration_benchmark_report",
    "build_supported_calibration_benchmark_report",
    "supported_calibration_benchmark_scenarios",
]
