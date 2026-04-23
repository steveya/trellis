"""Calibration throughput benchmarks and persisted report helpers."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as raw_np

from trellis.core.types import DayCountConvention


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


@dataclass(frozen=True)
class CalibrationPerturbationDiagnostic:
    """Metric-level stability summary for one deterministic quote perturbation."""

    label: str
    perturbation_size: float
    baseline_metrics: Mapping[str, float]
    perturbed_metrics: Mapping[str, float]
    absolute_changes: Mapping[str, float]
    relative_changes: Mapping[str, float]
    threshold_breaches: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "perturbation_size", float(self.perturbation_size))
        object.__setattr__(
            self,
            "baseline_metrics",
            _freeze_mapping({key: float(value) for key, value in self.baseline_metrics.items()}),
        )
        object.__setattr__(
            self,
            "perturbed_metrics",
            _freeze_mapping({key: float(value) for key, value in self.perturbed_metrics.items()}),
        )
        object.__setattr__(
            self,
            "absolute_changes",
            _freeze_mapping({key: float(value) for key, value in self.absolute_changes.items()}),
        )
        object.__setattr__(
            self,
            "relative_changes",
            _freeze_mapping({key: float(value) for key, value in self.relative_changes.items()}),
        )
        object.__setattr__(
            self,
            "threshold_breaches",
            _freeze_mapping({key: float(value) for key, value in self.threshold_breaches.items()}),
        )

    @property
    def max_abs_change(self) -> float:
        """Return the largest absolute metric move."""
        return max((abs(float(value)) for value in self.absolute_changes.values()), default=0.0)

    @property
    def max_relative_change(self) -> float:
        """Return the largest relative metric move."""
        return max((abs(float(value)) for value in self.relative_changes.values()), default=0.0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly perturbation diagnostic payload."""
        return {
            "label": self.label,
            "perturbation_size": float(self.perturbation_size),
            "baseline_metrics": dict(self.baseline_metrics),
            "perturbed_metrics": dict(self.perturbed_metrics),
            "absolute_changes": dict(self.absolute_changes),
            "relative_changes": dict(self.relative_changes),
            "max_abs_change": float(self.max_abs_change),
            "max_relative_change": float(self.max_relative_change),
            "threshold_breaches": dict(self.threshold_breaches),
            "status": "breach" if self.threshold_breaches else "pass",
        }


@dataclass(frozen=True)
class CalibrationLatencyEnvelope:
    """Explicit timing envelope for one calibration benchmark fixture."""

    workflow: str
    label: str
    fixture_style: str
    instrument_count: int | None = None
    quote_count: int | None = None
    cold_mean_limit_seconds: float = 0.0
    cold_max_limit_seconds: float | None = None
    warm_mean_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow", str(self.workflow))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "fixture_style", str(self.fixture_style))
        if self.instrument_count is not None:
            object.__setattr__(self, "instrument_count", int(self.instrument_count))
        if self.quote_count is not None:
            object.__setattr__(self, "quote_count", int(self.quote_count))
        object.__setattr__(self, "cold_mean_limit_seconds", float(self.cold_mean_limit_seconds))
        if self.cold_max_limit_seconds is not None:
            object.__setattr__(self, "cold_max_limit_seconds", float(self.cold_max_limit_seconds))
        if self.warm_mean_limit_seconds is not None:
            object.__setattr__(self, "warm_mean_limit_seconds", float(self.warm_mean_limit_seconds))

    def to_dict(self) -> dict[str, object]:
        """Return the envelope thresholds as a JSON-friendly payload."""
        return {
            "workflow": self.workflow,
            "label": self.label,
            "fixture_style": self.fixture_style,
            "instrument_count": self.instrument_count,
            "quote_count": self.quote_count,
            "cold_mean_limit_seconds": float(self.cold_mean_limit_seconds),
            "cold_max_limit_seconds": self.cold_max_limit_seconds,
            "warm_mean_limit_seconds": self.warm_mean_limit_seconds,
        }


@dataclass(frozen=True)
class _BenchmarkBasketTrancheSpec:
    """Local basket tranche spec for benchmark quote generation."""

    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date
    recovery: float = 0.4
    correlation: float | None = None


@dataclass(frozen=True)
class _BenchmarkQuantoOptionSpec:
    """Local quanto option spec used for bounded hybrid benchmark quotes."""

    notional: float
    strike: float
    expiry_date: date
    fx_pair: str
    underlier_currency: str = "EUR"
    domestic_currency: str = "USD"
    option_type: str = "call"
    quanto_correlation_key: str | None = "EURUSD_corr"
    day_count: DayCountConvention = DayCountConvention.ACT_365


def diagnose_metric_perturbation(
    *,
    label: str,
    perturbation_size: float,
    baseline_metrics: Mapping[str, float],
    perturbed_metrics: Mapping[str, float],
    relative_floor: float = 1.0e-12,
    instability_thresholds: Mapping[str, float] | None = None,
) -> CalibrationPerturbationDiagnostic:
    """Compare fitted metrics before and after a deterministic input perturbation."""
    baseline_keys = set(baseline_metrics)
    perturbed_keys = set(perturbed_metrics)
    if baseline_keys != perturbed_keys:
        missing = sorted(baseline_keys.symmetric_difference(perturbed_keys))
        raise ValueError(f"baseline and perturbed metrics must use the same keys: {missing}")
    thresholds = dict(instability_thresholds or {})
    absolute_changes: dict[str, float] = {}
    relative_changes: dict[str, float] = {}
    threshold_breaches: dict[str, float] = {}
    floor = max(float(relative_floor), 0.0)
    for key in sorted(baseline_keys):
        baseline_value = float(baseline_metrics[key])
        perturbed_value = float(perturbed_metrics[key])
        absolute_change = float(perturbed_value - baseline_value)
        relative_change = float(absolute_change / max(abs(baseline_value), floor))
        absolute_changes[key] = absolute_change
        relative_changes[key] = relative_change
        threshold = thresholds.get(key)
        if threshold is not None and abs(absolute_change) > float(threshold):
            threshold_breaches[key] = abs(absolute_change)
    return CalibrationPerturbationDiagnostic(
        label=label,
        perturbation_size=float(perturbation_size),
        baseline_metrics=baseline_metrics,
        perturbed_metrics=perturbed_metrics,
        absolute_changes=absolute_changes,
        relative_changes=relative_changes,
        threshold_breaches=threshold_breaches,
    )


def evaluate_latency_envelope(
    case: Mapping[str, object],
    envelope: CalibrationLatencyEnvelope,
) -> dict[str, object]:
    """Evaluate one benchmark case against its explicit latency envelope."""
    cold = dict(case["cold"])
    warm = case.get("warm")
    cold_mean_seconds = float(cold["mean_seconds"])
    cold_max_seconds = float(cold["max_seconds"])
    breaches: dict[str, float] = {}
    if cold_mean_seconds > float(envelope.cold_mean_limit_seconds):
        breaches["cold_mean_seconds"] = cold_mean_seconds
    if (
        envelope.cold_max_limit_seconds is not None
        and cold_max_seconds > float(envelope.cold_max_limit_seconds)
    ):
        breaches["cold_max_seconds"] = cold_max_seconds
    warm_mean_seconds = None
    if isinstance(warm, Mapping):
        warm_mean_seconds = float(warm["mean_seconds"])
        if (
            envelope.warm_mean_limit_seconds is not None
            and warm_mean_seconds > float(envelope.warm_mean_limit_seconds)
        ):
            breaches["warm_mean_seconds"] = warm_mean_seconds
    return {
        **envelope.to_dict(),
        "cold_mean_seconds": cold_mean_seconds,
        "cold_max_seconds": cold_max_seconds,
        "warm_mean_seconds": warm_mean_seconds,
        "breaches": breaches,
        "status": "fail" if breaches else "pass",
    }


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

    case = {
        "workflow": scenario.workflow,
        "label": scenario.label,
        "notes": list(scenario.notes),
        "metadata": dict(scenario.metadata),
        "cold": cold.to_dict(),
        "warm": None if warm is None else warm.to_dict(),
        "warm_speedup": None if warm_speedup is None else round(warm_speedup, 3),
    }
    latency_payload = scenario.metadata.get("latency_envelope")
    if isinstance(latency_payload, Mapping):
        envelope = CalibrationLatencyEnvelope(
            workflow=str(latency_payload.get("workflow") or scenario.workflow),
            label=str(latency_payload.get("label") or scenario.label),
            fixture_style=str(latency_payload.get("fixture_style") or scenario.metadata.get("fixture_style") or "synthetic"),
            instrument_count=latency_payload.get("instrument_count"),  # type: ignore[arg-type]
            quote_count=latency_payload.get("quote_count"),  # type: ignore[arg-type]
            cold_mean_limit_seconds=float(latency_payload["cold_mean_limit_seconds"]),
            cold_max_limit_seconds=(
                None
                if latency_payload.get("cold_max_limit_seconds") is None
                else float(latency_payload["cold_max_limit_seconds"])
            ),
            warm_mean_limit_seconds=(
                None
                if latency_payload.get("warm_mean_limit_seconds") is None
                else float(latency_payload["warm_mean_limit_seconds"])
            ),
        )
        case["latency_envelope"] = evaluate_latency_envelope(case, envelope)
    return case


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
            "desk_like_workflow_count": sum(
                1
                for case in cases
                if dict(case.get("metadata") or {}).get("fixture_style") == "desk_like"
            ),
            "perturbation_diagnostic_count": sum(
                1
                for case in cases
                if isinstance(dict(case.get("metadata") or {}).get("perturbation_diagnostic"), Mapping)
            ),
            "latency_envelope_count": sum(1 for case in cases if isinstance(case.get("latency_envelope"), Mapping)),
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
        f"- Desk-like workflows: `{summary.get('desk_like_workflow_count', 0)}`",
        f"- Perturbation diagnostics: `{summary.get('perturbation_diagnostic_count', 0)}`",
        f"- Latency envelopes: `{summary.get('latency_envelope_count', 0)}`",
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
        latency = case.get("latency_envelope")
        if isinstance(latency, Mapping):
            lines.append(
                "- Latency envelope: "
                f"`{latency['status']}` "
                f"(cold mean `{latency['cold_mean_seconds']}` s <= "
                f"`{latency['cold_mean_limit_seconds']}` s)"
            )
        perturbation = dict(case.get("metadata") or {}).get("perturbation_diagnostic")
        if isinstance(perturbation, Mapping):
            lines.append(
                "- Perturbation diagnostic: "
                f"`{perturbation.get('status')}` "
                f"(max abs change `{perturbation.get('max_abs_change')}`)"
            )
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
    from trellis.core.date_utils import add_months
    from trellis.core.market_state import MarketState
    from trellis.core.types import DayCountConvention, Frequency
    from trellis.curves.yield_curve import YieldCurve
    from trellis.data.mock import MockDataProvider
    from trellis.data.schema import MarketSnapshot
    from trellis.instruments.fx import FXRate
    from trellis.instruments._agent.swaption import SwaptionSpec
    from trellis.instruments.cap import CapFloorSpec, CapPayoff
    from trellis.models.bermudan_swaption_tree import BermudanSwaptionTreeSpec, price_bermudan_swaption_tree
    from trellis.models.calibration.basket_credit import (
        BasketCreditTrancheQuote,
        calibrate_homogeneous_basket_tranche_correlation_workflow,
    )
    from trellis.models.calibration.credit import (
        CreditHazardCalibrationQuote,
        calibrate_single_name_credit_curve_workflow,
    )
    from trellis.models.calibration.equity_vol_surface import calibrate_equity_vol_surface_workflow
    from trellis.models.calibration.heston_fit import (
        calibrate_heston_smile_workflow,
        calibrate_heston_surface_from_equity_vol_surface_workflow,
    )
    from trellis.models.calibration.local_vol import calibrate_local_vol_surface_workflow
    from trellis.models.calibration.materialization import materialize_black_vol_surface
    from trellis.models.calibration.quanto import (
        QuantoCorrelationCalibrationQuote,
        calibrate_quanto_correlation_workflow,
    )
    from trellis.models.calibration.rates import HullWhiteCalibrationInstrument, calibrate_hull_white, swaption_terms
    from trellis.models.calibration.rates_vol_surface import (
        CapletStripQuote,
        SwaptionCubeQuote,
        calibrate_caplet_vol_strip_workflow,
        calibrate_swaption_vol_cube_workflow,
    )
    from trellis.models.calibration.sabr_fit import calibrate_sabr_smile_workflow
    from trellis.models.credit_basket_copula import price_credit_basket_tranche_result
    from trellis.models.processes.sabr import SABRProcess
    from trellis.models.quanto_option import price_quanto_option_analytical_from_market_state
    from trellis.models.rate_style_swaption import price_swaption_black76
    from trellis.models.vol_surface import FlatVol, GridVolSurface

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
    hw_roles = {
        "discount_curve": dict(hw_state.selected_curve_names or {}).get("discount_curve"),
        "forecast_curve": dict(hw_state.selected_curve_names or {}).get("forecast_curve"),
        "rate_index": "USD-SOFR-3M",
    }
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

    caplet_state = market_state()
    caplet_surface = GridVolSurface(
        expiries=(0.2493150684931507, 0.4986301369863014, 0.7506849315068493, 1.0),
        strikes=(0.04, 0.05),
        vols=(
            (0.185, 0.195),
            (0.188, 0.198),
            (0.191, 0.201),
            (0.194, 0.204),
        ),
    )
    caplet_target_state = replace(caplet_state, vol_surface=caplet_surface)
    caplet_start_date = date(2025, 2, 15)
    caplet_end_dates = (
        date(2025, 5, 15),
        date(2025, 8, 15),
        date(2025, 11, 15),
        date(2026, 2, 15),
    )
    caplet_quotes = tuple(
        CapletStripQuote(
            spec=CapFloorSpec(
                notional=1_000_000.0,
                strike=float(strike),
                start_date=caplet_start_date,
                end_date=end_date,
                frequency=Frequency.QUARTERLY,
                day_count=DayCountConvention.ACT_360,
                rate_index="USD-SOFR-3M",
            ),
            quote=CapPayoff(
                CapFloorSpec(
                    notional=1_000_000.0,
                    strike=float(strike),
                    start_date=caplet_start_date,
                    end_date=end_date,
                    frequency=Frequency.QUARTERLY,
                    day_count=DayCountConvention.ACT_360,
                    rate_index="USD-SOFR-3M",
                )
            ).evaluate(caplet_target_state),
            quote_kind="price",
            kind="cap",
            label=f"cap_{float(strike):.4f}_{end_date.isoformat()}",
        )
        for strike in caplet_surface.strikes
        for end_date in caplet_end_dates
    )

    mock_snapshot = MockDataProvider().fetch_market_snapshot(settle)
    prior_parameters = dict(mock_snapshot.provenance.get("prior_parameters") or {})
    synthetic_generation_contract = dict(prior_parameters.get("synthetic_generation_contract") or {})
    model_consistency_contract = dict(prior_parameters.get("model_consistency_contract") or {})
    rates_contract = dict(model_consistency_contract.get("rates") or {})
    credit_contract = dict(model_consistency_contract.get("credit") or {})
    model_packs = dict(synthetic_generation_contract.get("model_packs") or {})
    rates_pack = dict(model_packs.get("rates") or {})
    volatility_pack = dict(model_packs.get("volatility") or {})
    volatility_quotes = dict((synthetic_generation_contract.get("quote_bundles") or {}).get("volatility") or {})
    curve_roles = dict(rates_contract.get("curve_roles") or {})
    discount_curve_name = str(curve_roles.get("discount_curve") or mock_snapshot.default_discount_curve or "")
    forecast_curve_name = str(curve_roles.get("forecast_curve") or mock_snapshot.default_forecast_curve or "")
    rate_vol_model = dict(rates_pack.get("rate_vol_model") or {})
    rate_surface_name = "usd_rates_smile"
    rate_surface = mock_snapshot.vol_surfaces[rate_surface_name]
    sabr_expiry = 1.0
    sabr_strikes = list(rate_surface.strikes)
    sabr_forward = float(mock_snapshot.forecast_curves[forecast_curve_name].zero_rate(sabr_expiry))
    sabr_beta = float(rate_vol_model.get("beta", 0.5))
    sabr_market_vols = [rate_surface.black_vol(sabr_expiry, strike) for strike in sabr_strikes]

    swaption_cube_state = market_state()
    swaption_cube_expiries = (date(2025, 11, 15), date(2026, 11, 15))
    swaption_cube_tenors = (5, 10)
    swaption_cube_strikes = (0.03, 0.04, 0.05)
    swaption_cube_quotes: list[SwaptionCubeQuote] = []
    cube_alpha = float(rate_vol_model.get("alpha", 0.03))
    cube_rho = float(rate_vol_model.get("rho", -0.25))
    cube_nu = float(rate_vol_model.get("nu", 0.35))
    cube_beta = float(rate_vol_model.get("beta", 0.5))
    for expiry_date in swaption_cube_expiries:
        for tenor_years in swaption_cube_tenors:
            tenor_scale = 1.0 + 0.08 * ((tenor_years - swaption_cube_tenors[0]) / 5.0)
            swap_end = date(expiry_date.year + tenor_years, expiry_date.month, expiry_date.day)
            atm_spec = SwaptionSpec(
                notional=5_000_000.0,
                strike=0.04,
                expiry_date=expiry_date,
                swap_start=expiry_date,
                swap_end=swap_end,
                swap_frequency=Frequency.SEMI_ANNUAL,
                day_count=DayCountConvention.ACT_360,
                rate_index="USD-SOFR-3M",
                is_payer=True,
            )
            expiry_years, _annuity, forward_swap_rate, _payment_count = swaption_terms(atm_spec, swaption_cube_state)
            sabr = SABRProcess(
                alpha=cube_alpha * tenor_scale,
                beta=cube_beta,
                rho=cube_rho - 0.05 * ((tenor_years - swaption_cube_tenors[0]) / 5.0),
                nu=cube_nu + 0.05 * ((tenor_years - swaption_cube_tenors[0]) / 5.0),
            )
            for strike in swaption_cube_strikes:
                spec = replace(atm_spec, strike=float(strike))
                market_vol = float(sabr.implied_vol(forward_swap_rate, float(strike), expiry_years))
                swaption_cube_quotes.append(
                    SwaptionCubeQuote(
                        spec=spec,
                        quote=price_swaption_black76(
                            replace(swaption_cube_state, vol_surface=FlatVol(market_vol)),
                            spec,
                        ),
                        quote_kind="price",
                        label=f"{expiry_date.isoformat()}_{tenor_years}Y_{float(strike):.4f}",
                    )
                )
    swaption_cube_quotes = tuple(swaption_cube_quotes)

    heston_surface_name = str(
        next(iter(volatility_quotes.get("implied_vol_surface_names") or ("spx_heston_implied_vol",)))
    )
    heston_surface = mock_snapshot.vol_surfaces[heston_surface_name]
    heston_model = dict((volatility_pack.get("model_parameter_sets") or {}).get("heston_equity") or {})
    heston_spot = float(mock_snapshot.underlier_spots["SPX"])
    heston_rate = float(mock_snapshot.discount_curves[discount_curve_name].zero_rate(1.0))
    heston_expiry = 1.0
    heston_strikes = list(heston_surface.strikes)
    heston_market_vols = [heston_surface.black_vol(heston_expiry, strike) for strike in heston_strikes]
    heston_surface_expiries = tuple(float(expiry) for expiry in heston_surface.expiries)
    heston_surface_strikes = tuple(float(strike) for strike in heston_surface.strikes)
    heston_surface_market_vols = tuple(
        tuple(float(vol) for vol in row)
        for row in heston_surface.vols
    )

    def equity_surface_authority():
        return calibrate_equity_vol_surface_workflow(
            heston_spot,
            heston_surface_expiries,
            heston_surface_strikes,
            heston_surface_market_vols,
            rate=heston_rate,
            surface_name="spx_surface_authority",
        )
    equity_surface_authority_result = equity_surface_authority()

    local_vol_sources = dict(volatility_quotes.get("local_vol_surface_sources") or {})
    local_vol_surface_name = str(next(iter(local_vol_sources or {"spx_local_vol": heston_surface_name})))
    local_vol_source_name = str(local_vol_sources.get(local_vol_surface_name, heston_surface_name))
    local_vol_source_surface = mock_snapshot.vol_surfaces[local_vol_source_name]
    local_vol_strikes = raw_np.asarray(local_vol_source_surface.strikes, dtype=float)
    local_vol_expiries = raw_np.asarray(local_vol_source_surface.expiries, dtype=float)
    local_vol_surface = raw_np.asarray(local_vol_source_surface.vols, dtype=float)

    spread_inputs = dict(credit_contract.get("spread_inputs_decimal") or {})
    credit_curve_name = str(mock_snapshot.default_credit_curve or "")
    if (not credit_curve_name or credit_curve_name not in spread_inputs) and spread_inputs:
        credit_curve_name = str(next(iter(spread_inputs)))
    spread_grid = dict(spread_inputs.get(credit_curve_name) or {})
    if not spread_grid:
        spread_grid = {"1.0": 0.012, "3.0": 0.014, "5.0": 0.016}
    credit_quotes = tuple(
        CreditHazardCalibrationQuote(
            maturity_years=float(tenor_text),
            quote=float(spread_quote),
            quote_kind="spread",
            label=f"{credit_curve_name}_{tenor_text}y",
        )
        for tenor_text, spread_quote in sorted(
            spread_grid.items(),
            key=lambda item: float(item[0]),
        )
    )
    credit_recovery = float(credit_contract.get("recovery", 0.4))
    credit_curve_name = credit_curve_name or "benchmark_credit_curve"
    credit_state = mock_snapshot.to_market_state(
        settlement=settle,
        discount_curve=discount_curve_name or None,
        forecast_curve=forecast_curve_name or None,
        credit_curve=credit_curve_name,
    )
    benchmark_credit_result = calibrate_single_name_credit_curve_workflow(
        credit_quotes,
        credit_state,
        recovery=credit_recovery,
        curve_name="benchmark_single_name_credit",
    )
    basket_credit_state = benchmark_credit_result.apply_to_market_state(credit_state)
    basket_notional = 100_000_000.0
    basket_n_names = 125
    basket_maturities = (5.0, 7.0)
    basket_tranches = ((0.0, 0.03), (0.03, 0.07), (0.07, 0.10))
    basket_correlations = {
        (5.0, 0.0, 0.03): 0.18,
        (5.0, 0.03, 0.07): 0.24,
        (5.0, 0.07, 0.10): 0.34,
        (7.0, 0.0, 0.03): 0.39,
        (7.0, 0.03, 0.07): 0.48,
        (7.0, 0.07, 0.10): 0.54,
    }

    def basket_quote(
        maturity_years: float,
        attachment: float,
        detachment: float,
        correlation: float,
    ) -> BasketCreditTrancheQuote:
        spec = _BenchmarkBasketTrancheSpec(
            notional=basket_notional,
            n_names=basket_n_names,
            attachment=attachment,
            detachment=detachment,
            end_date=add_months(settle, int(round(maturity_years * 12.0))),
            recovery=credit_recovery,
            correlation=correlation,
        )
        priced = price_credit_basket_tranche_result(basket_credit_state, spec)
        return BasketCreditTrancheQuote(
            maturity_years=maturity_years,
            attachment=attachment,
            detachment=detachment,
            quote_value=float(priced.expected_loss_fraction),
            quote_family="price",
            quote_style="expected_loss_fraction",
            label=f"{maturity_years:g}y_{attachment:.2f}_{detachment:.2f}",
        )

    basket_credit_quotes = tuple(
        basket_quote(
            maturity_years,
            attachment,
            detachment,
            basket_correlations[(maturity_years, attachment, detachment)],
        )
        for maturity_years in basket_maturities
        for attachment, detachment in basket_tranches
    )

    def basket_credit_calibration(quotes=basket_credit_quotes):
        return calibrate_homogeneous_basket_tranche_correlation_workflow(
            quotes,
            basket_credit_state,
            n_names=basket_n_names,
            recovery=credit_recovery,
            notional=basket_notional,
            surface_name="benchmark_tranche_correlation",
            smoothness_jump_threshold=0.35,
        )

    basket_credit_base_result = basket_credit_calibration()
    basket_perturbed_quotes = tuple(
        replace(quote, quote_value=float(quote.quote_value) * 1.0025)
        for quote in basket_credit_quotes
    )
    basket_credit_perturbed_result = basket_credit_calibration(basket_perturbed_quotes)
    basket_credit_diagnostic = diagnose_metric_perturbation(
        label="basket_credit_parallel_quote_up",
        perturbation_size=0.0025,
        baseline_metrics={
            point.quote_label: float(point.correlation)
            for point in basket_credit_base_result.surface.points
        },
        perturbed_metrics={
            point.quote_label: float(point.correlation)
            for point in basket_credit_perturbed_result.surface.points
        },
        instability_thresholds={
            point.quote_label: 0.08
            for point in basket_credit_base_result.surface.points
        },
    ).to_dict()
    basket_credit_latency_envelope = CalibrationLatencyEnvelope(
        workflow="basket_credit",
        label="desk_tranche_surface",
        fixture_style="desk_like",
        quote_count=len(basket_credit_quotes),
        cold_mean_limit_seconds=6.0,
        cold_max_limit_seconds=8.0,
    ).to_dict()

    quanto_truth_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(0.05),
        forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
        fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
        spot=100.0,
        underlier_spots={"EUR": 100.0},
        vol_surface=FlatVol(0.20),
        model_parameters={"EURUSD_corr": 0.35},
        selected_curve_names={
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
        },
        market_provenance={"source_kind": "benchmark_fixture", "source_ref": "quanto_validation_fixture"},
    )
    quanto_truth_state = materialize_black_vol_surface(
        quanto_truth_state,
        surface_name="quanto_flat_vol",
        vol_surface=FlatVol(0.20),
        source_kind="calibrated_surface",
        source_ref="calibrate_equity_vol_surface_workflow",
        selected_curve_roles={
            "discount_curve": "usd_ois",
            "forecast_curve": "EUR-DISC",
        },
        metadata={"instrument_family": "equity_fx_quanto"},
    )
    quanto_calibration_state = replace(
        quanto_truth_state,
        model_parameters={"EURUSD_corr": -0.10},
        model_parameter_sets=None,
    )
    quanto_specs = (
        _BenchmarkQuantoOptionSpec(
            notional=1_000_000.0,
            strike=95.0,
            expiry_date=date(2025, 11, 15),
            fx_pair="EURUSD",
        ),
        _BenchmarkQuantoOptionSpec(
            notional=1_000_000.0,
            strike=100.0,
            expiry_date=date(2026, 5, 15),
            fx_pair="EURUSD",
        ),
        _BenchmarkQuantoOptionSpec(
            notional=1_000_000.0,
            strike=105.0,
            expiry_date=date(2026, 11, 15),
            fx_pair="EURUSD",
        ),
    )
    quanto_quotes = tuple(
        QuantoCorrelationCalibrationQuote(
            market_price=price_quanto_option_analytical_from_market_state(quanto_truth_state, spec),
            notional=spec.notional,
            strike=spec.strike,
            expiry_date=spec.expiry_date,
            fx_pair=spec.fx_pair,
            underlier_currency=spec.underlier_currency,
            domestic_currency=spec.domestic_currency,
            option_type=spec.option_type,
            day_count=DayCountConvention.ACT_365,
            label=f"quanto_{index}",
            weight=1.0,
            quanto_correlation_key=spec.quanto_correlation_key,
        )
        for index, spec in enumerate(quanto_specs)
    )

    def quanto_correlation_calibration(
        quotes=quanto_quotes,
        *,
        initial_correlation: float = 0.0,
    ):
        return calibrate_quanto_correlation_workflow(
            quotes,
            quanto_calibration_state,
            parameter_set_name="benchmark_quanto_rho",
            initial_correlation=initial_correlation,
        )

    quanto_base_result = quanto_correlation_calibration()
    quanto_perturbed_quotes = tuple(
        replace(quote, market_price=float(quote.market_price) * 1.0025)
        for quote in quanto_quotes
    )
    quanto_perturbed_result = quanto_correlation_calibration(
        quanto_perturbed_quotes,
        initial_correlation=-0.20,
    )
    quanto_perturbation_diagnostic = diagnose_metric_perturbation(
        label="quanto_correlation_parallel_quote_up",
        perturbation_size=0.0025,
        baseline_metrics={"quanto_correlation": float(quanto_base_result.correlation)},
        perturbed_metrics={"quanto_correlation": float(quanto_perturbed_result.correlation)},
        instability_thresholds={"quanto_correlation": 0.08},
    ).to_dict()
    quanto_latency_envelope = CalibrationLatencyEnvelope(
        workflow="quanto_correlation",
        label="desk_quanto_correlation",
        fixture_style="desk_like",
        quote_count=len(quanto_quotes),
        cold_mean_limit_seconds=1.5,
        cold_max_limit_seconds=2.0,
        warm_mean_limit_seconds=0.5,
    ).to_dict()

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
            metadata={
                "instrument_count": len(hw_instruments),
                "warm_start": True,
                "multi_curve_roles": hw_roles,
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="caplet_strip",
            label="price_bootstrap_surface",
            cold_runner=lambda: calibrate_caplet_vol_strip_workflow(
                caplet_quotes,
                caplet_state,
                surface_name="usd_caplet_strip",
            ),
            notes=("bootstrap", "caplet_surface", "price_quotes"),
            metadata={
                "quote_count": len(caplet_quotes),
                "grid_shape": [len(caplet_surface.expiries), len(caplet_surface.strikes)],
                "warm_start": False,
                "surface_name": "usd_caplet_strip",
            },
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
                surface_name=rate_surface_name,
            ),
            warm_runner=lambda: calibrate_sabr_smile_workflow(
                sabr_forward,
                sabr_expiry,
                sabr_strikes,
                sabr_market_vols,
                beta=sabr_beta,
                surface_name=rate_surface_name,
                initial_guess=(
                    float(rate_vol_model.get("alpha", 0.20)),
                    float(rate_vol_model.get("rho", -0.3)),
                    float(rate_vol_model.get("nu", 0.4)),
                ),
            ),
            notes=("least_squares", "implied_vol_fit", "synthetic_generation_contract_fixture"),
            metadata={
                "point_count": len(sabr_strikes),
                "warm_start": True,
                "surface_name": rate_surface_name,
                "synthetic_generation_contract_version": str(synthetic_generation_contract.get("version", "")),
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="swaption_cube",
            label="price_normalized_cube",
            cold_runner=lambda: calibrate_swaption_vol_cube_workflow(
                swaption_cube_quotes,
                swaption_cube_state,
                surface_name="usd_swaption_cube",
            ),
            notes=("cube_assembly", "swaption_surface", "price_quotes", "synthetic_generation_contract_fixture"),
            metadata={
                "quote_count": len(swaption_cube_quotes),
                "grid_shape": [len(swaption_cube_expiries), len(swaption_cube_tenors), len(swaption_cube_strikes)],
                "warm_start": False,
                "surface_name": "usd_swaption_cube",
                "synthetic_generation_contract_version": str(synthetic_generation_contract.get("version", "")),
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="equity_vol_surface",
            label="repaired_surface_authority",
            cold_runner=lambda: equity_surface_authority(),
            notes=("svi_surface", "quote_governance", "synthetic_generation_contract_fixture"),
            metadata={
                "grid_shape": [len(heston_surface_expiries), len(heston_surface_strikes)],
                "warm_start": False,
                "surface_name": "spx_surface_authority",
                "synthetic_generation_contract_version": str(synthetic_generation_contract.get("version", "")),
            },
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
                surface_name=heston_surface_name,
                parameter_set_name="heston_equity",
            ),
            warm_runner=lambda: calibrate_heston_smile_workflow(
                heston_spot,
                heston_expiry,
                heston_strikes,
                heston_market_vols,
                rate=heston_rate,
                surface_name=heston_surface_name,
                parameter_set_name="heston_equity",
                warm_start=(
                    float(heston_model.get("kappa", 1.8)),
                    float(heston_model.get("theta", 0.04)),
                    float(heston_model.get("xi", 0.35)),
                    float(heston_model.get("rho", -0.6)),
                    float(heston_model.get("v0", 0.05)),
                ),
            ),
            notes=("least_squares", "fft_pricing", "synthetic_generation_contract_fixture"),
            metadata={
                "point_count": len(heston_strikes),
                "warm_start": True,
                "surface_name": heston_surface_name,
                "synthetic_generation_contract_version": str(synthetic_generation_contract.get("version", "")),
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="heston_surface",
            label="surface_compression",
            cold_runner=lambda: calibrate_heston_surface_from_equity_vol_surface_workflow(
                equity_surface_authority_result,
                parameter_set_name="heston_equity_surface",
            ),
            warm_runner=lambda: calibrate_heston_surface_from_equity_vol_surface_workflow(
                equity_surface_authority_result,
                parameter_set_name="heston_equity_surface",
                warm_start=(
                    float(heston_model.get("kappa", 1.8)),
                    float(heston_model.get("theta", 0.04)),
                    float(heston_model.get("xi", 0.35)),
                    float(heston_model.get("rho", -0.6)),
                    float(heston_model.get("v0", 0.05)),
                ),
            ),
            notes=("least_squares", "fft_pricing", "surface_compression", "synthetic_generation_contract_fixture"),
            metadata={
                "grid_shape": [len(heston_surface_expiries), len(heston_surface_strikes)],
                "warm_start": True,
                "surface_name": "spx_surface_authority",
                "synthetic_generation_contract_version": str(synthetic_generation_contract.get("version", "")),
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="local_vol",
            label="dupire_surface",
            cold_runner=lambda: calibrate_local_vol_surface_workflow(
                local_vol_strikes,
                local_vol_expiries,
                local_vol_surface,
                heston_spot,
                heston_rate,
                surface_name=local_vol_surface_name,
            ),
            notes=("dupire", "workflow_surface", "synthetic_generation_contract_fixture"),
            metadata={
                "grid_shape": [len(local_vol_expiries), len(local_vol_strikes)],
                "warm_start": False,
                "source_surface_name": local_vol_source_name,
                "surface_name": local_vol_surface_name,
                "synthetic_generation_contract_version": str(synthetic_generation_contract.get("version", "")),
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="credit",
            label="single_name_curve",
            cold_runner=lambda: calibrate_single_name_credit_curve_workflow(
                credit_quotes,
                credit_state,
                recovery=credit_recovery,
                curve_name="benchmark_single_name_credit",
            ),
            notes=("least_squares", "model_consistency_contract_fixture"),
            metadata={
                "point_count": len(credit_quotes),
                "quote_family": "spread",
                "warm_start": False,
                "model_consistency_contract_version": str(model_consistency_contract.get("version", "")),
                "curve_name": credit_curve_name,
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="basket_credit",
            label="desk_tranche_surface",
            cold_runner=lambda: basket_credit_calibration(),
            notes=(
                "brentq_root_scan",
                "homogeneous_basket_credit",
                "desk_like_fixture",
                "linked_single_name_credit_curve",
            ),
            metadata={
                "fixture_style": "desk_like",
                "quote_count": len(basket_credit_quotes),
                "tranche_count": len(basket_tranches),
                "maturity_count": len(basket_maturities),
                "warm_start": False,
                "surface_name": "benchmark_tranche_correlation",
                "linked_credit_curve": "benchmark_single_name_credit",
                "support_boundary": "homogeneous_representative_curve",
                "perturbation_diagnostic": basket_credit_diagnostic,
                "latency_envelope": basket_credit_latency_envelope,
            },
        ),
        CalibrationBenchmarkScenario(
            workflow="quanto_correlation",
            label="desk_quanto_correlation",
            cold_runner=lambda: quanto_correlation_calibration(initial_correlation=0.0),
            warm_runner=lambda: quanto_correlation_calibration(initial_correlation=0.35),
            notes=(
                "least_squares",
                "desk_like_fixture",
                "bounded_quanto_correlation",
                "linked_market_state_materialization",
            ),
            metadata={
                "fixture_style": "desk_like",
                "quote_count": len(quanto_quotes),
                "warm_start": True,
                "parameter_set_name": "benchmark_quanto_rho",
                "fx_pair": "EURUSD",
                "correlation_keys": ["EURUSD_corr"],
                "support_boundary": "bounded_quanto_correlation",
                "linked_vol_surface": "quanto_flat_vol",
                "linked_curve_roles": {
                    "discount_curve": "usd_ois",
                    "forecast_curve": "EUR-DISC",
                },
                "perturbation_diagnostic": quanto_perturbation_diagnostic,
                "latency_envelope": quanto_latency_envelope,
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
    "CalibrationLatencyEnvelope",
    "CalibrationPerturbationDiagnostic",
    "benchmark_calibration_workflow",
    "benchmark_calibration_scenario",
    "build_calibration_benchmark_report",
    "diagnose_metric_perturbation",
    "evaluate_latency_envelope",
    "render_calibration_benchmark_report",
    "save_calibration_benchmark_report",
    "build_supported_calibration_benchmark_report",
    "supported_calibration_benchmark_scenarios",
]
