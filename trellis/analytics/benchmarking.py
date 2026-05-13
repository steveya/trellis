"""Pod-risk throughput benchmarks and persisted report helpers."""

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


DEFAULT_REPORT_ROOT = Path("docs") / "benchmarks"
DEFAULT_REPORT_STEM = "pod_risk_workflows"
DEFAULT_PORTFOLIO_AAD_REPORT_ROOT = Path("benchmark_runs") / "portfolio_aad"
DEFAULT_PORTFOLIO_AAD_REPORT_STEM = "portfolio_aad_workflows"
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


@dataclass(frozen=True)
class PortfolioAADBenchmarkScenario:
    """One reproducible portfolio-AAD benchmark scenario."""

    case_id: str
    label: str
    lane_mix: tuple[str, ...]
    book_size: int
    factor_count: int
    aad_runner: Callable[[], object] = field(repr=False, compare=False)
    baseline_runner: Callable[[], object] = field(repr=False, compare=False)
    baseline_name: str = "bump_reprice"
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "lane_mix", tuple(str(lane) for lane in self.lane_mix))
        object.__setattr__(self, "book_size", int(self.book_size))
        object.__setattr__(self, "factor_count", int(self.factor_count))
        object.__setattr__(self, "baseline_name", str(self.baseline_name))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


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
    report_title: str = "Pod Risk Benchmark",
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
        "report_title": str(report_title or "Pod Risk Benchmark"),
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
        f"# {report.get('report_title', 'Pod Risk Benchmark')}: `{report['benchmark_name']}`",
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
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_risk_benchmark_report(payload))
    return RiskBenchmarkArtifacts(report=payload, json_path=json_path, text_path=text_path)


def benchmark_portfolio_aad_scenario(
    scenario: PortfolioAADBenchmarkScenario,
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark one supported portfolio-AAD scenario against bump/reprice."""
    aad_measurement, aad_result = _benchmark_runner_with_result(
        label=scenario.label,
        mode="portfolio_aad_vjp",
        runner=scenario.aad_runner,
        repeats=repeats,
        warmups=warmups,
        timer=timer,
        notes=scenario.notes,
        metadata=scenario.metadata,
    )
    baseline_measurement, baseline_result = _benchmark_runner_with_result(
        label=scenario.label,
        mode=scenario.baseline_name,
        runner=scenario.baseline_runner,
        repeats=repeats,
        warmups=warmups,
        timer=timer,
        notes=scenario.notes,
        metadata=scenario.metadata,
    )
    aad_summary = _portfolio_aad_output_summary(aad_result)
    baseline_output_size = _risk_output_size(baseline_result)
    relative_speedup = (
        float(baseline_measurement.mean_seconds / aad_measurement.mean_seconds)
        if aad_measurement.mean_seconds > 0.0
        else float("inf")
    )
    aad_payload = aad_measurement.to_dict()
    baseline_payload = baseline_measurement.to_dict()
    factor_count = int(aad_summary.get("factor_count") or scenario.factor_count)
    return {
        "case_id": scenario.case_id,
        "label": scenario.label,
        "lane_mix": list(scenario.lane_mix),
        "book_size": scenario.book_size,
        "factor_count": factor_count,
        "expected_factor_count": scenario.factor_count,
        "risk_vector_size": int(aad_summary.get("risk_vector_size", 0)),
        "baseline_output_size": baseline_output_size,
        "aad_support_status": aad_summary.get("support_status"),
        "unsupported_position_count": int(
            aad_summary.get("unsupported_position_count", 0)
        ),
        "aad_elapsed_seconds": aad_payload["mean_seconds"],
        "baseline_elapsed_seconds": baseline_payload["mean_seconds"],
        "relative_speedup": round(relative_speedup, 3),
        "aad": aad_payload,
        "baseline": baseline_payload,
        "notes": list(scenario.notes),
        "metadata": dict(scenario.metadata),
    }


def build_portfolio_aad_benchmark_report(
    *,
    benchmark_name: str,
    cases: Sequence[Mapping[str, object]],
    notes: Sequence[str] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble a portfolio-AAD benchmark report payload."""
    cases = [dict(case) for case in cases]
    speedups = [
        float(case["relative_speedup"])
        for case in cases
        if case.get("relative_speedup") is not None
    ]
    aad_means = [float(case["aad_elapsed_seconds"]) for case in cases]
    baseline_means = [float(case["baseline_elapsed_seconds"]) for case in cases]
    return {
        "benchmark_name": benchmark_name,
        "report_title": "Portfolio AAD Benchmark",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": dict(environment or _default_environment()),
        "summary": {
            "case_count": len(cases),
            "total_book_size": sum(int(case["book_size"]) for case in cases),
            "total_factor_count": sum(int(case["factor_count"]) for case in cases),
            "average_aad_seconds": round(float(raw_np.mean(aad_means)), 6)
            if aad_means
            else 0.0,
            "average_baseline_seconds": round(float(raw_np.mean(baseline_means)), 6)
            if baseline_means
            else 0.0,
            "average_relative_speedup": round(float(raw_np.mean(speedups)), 3)
            if speedups
            else None,
        },
        "cases": cases,
        "notes": list(notes or ()),
    }


def build_supported_portfolio_aad_benchmark_report(
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark the supported bounded portfolio-AAD lanes."""
    cases = [
        benchmark_portfolio_aad_scenario(
            scenario,
            repeats=repeats,
            warmups=warmups,
            timer=timer,
        )
        for scenario in supported_portfolio_aad_benchmark_scenarios()
    ]
    return build_portfolio_aad_benchmark_report(
        benchmark_name=DEFAULT_PORTFOLIO_AAD_REPORT_STEM,
        cases=cases,
        notes=(
            "AAD timings use the bounded public portfolio-AAD entrypoints.",
            "Baseline timings use deterministic one-sided bump/reprice loops over the same factor families.",
            "Small one-factor fixtures may show traced overhead rather than speedup; the gate records evidence instead of enforcing wall-clock thresholds.",
            "The report is local evidence for supported lanes only, not a backend-wide portfolio-AAD capability claim.",
        ),
    )


def render_portfolio_aad_benchmark_report(report: Mapping[str, object]) -> str:
    """Render a Markdown portfolio-AAD benchmark report."""
    summary = report["summary"]
    environment = report["environment"]
    lines = [
        f"# Portfolio AAD Benchmark: `{report['benchmark_name']}`",
        f"- Created at: `{report.get('created_at', '')}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Total book size: `{summary['total_book_size']}`",
        f"- Total factor count: `{summary['total_factor_count']}`",
        f"- Avg AAD mean seconds: `{summary['average_aad_seconds']}`",
        f"- Avg baseline mean seconds: `{summary['average_baseline_seconds']}`",
    ]
    if summary.get("average_relative_speedup") is not None:
        lines.append(f"- Avg relative speedup: `{summary['average_relative_speedup']}`x")

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

    lines.extend(["", "## Case Results"])
    for case in report["cases"]:
        lines.extend(
            [
                "",
                f"### `{case['case_id']}` {case['label']}",
                f"- Lane mix: `{', '.join(case['lane_mix'])}`",
                f"- Book size: `{case['book_size']}`",
                f"- Factor count: `{case['factor_count']}`",
                f"- Risk vector size: `{case['risk_vector_size']}`",
                f"- AAD mean: `{case['aad_elapsed_seconds']}` s",
                f"- Baseline mean: `{case['baseline_elapsed_seconds']}` s",
                f"- Relative speedup: `{case['relative_speedup']}`x",
                f"- Support status: `{case['aad_support_status']}`",
                f"- Unsupported positions: `{case['unsupported_position_count']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def save_portfolio_aad_benchmark_report(
    report: Mapping[str, object],
    *,
    root: Path | None = None,
    stem: str = DEFAULT_PORTFOLIO_AAD_REPORT_STEM,
) -> RiskBenchmarkArtifacts:
    """Persist a local portfolio-AAD benchmark report as JSON plus Markdown."""
    root = (root or DEFAULT_PORTFOLIO_AAD_REPORT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    text_path = root / f"{stem}.md"
    payload = dict(report)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    text_path.write_text(render_portfolio_aad_benchmark_report(payload))
    return RiskBenchmarkArtifacts(report=payload, json_path=json_path, text_path=text_path)


def supported_portfolio_aad_benchmark_scenarios() -> tuple[PortfolioAADBenchmarkScenario, ...]:
    """Return deterministic fixtures for supported bounded portfolio-AAD lanes."""
    from trellis.analytics.portfolio_aad import (
        BondCurveAADMarketContext,
        VanillaEquityOptionVolAADMarketContext,
    )
    from trellis.book import (
        Book,
        portfolio_aad_curve_risk,
        portfolio_aad_equity_option_vol_risk,
        portfolio_aad_supported_book_risk,
    )
    from trellis.core.market_state import MarketState
    from trellis.curves.yield_curve import YieldCurve
    from trellis.instruments.bond import Bond
    from trellis.models.vol_surface import FlatVol, GridVolSurface

    @dataclass(frozen=True)
    class _BenchmarkVanillaOption:
        spot: float
        strike: float
        expiry_date: date
        option_type: str = "call"
        notional: float = 1.0
        exercise_style: str = "european"

    curve = YieldCurve(
        [1.0, 2.0, 5.0, 10.0, 30.0],
        [0.04, 0.041, 0.043, 0.046, 0.048],
    )
    bond_maturities = (
        (3, 2027),
        (5, 2029),
        (7, 2031),
        (10, 2034),
        (15, 2039),
        (20, 2044),
        (25, 2049),
        (30, 2054),
    )
    bond_book = Book(
        {
            f"bond_{index}": Bond(
                face=100.0,
                coupon=0.035 + 0.001 * index,
                maturity_date=date(maturity_year, 11, 15),
                maturity=maturity,
                frequency=2,
            )
            for index, (maturity, maturity_year) in enumerate(bond_maturities)
        },
        notionals={f"bond_{index}": 500_000.0 + 25_000.0 * index for index in range(8)},
    )
    flat_market = MarketState(
        as_of=BENCHMARK_SETTLE,
        settlement=BENCHMARK_SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(0.20),
    )
    grid_market = MarketState(
        as_of=BENCHMARK_SETTLE,
        settlement=BENCHMARK_SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=GridVolSurface(
            expiries=(0.5, 1.5),
            strikes=(90.0, 110.0),
            vols=((0.18, 0.21), (0.24, 0.27)),
        ),
    )
    flat_context = VanillaEquityOptionVolAADMarketContext(
        market_state=flat_market,
        vol_surface_name="spx_flat",
        currency="USD",
    )
    grid_context = VanillaEquityOptionVolAADMarketContext(
        market_state=grid_market,
        vol_surface_name="spx_grid",
        currency="USD",
    )
    flat_option_book = _vanilla_option_benchmark_book(_BenchmarkVanillaOption, count=12)
    grid_option_book = _vanilla_option_benchmark_book(_BenchmarkVanillaOption, count=8)
    mixed_bond_names = ("bond_0", "bond_1", "bond_3", "bond_7")
    mixed_option_names = tuple(list(flat_option_book.names)[:6])
    mixed_book = Book(
        {
            **{name: bond_book[name] for name in mixed_bond_names},
            **{name: flat_option_book[name] for name in mixed_option_names},
        },
        notionals={
            **{name: bond_book.notional(name) for name in mixed_bond_names},
            **{name: flat_option_book.notional(name) for name in mixed_option_names},
        },
    )
    mixed_bond_book = Book(
        {name: mixed_book[name] for name in mixed_bond_names},
        notionals={name: mixed_book.notional(name) for name in mixed_bond_names},
    )
    mixed_option_book = Book(
        {name: mixed_book[name] for name in mixed_option_names},
        notionals={
            name: mixed_book.notional(name)
            for name in mixed_option_names
        },
    )
    bond_context = BondCurveAADMarketContext(
        curve=curve,
        settlement=BENCHMARK_SETTLE,
        curve_name="usd_ois",
        currency="USD",
    )

    return (
        PortfolioAADBenchmarkScenario(
            case_id="bond_curve_nodes",
            label="shared_curve_bond_book",
            lane_mix=("bond_curve",),
            book_size=len(bond_book),
            factor_count=len(bond_context.coordinates()),
            aad_runner=lambda: portfolio_aad_curve_risk(
                bond_book,
                curve,
                BENCHMARK_SETTLE,
            ),
            baseline_runner=lambda: _bond_curve_bump_reprice_vector(
                bond_book,
                curve,
                BENCHMARK_SETTLE,
                curve_name="shared_curve",
                currency=None,
            ),
            notes=("bond_book", "curve_node_bumps"),
            metadata={"baseline": "one_sided_curve_node_bump"},
        ),
        PortfolioAADBenchmarkScenario(
            case_id="flat_vol_options",
            label="shared_flat_vol_option_book",
            lane_mix=("equity_option_flat_vol",),
            book_size=len(flat_option_book),
            factor_count=len(flat_context.coordinates()),
            aad_runner=lambda: portfolio_aad_equity_option_vol_risk(
                flat_option_book,
                flat_context,
            ),
            baseline_runner=lambda: _vanilla_option_vol_bump_reprice_vector(
                flat_option_book,
                flat_context,
            ),
            notes=("vanilla_option", "flat_vol_bump"),
            metadata={"baseline": "one_sided_flat_vol_bump"},
        ),
        PortfolioAADBenchmarkScenario(
            case_id="grid_vol_options",
            label="shared_grid_vol_option_book",
            lane_mix=("equity_option_grid_vol",),
            book_size=len(grid_option_book),
            factor_count=len(grid_context.coordinates()),
            aad_runner=lambda: portfolio_aad_equity_option_vol_risk(
                grid_option_book,
                grid_context,
            ),
            baseline_runner=lambda: _vanilla_option_vol_bump_reprice_vector(
                grid_option_book,
                grid_context,
            ),
            notes=("vanilla_option", "grid_node_bumps"),
            metadata={"baseline": "one_sided_grid_node_bump"},
        ),
        PortfolioAADBenchmarkScenario(
            case_id="mixed_supported_book",
            label="bond_and_flat_vol_option_book",
            lane_mix=("bond_curve", "equity_option_flat_vol"),
            book_size=len(mixed_book),
            factor_count=len(bond_context.coordinates()) + len(flat_context.coordinates()),
            aad_runner=lambda: portfolio_aad_supported_book_risk(
                mixed_book,
                bond_curve_context=bond_context,
                equity_option_vol_context=flat_context,
            ),
            baseline_runner=lambda: (
                _bond_curve_bump_reprice_vector(
                    mixed_bond_book,
                    curve,
                    BENCHMARK_SETTLE,
                    curve_name="usd_ois",
                    currency="USD",
                )
                + _vanilla_option_vol_bump_reprice_vector(
                    mixed_option_book,
                    flat_context,
                )
            ),
            notes=("mixed_supported_book", "curve_and_flat_vol_bumps"),
            metadata={"baseline": "one_sided_curve_and_flat_vol_bumps"},
        ),
    )


def _benchmark_runner_with_result(
    *,
    label: str,
    mode: str,
    runner: Callable[[], object],
    repeats: int,
    warmups: int,
    timer: Callable[[], float] | None,
    notes: Sequence[str],
    metadata: Mapping[str, object],
) -> tuple[RiskBenchmarkMeasurement, object]:
    """Time a runner and retain the last measured result."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")

    timer = timer or time.perf_counter
    for _ in range(warmups):
        runner()

    run_seconds: list[float] = []
    last_result: object | None = None
    for _ in range(repeats):
        start = float(timer())
        last_result = runner()
        stop = float(timer())
        run_seconds.append(stop - start)

    samples = raw_np.asarray(run_seconds, dtype=float)
    mean_seconds = float(raw_np.mean(samples))
    measurement = RiskBenchmarkMeasurement(
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
        notes=tuple(notes),
        metadata=metadata,
    )
    return measurement, last_result


def _portfolio_aad_output_summary(result: object) -> dict[str, object]:
    """Return stable output-size metadata for a portfolio-AAD result object."""
    from trellis.analytics.portfolio_aad import PortfolioAADResult

    aad_result: PortfolioAADResult | None = None
    if isinstance(result, PortfolioAADResult):
        aad_result = result
    else:
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping) and "portfolio_aad_result" in metadata:
            aad_result = PortfolioAADResult.from_payload(metadata["portfolio_aad_result"])

    if aad_result is None:
        return {
            "factor_count": 0,
            "risk_vector_size": _risk_output_size(result),
            "support_status": None,
            "unsupported_position_count": 0,
        }
    return {
        "factor_count": len(aad_result.coordinates),
        "risk_vector_size": len(aad_result.risk_vector),
        "support_status": aad_result.support_status,
        "unsupported_position_count": len(aad_result.unsupported_positions),
    }


def _risk_output_size(result: object) -> int:
    """Return a best-effort output length for benchmark payloads."""
    if result is None:
        return 0
    try:
        return len(result)  # type: ignore[arg-type]
    except TypeError:
        return 1


def _vanilla_option_benchmark_book(option_cls: type, *, count: int):
    """Build a deterministic vanilla-option book for benchmark fixtures."""
    from trellis.book import Book

    strikes = (90.0, 95.0, 100.0, 105.0, 110.0)
    option_types = ("call", "put")
    instruments = {}
    notionals = {}
    for index in range(count):
        name = f"option_{index}"
        instruments[name] = option_cls(
            spot=100.0,
            strike=strikes[index % len(strikes)],
            expiry_date=date(2025 + (index % 2), 11, 15),
            option_type=option_types[index % len(option_types)],
        )
        notionals[name] = 100.0 + 10.0 * index
    return Book(instruments, notionals=notionals)


def _bond_curve_bump_reprice_vector(
    book: object,
    curve: object,
    settlement: date,
    *,
    curve_name: str,
    currency: str | None,
    bump_size: float = 1.0e-4,
):
    """Return one-sided bump/reprice curve risk for a bond book."""
    from trellis.analytics.portfolio_aad import BondCurveAADMarketContext
    from trellis.analytics.risk_factors import SparseRiskVector

    tenors = raw_np.asarray(getattr(curve, "tenors"), dtype=float)
    rates = raw_np.asarray(getattr(curve, "rates"), dtype=float)
    curve_cls = type(curve)
    coordinates = BondCurveAADMarketContext(
        curve=curve,
        settlement=settlement,
        curve_name=curve_name,
        currency=currency,
    ).coordinates()
    base_value = _bond_book_value(book, curve, settlement)
    items = []
    for index, coordinate in enumerate(coordinates):
        bumped_rates = raw_np.array(rates, copy=True)
        bumped_rates[index] += bump_size
        bumped_curve = curve_cls(tenors, bumped_rates)
        sensitivity = (
            _bond_book_value(book, bumped_curve, settlement) - base_value
        ) / bump_size
        items.append((coordinate.factor_id, sensitivity))
    return SparseRiskVector.from_items(items)


def _bond_book_value(book: object, curve: object, settlement: date) -> float:
    """Return notional-weighted value for the bond positions in a book."""
    total = 0.0
    for name in book:
        total += float(book.notional(name)) * float(book[name].price(curve, settlement))
    return total


def _vanilla_option_vol_bump_reprice_vector(
    book: object,
    context: object,
    *,
    bump_size: float = 1.0e-4,
):
    """Return one-sided bump/reprice vol risk for a vanilla option book."""
    from trellis.analytics.portfolio_aad import VanillaEquityOptionVolAADMarketContext
    from trellis.analytics.risk_factors import SparseRiskVector
    from trellis.models.vol_surface import FlatVol, GridVolSurface

    if not isinstance(context, VanillaEquityOptionVolAADMarketContext):
        raise TypeError("vanilla option vol benchmark requires a vol AAD context")

    vol_surface = getattr(context.market_state, "vol_surface", None)
    coordinates = context.coordinates()
    base_value = _vanilla_option_book_value(book, context)
    if isinstance(vol_surface, FlatVol):
        bumped_context = _replace_vol_context(
            context,
            FlatVol(float(vol_surface.vol) + bump_size),
        )
        sensitivity = (
            _vanilla_option_book_value(book, bumped_context) - base_value
        ) / bump_size
        return SparseRiskVector.from_items(((coordinates[0].factor_id, sensitivity),))

    if isinstance(vol_surface, GridVolSurface):
        base_nodes = raw_np.asarray(vol_surface.vols, dtype=float)
        items = []
        for flat_index, coordinate in enumerate(coordinates):
            row, col = raw_np.unravel_index(flat_index, base_nodes.shape)
            bumped_nodes = raw_np.array(base_nodes, copy=True)
            bumped_nodes[row, col] += bump_size
            bumped_surface = GridVolSurface(
                tuple(float(expiry) for expiry in vol_surface.expiries),
                tuple(float(strike) for strike in vol_surface.strikes),
                tuple(tuple(float(value) for value in node_row) for node_row in bumped_nodes),
            )
            bumped_context = _replace_vol_context(context, bumped_surface)
            sensitivity = (
                _vanilla_option_book_value(book, bumped_context) - base_value
            ) / bump_size
            items.append((coordinate.factor_id, sensitivity))
        return SparseRiskVector.from_items(items)

    raise ValueError("unsupported benchmark vol surface parameterization")


def _replace_vol_context(context: object, vol_surface: object):
    """Return a copy of a vol AAD context with a replaced vol surface."""
    return replace(
        context,
        market_state=replace(context.market_state, vol_surface=vol_surface),
    )


def _vanilla_option_book_value(book: object, context: object) -> float:
    """Return notional-weighted vanilla option value under one context."""
    from trellis.analytics.portfolio_aad import (
        PortfolioAADRequest,
        VanillaEquityOptionVolAADAdapter,
    )

    adapter = VanillaEquityOptionVolAADAdapter()
    request = PortfolioAADRequest()
    total = 0.0
    for name in book:
        instrument = book[name]
        decision = adapter.support_decision(name, instrument, context, request)
        if not decision.supported:
            continue
        total += float(book.notional(name)) * adapter.value(instrument, context, request)
    return total


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
    from trellis.book import portfolio_aad_curve_risk
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

    def benchmark_book_krd(result, book: Book) -> dict[float, float]:
        total_mv = result.total_mv
        agg: dict[float, float] = {}
        for name in result:
            pricing_result = result[name]
            mv = pricing_result.dirty_price * book.notional(name)
            weight = mv / total_mv if total_mv else 0.0
            for tenor, krd in pricing_result.greeks.get("key_rate_durations", {}).items():
                tenor_key = float(tenor)
                agg[tenor_key] = agg.get(tenor_key, 0.0) + float(krd) * weight
        return agg

    def cold_portfolio_aad_fallback() -> dict[float, float]:
        book = benchmark_book()
        session = Session(
            curve=benchmark_curve(),
            settlement=BENCHMARK_SETTLE,
        )
        return benchmark_book_krd(
            session.price(book, greeks="all"),
            book,
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
    warm_portfolio_aad_book = benchmark_book()
    warm_portfolio_aad_curve = benchmark_curve()
    warm_portfolio_aad_session = Session(
        curve=warm_portfolio_aad_curve,
        settlement=BENCHMARK_SETTLE,
    )
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
            workflow="portfolio_aad",
            label="bond_book_reverse_mode",
            cold_runner=cold_portfolio_aad_fallback,
            steady_runner=lambda: portfolio_aad_curve_risk(
                warm_portfolio_aad_book,
                warm_portfolio_aad_curve,
                BENCHMARK_SETTLE,
            ),
            notes=("bond_book", "reverse_mode", "supported_curve"),
            metadata={
                "curve_nodes": 5,
                "position_count": 2,
                "supported_route": "bond_only",
            },
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


def build_supported_counterparty_exposure_benchmark_report(
    *,
    repeats: int = 3,
    warmups: int = 1,
    timer: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Benchmark supported institutional exposure workflows."""
    cases = [
        benchmark_risk_scenario(
            scenario,
            repeats=repeats,
            warmups=warmups,
            timer=timer,
        )
        for scenario in supported_counterparty_exposure_benchmark_scenarios()
    ]
    return build_risk_benchmark_report(
        benchmark_name="counterparty_exposure_workflows",
        cases=cases,
        report_title="Counterparty Exposure Benchmark",
        notes=(
            "Cold runs rebuild the future-value cube and exposure stack; steady-state runs reuse upstream artifacts where the workflow allows it.",
            "Coverage is limited to supported vanilla IRS future-value cubes, bounded collateral projection, netting-set exposure inputs, and EE/EPE/PFE metrics.",
        ),
    )


def supported_counterparty_exposure_benchmark_scenarios() -> tuple[RiskBenchmarkScenario, ...]:
    """Return checked benchmark fixtures for institutional exposure workflows."""
    from trellis.analytics.counterparty import (
        CollateralAgreement,
        NettingSet,
        aggregate_netting_set_exposures,
        compute_exposure_metrics,
        project_collateral_state,
    )
    from trellis.core.market_state import MarketState
    from trellis.core.types import DayCountConvention, Frequency
    from trellis.curves.yield_curve import YieldCurve
    from trellis.instruments.swap import SwapPayoff, SwapSpec
    from trellis.models.monte_carlo.simulation_substrate import (
        price_interest_rate_swap_portfolio_future_value_cube,
    )
    from trellis.models.vol_surface import FlatVol

    settle = BENCHMARK_SETTLE
    n_paths = 96
    n_steps = 36

    def market_state() -> MarketState:
        return MarketState(
            as_of=settle,
            settlement=settle,
            discount=YieldCurve.flat(0.041, max_tenor=10.0),
            forecast_curves={"USD-SOFR-3M": YieldCurve.flat(0.045, max_tenor=10.0)},
            vol_surface=FlatVol(0.18),
            selected_curve_names={
                "discount_curve": "usd_ois",
                "forecast_curve": "USD-SOFR-3M",
            },
        )

    def payer_swap() -> SwapSpec:
        return SwapSpec(
            notional=1_000_000.0,
            fixed_rate=0.043,
            start_date=settle,
            end_date=date(2027, 11, 15),
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
            fixed_day_count=DayCountConvention.THIRTY_360,
            float_day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            is_payer=True,
        )

    def receiver_swap() -> SwapSpec:
        return SwapSpec(
            notional=750_000.0,
            fixed_rate=0.039,
            start_date=settle,
            end_date=date(2026, 11, 15),
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
            fixed_day_count=DayCountConvention.THIRTY_360,
            float_day_count=DayCountConvention.ACT_360,
            rate_index="USD-SOFR-3M",
            is_payer=False,
        )

    def future_value_cube():
        return price_interest_rate_swap_portfolio_future_value_cube(
            positions={
                "payer_swap": SwapPayoff(payer_swap()),
                "receiver_swap": SwapPayoff(receiver_swap()),
            },
            market_state=market_state(),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=642,
            mean_reversion=0.10,
            sigma=0.0,
        )

    agreement = CollateralAgreement(
        agreement_id="csa_benchmark",
        collateral_currency="USD",
        threshold=50_000.0,
        minimum_transfer_amount=10_000.0,
        margin_period_of_risk_days=14,
        valuation_lag_days=2,
    )
    netting_set = NettingSet(
        netting_set_id="ns_benchmark",
        counterparty_id="bank_benchmark",
        position_names=("payer_swap", "receiver_swap"),
        collateral_agreement_id=agreement.agreement_id,
        exposure_currency="USD",
    )

    def exposure_metrics_from_cube(cube):
        projection = project_collateral_state(
            cube,
            netting_set=netting_set,
            collateral_agreement=agreement,
        )
        exposure_cube = aggregate_netting_set_exposures(
            cube,
            netting_sets=(netting_set,),
            collateral_projections={netting_set.netting_set_id: projection},
        )
        return compute_exposure_metrics(exposure_cube, pfe_levels=(0.95, 0.99))

    warm_cube = future_value_cube()

    return (
        RiskBenchmarkScenario(
            workflow="swap_portfolio_future_value_cube",
            label="hull_white_shared_path_irs_book",
            cold_runner=future_value_cube,
            steady_runner=future_value_cube,
            notes=("institutional_exposure", "future_value_cube", "shared_paths"),
            metadata={
                "position_count": 2,
                "n_paths": n_paths,
                "n_steps": n_steps,
                "process_family": "hull_white_1f",
            },
        ),
        RiskBenchmarkScenario(
            workflow="counterparty_exposure_metrics",
            label="netting_collateral_ee_epe_pfe",
            cold_runner=lambda: exposure_metrics_from_cube(future_value_cube()),
            steady_runner=lambda: exposure_metrics_from_cube(warm_cube),
            notes=("institutional_exposure", "netting", "collateral", "ee_epe_pfe"),
            metadata={
                "position_count": 2,
                "netting_set_count": 1,
                "pfe_levels": [0.95, 0.99],
                "warm_start": "reuse_future_value_cube",
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
    "RiskBenchmarkMeasurement",
    "RiskBenchmarkScenario",
    "RiskBenchmarkArtifacts",
    "benchmark_risk_workflow",
    "benchmark_risk_scenario",
    "build_risk_benchmark_report",
    "render_risk_benchmark_report",
    "save_risk_benchmark_report",
    "build_supported_pod_risk_benchmark_report",
    "build_supported_counterparty_exposure_benchmark_report",
    "supported_pod_risk_benchmark_scenarios",
    "supported_counterparty_exposure_benchmark_scenarios",
]
