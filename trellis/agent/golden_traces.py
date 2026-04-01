"""Golden trace snapshots and drift detection (QUA-426).

Golden traces are committed-to-git YAML files representing the last known-good
decision checkpoint for each canary task.  After each canary run the current
checkpoint is compared against the golden snapshot to detect drift.

Three severity levels:
  - **decision** — agent made a different choice (e.g. method family changed)
  - **metadata** — same decision, different details (e.g. code line count)
  - **price**    — same decisions, price drifted toward tolerance boundary

Usage::

    from trellis.agent.golden_traces import (
        load_golden, save_golden, detect_drift, format_drift_report,
    )

    golden = load_golden("T38")
    current = capture_checkpoint(...)
    report = detect_drift(golden, current)
    print(format_drift_report([report]))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trellis.agent.checkpoints import (
    DecisionCheckpoint,
    StageDivergence,
    _checkpoint_to_dict,
    _dict_to_checkpoint,
    diff_checkpoints,
)

_log = logging.getLogger(__name__)

# Golden traces live in the repo root so they can be committed to git.
GOLDEN_DIR = Path(__file__).parent.parent.parent / "golden_traces"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskDriftReport:
    """Drift report for a single canary task."""

    task_id: str
    engine_family: str
    has_golden: bool
    divergences: tuple[StageDivergence, ...] = ()
    current: DecisionCheckpoint | None = None
    golden: DecisionCheckpoint | None = None

    @property
    def has_decision_drift(self) -> bool:
        return any(d.severity == "decision" for d in self.divergences)

    @property
    def has_metadata_drift(self) -> bool:
        return any(d.severity == "metadata" for d in self.divergences)

    @property
    def has_price_drift(self) -> bool:
        return any(d.severity == "price" for d in self.divergences)

    @property
    def is_stable(self) -> bool:
        return self.has_golden and len(self.divergences) == 0

    @property
    def max_severity(self) -> str:
        """Return the worst severity: decision > price > metadata > stable."""
        if self.has_decision_drift:
            return "decision"
        if self.has_price_drift:
            return "price"
        if self.has_metadata_drift:
            return "metadata"
        return "stable"


@dataclass(frozen=True)
class DriftSummary:
    """Aggregate drift report across all canary tasks."""

    reports: tuple[TaskDriftReport, ...]

    @property
    def stable_count(self) -> int:
        return sum(1 for r in self.reports if r.is_stable)

    @property
    def decision_drift_count(self) -> int:
        return sum(1 for r in self.reports if r.has_decision_drift)

    @property
    def metadata_drift_count(self) -> int:
        return sum(1 for r in self.reports if r.has_metadata_drift and not r.has_decision_drift)

    @property
    def price_drift_count(self) -> int:
        return sum(1 for r in self.reports if r.has_price_drift and not r.has_decision_drift)

    @property
    def no_golden_count(self) -> int:
        return sum(1 for r in self.reports if not r.has_golden)

    @property
    def has_blocking_drift(self) -> bool:
        """True if any task has decision-level drift (should block release)."""
        return self.decision_drift_count > 0


# ---------------------------------------------------------------------------
# Golden snapshot I/O
# ---------------------------------------------------------------------------

def load_golden(
    task_id: str,
    *,
    directory: Path | None = None,
) -> DecisionCheckpoint | None:
    """Load the golden snapshot for *task_id*, or None if not found."""
    base = directory or GOLDEN_DIR
    path = base / f"{task_id}.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return _dict_to_checkpoint(data)
    except Exception as exc:
        _log.warning("Failed to load golden trace %s: %s", path, exc)
        return None


def save_golden(
    checkpoint: DecisionCheckpoint,
    *,
    directory: Path | None = None,
) -> Path:
    """Save a checkpoint as the golden snapshot for its task_id.

    Overwrites any existing golden for the same task.
    Returns the path written.
    """
    base = directory or GOLDEN_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{checkpoint.task_id}.yaml"
    data = _checkpoint_to_dict(checkpoint)
    path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _log.info("Golden trace saved: %s", path)
    return path


def list_golden(*, directory: Path | None = None) -> list[str]:
    """Return task IDs that have golden snapshots."""
    base = directory or GOLDEN_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_drift(
    golden: DecisionCheckpoint | None,
    current: DecisionCheckpoint,
    *,
    engine_family: str = "",
) -> TaskDriftReport:
    """Compare *current* checkpoint against *golden* and return a drift report."""
    task_id = current.task_id

    if golden is None:
        return TaskDriftReport(
            task_id=task_id,
            engine_family=engine_family,
            has_golden=False,
            current=current,
        )

    divergences = diff_checkpoints(golden, current)
    return TaskDriftReport(
        task_id=task_id,
        engine_family=engine_family,
        has_golden=True,
        divergences=tuple(divergences),
        current=current,
        golden=golden,
    )


def detect_drift_for_canary(
    task_id: str,
    current: DecisionCheckpoint,
    *,
    engine_family: str = "",
    golden_dir: Path | None = None,
) -> TaskDriftReport:
    """Load golden for *task_id* and compare against *current*."""
    golden = load_golden(task_id, directory=golden_dir)
    return detect_drift(golden, current, engine_family=engine_family)


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def update_golden_from_results(
    results: list[dict[str, Any]],
    checkpoints: dict[str, DecisionCheckpoint],
    *,
    directory: Path | None = None,
    require_all_pass: bool = True,
) -> list[str]:
    """Promote current checkpoints to golden after a successful canary run.

    Args:
        results: canary run result dicts (must have 'success' and 'canary_id')
        checkpoints: mapping of task_id -> DecisionCheckpoint
        directory: golden trace directory (default: repo root golden_traces/)
        require_all_pass: if True, only update if ALL canaries passed

    Returns:
        List of task IDs that were updated.
    """
    if require_all_pass:
        non_skipped = [r for r in results if not r.get("skipped")]
        if not all(r.get("success") for r in non_skipped):
            failed = [r.get("canary_id", "?") for r in non_skipped if not r.get("success")]
            _log.warning(
                "Cannot update golden: %d canaries failed (%s). "
                "All canaries must pass to update golden traces.",
                len(failed), ", ".join(failed),
            )
            return []

    updated = []
    for task_id, checkpoint in checkpoints.items():
        matching = [r for r in results if r.get("canary_id") == task_id]
        if matching and matching[0].get("success"):
            save_golden(checkpoint, directory=directory)
            updated.append(task_id)

    return updated


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_drift_report(
    reports: list[TaskDriftReport] | DriftSummary,
    *,
    show_details: bool = True,
) -> str:
    """Format drift reports into a terminal-friendly report."""
    if isinstance(reports, DriftSummary):
        task_reports = list(reports.reports)
    else:
        task_reports = list(reports)

    if not task_reports:
        return "No drift reports."

    lines = [
        "\u2550" * 65,
        f"  CANARY DRIFT REPORT \u2014 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "\u2550" * 65,
    ]

    for r in task_reports:
        severity = r.max_severity
        icon = {
            "stable": "\u2713",
            "metadata": "\u26a0",
            "price": "\u26a0",
            "decision": "\u2717",
        }.get(severity, "?")

        label = {
            "stable": "no drift",
            "metadata": "metadata drift",
            "price": "price drift",
            "decision": "DECISION DRIFT",
        }.get(severity, "no golden" if not r.has_golden else "unknown")

        if not r.has_golden:
            icon = "\u2022"
            label = "no golden baseline"

        line = f"  {icon} {r.task_id:6s}  {r.engine_family:14s}  {label}"
        lines.append(line)

        if show_details and r.divergences:
            for d in r.divergences:
                lines.append(
                    f"         {d.agent}: {d.old_decision} \u2192 {d.new_decision} [{d.severity}]"
                )

    lines.append("\u2500" * 65)

    # Summary line
    summary = DriftSummary(reports=tuple(task_reports))
    parts = []
    if summary.stable_count:
        parts.append(f"{summary.stable_count} stable")
    if summary.no_golden_count:
        parts.append(f"{summary.no_golden_count} no baseline")
    if summary.metadata_drift_count:
        parts.append(f"{summary.metadata_drift_count} metadata drift")
    if summary.price_drift_count:
        parts.append(f"{summary.price_drift_count} price drift")
    if summary.decision_drift_count:
        parts.append(f"{summary.decision_drift_count} DECISION DRIFT")

    lines.append(f"  {' | '.join(parts)}")
    lines.append("\u2550" * 65)

    return "\n".join(lines)
