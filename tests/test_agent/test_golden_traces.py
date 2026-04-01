"""Tests for golden trace snapshots and drift detection (QUA-426).

All tests are pure unit tests — no LLM calls, no tokens spent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trellis.agent.checkpoints import DecisionCheckpoint, StageDecision
from trellis.agent.golden_traces import (
    DriftSummary,
    TaskDriftReport,
    detect_drift,
    detect_drift_for_canary,
    format_drift_report,
    list_golden,
    load_golden,
    save_golden,
    update_golden_from_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cp(
    task_id: str = "T38",
    instrument: str = "callable_bond",
    quant_decision: str = "rate_tree",
    builder_hash: str = "abc123",
    price: float = 98.234,
    tolerance: float = 0.5,
    outcome: str = "pass",
) -> DecisionCheckpoint:
    return DecisionCheckpoint(
        task_id=task_id,
        instrument_type=instrument,
        timestamp="2026-03-30T12:00:00+00:00",
        stages=(
            StageDecision(agent="quant", decision=quant_decision, metadata={"method_modules": ["bdt"]}),
            StageDecision(agent="planner", decision="CallableBondSpec", metadata={"field_count": 3}),
            StageDecision(agent="builder", decision="compiled", metadata={"code_lines": 47}, output_hash=builder_hash),
            StageDecision(agent="validator", decision="pass", metadata={"final_price": price, "tolerance": tolerance}),
        ),
        outcome=outcome,
        total_tokens=5000,
        final_price=price,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# Golden I/O
# ---------------------------------------------------------------------------

class TestGoldenIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        cp = _cp()
        path = save_golden(cp, directory=tmp_path)
        assert path.exists()
        assert path.name == "T38.yaml"

        loaded = load_golden("T38", directory=tmp_path)
        assert loaded is not None
        assert loaded.task_id == "T38"
        assert loaded.outcome == "pass"
        assert len(loaded.stages) == 4
        assert loaded.stages[0].decision == "rate_tree"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_golden("T99", directory=tmp_path) is None

    def test_save_overwrites_existing(self, tmp_path):
        cp1 = _cp(price=98.0)
        cp2 = _cp(price=99.0)
        save_golden(cp1, directory=tmp_path)
        save_golden(cp2, directory=tmp_path)

        loaded = load_golden("T38", directory=tmp_path)
        assert loaded.final_price == 99.0

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "deep" / "golden"
        save_golden(_cp(), directory=nested)
        assert (nested / "T38.yaml").exists()

    def test_list_golden(self, tmp_path):
        save_golden(_cp(task_id="T38"), directory=tmp_path)
        save_golden(_cp(task_id="T13"), directory=tmp_path)
        save_golden(_cp(task_id="T01"), directory=tmp_path)

        ids = list_golden(directory=tmp_path)
        assert ids == ["T01", "T13", "T38"]

    def test_list_golden_empty_dir(self, tmp_path):
        assert list_golden(directory=tmp_path) == []

    def test_list_golden_nonexistent_dir(self, tmp_path):
        assert list_golden(directory=tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# detect_drift
# ---------------------------------------------------------------------------

class TestDetectDrift:
    def test_no_golden_baseline(self):
        current = _cp()
        report = detect_drift(None, current, engine_family="lattice")
        assert not report.has_golden
        assert report.task_id == "T38"
        assert report.engine_family == "lattice"
        assert len(report.divergences) == 0
        assert not report.is_stable  # no golden = not stable

    def test_identical_no_drift(self, tmp_path):
        golden = _cp()
        current = _cp()
        report = detect_drift(golden, current)
        assert report.has_golden
        assert report.is_stable
        assert report.max_severity == "stable"
        assert len(report.divergences) == 0

    def test_decision_drift(self):
        golden = _cp(quant_decision="rate_tree")
        current = _cp(quant_decision="monte_carlo")
        report = detect_drift(golden, current, engine_family="credit")
        assert report.has_decision_drift
        assert report.max_severity == "decision"
        assert not report.is_stable
        decision_divs = [d for d in report.divergences if d.severity == "decision"]
        assert len(decision_divs) >= 1
        assert decision_divs[0].agent == "quant"

    def test_metadata_drift(self):
        golden = _cp(builder_hash="hash_old")
        current = _cp(builder_hash="hash_new")
        report = detect_drift(golden, current)
        assert report.has_metadata_drift
        assert not report.has_decision_drift
        assert report.max_severity == "metadata"

    def test_price_drift(self):
        golden = _cp(price=98.234, tolerance=0.5)
        current = _cp(price=97.9, tolerance=0.5)  # drift ratio = 0.668 > 0.5
        report = detect_drift(golden, current)
        assert report.has_price_drift
        assert report.max_severity == "price"

    def test_no_price_drift_within_tolerance(self):
        golden = _cp(price=98.234, tolerance=0.5)
        current = _cp(price=98.1, tolerance=0.5)  # drift ratio = 0.268 < 0.5
        report = detect_drift(golden, current)
        assert not report.has_price_drift

    def test_detect_drift_for_canary(self, tmp_path):
        golden = _cp()
        save_golden(golden, directory=tmp_path)

        current = _cp(quant_decision="monte_carlo")
        report = detect_drift_for_canary(
            "T38", current, engine_family="credit", golden_dir=tmp_path,
        )
        assert report.has_golden
        assert report.has_decision_drift

    def test_detect_drift_for_canary_no_golden(self, tmp_path):
        current = _cp()
        report = detect_drift_for_canary(
            "T99", current, engine_family="pde", golden_dir=tmp_path,
        )
        assert not report.has_golden


# ---------------------------------------------------------------------------
# TaskDriftReport properties
# ---------------------------------------------------------------------------

class TestTaskDriftReport:
    def test_severity_ordering(self):
        from trellis.agent.checkpoints import StageDivergence

        # decision > price > metadata
        r_decision = TaskDriftReport(
            task_id="T1", engine_family="x", has_golden=True,
            divergences=(StageDivergence(agent="quant", old_decision="a", new_decision="b", severity="decision"),),
        )
        assert r_decision.max_severity == "decision"

        r_price = TaskDriftReport(
            task_id="T1", engine_family="x", has_golden=True,
            divergences=(StageDivergence(agent="val", old_decision="p=1", new_decision="p=2", severity="price"),),
        )
        assert r_price.max_severity == "price"

        r_meta = TaskDriftReport(
            task_id="T1", engine_family="x", has_golden=True,
            divergences=(StageDivergence(agent="builder", old_decision="c", new_decision="c", severity="metadata"),),
        )
        assert r_meta.max_severity == "metadata"

        r_stable = TaskDriftReport(task_id="T1", engine_family="x", has_golden=True)
        assert r_stable.max_severity == "stable"


# ---------------------------------------------------------------------------
# DriftSummary
# ---------------------------------------------------------------------------

class TestDriftSummary:
    def _make_summary(self) -> DriftSummary:
        from trellis.agent.checkpoints import StageDivergence

        return DriftSummary(reports=(
            TaskDriftReport(task_id="T01", engine_family="lattice", has_golden=True),  # stable
            TaskDriftReport(task_id="T38", engine_family="credit", has_golden=True,
                divergences=(StageDivergence(agent="quant", old_decision="a", new_decision="b", severity="decision"),)),
            TaskDriftReport(task_id="T13", engine_family="pde", has_golden=True,
                divergences=(StageDivergence(agent="builder", old_decision="c", new_decision="c", severity="metadata"),)),
            TaskDriftReport(task_id="T99", engine_family="copula", has_golden=False),
        ))

    def test_counts(self):
        s = self._make_summary()
        assert s.stable_count == 1
        assert s.decision_drift_count == 1
        assert s.metadata_drift_count == 1
        assert s.no_golden_count == 1

    def test_has_blocking_drift(self):
        s = self._make_summary()
        assert s.has_blocking_drift

    def test_no_blocking_drift(self):
        s = DriftSummary(reports=(
            TaskDriftReport(task_id="T01", engine_family="lattice", has_golden=True),
        ))
        assert not s.has_blocking_drift


# ---------------------------------------------------------------------------
# update_golden_from_results
# ---------------------------------------------------------------------------

class TestUpdateGolden:
    def test_update_on_all_pass(self, tmp_path):
        cp38 = _cp(task_id="T38")
        cp13 = _cp(task_id="T13")
        results = [
            {"canary_id": "T38", "success": True},
            {"canary_id": "T13", "success": True},
        ]
        updated = update_golden_from_results(
            results, {"T38": cp38, "T13": cp13}, directory=tmp_path,
        )
        assert set(updated) == {"T38", "T13"}
        assert load_golden("T38", directory=tmp_path) is not None
        assert load_golden("T13", directory=tmp_path) is not None

    def test_no_update_on_any_failure(self, tmp_path):
        results = [
            {"canary_id": "T38", "success": True},
            {"canary_id": "T13", "success": False},
        ]
        updated = update_golden_from_results(
            results, {"T38": _cp(task_id="T38")}, directory=tmp_path,
        )
        assert updated == []
        assert load_golden("T38", directory=tmp_path) is None

    def test_update_ignores_skipped(self, tmp_path):
        results = [
            {"canary_id": "T38", "success": True},
            {"canary_id": "T99", "skipped": True, "success": False},
        ]
        updated = update_golden_from_results(
            results, {"T38": _cp(task_id="T38")}, directory=tmp_path,
        )
        assert updated == ["T38"]

    def test_update_without_require_all_pass(self, tmp_path):
        results = [
            {"canary_id": "T38", "success": True},
            {"canary_id": "T13", "success": False},
        ]
        updated = update_golden_from_results(
            results, {"T38": _cp(task_id="T38"), "T13": _cp(task_id="T13")},
            directory=tmp_path,
            require_all_pass=False,
        )
        assert updated == ["T38"]  # T38 passes, T13 fails — only T38 updated


# ---------------------------------------------------------------------------
# format_drift_report
# ---------------------------------------------------------------------------

class TestFormatDriftReport:
    def test_empty_report(self):
        out = format_drift_report([])
        assert "No drift reports" in out

    def test_stable_report(self):
        report = TaskDriftReport(task_id="T38", engine_family="credit", has_golden=True)
        out = format_drift_report([report])
        assert "T38" in out
        assert "no drift" in out
        assert "DRIFT REPORT" in out

    def test_decision_drift_report(self):
        from trellis.agent.checkpoints import StageDivergence

        report = TaskDriftReport(
            task_id="T38", engine_family="credit", has_golden=True,
            divergences=(StageDivergence(
                agent="quant", old_decision="rate_tree", new_decision="monte_carlo", severity="decision",
            ),),
        )
        out = format_drift_report([report])
        assert "DECISION DRIFT" in out
        assert "quant" in out
        assert "rate_tree" in out
        assert "monte_carlo" in out

    def test_no_golden_report(self):
        report = TaskDriftReport(task_id="T99", engine_family="copula", has_golden=False)
        out = format_drift_report([report])
        assert "no golden baseline" in out

    def test_summary_line(self):
        from trellis.agent.checkpoints import StageDivergence

        reports = [
            TaskDriftReport(task_id="T01", engine_family="lattice", has_golden=True),
            TaskDriftReport(task_id="T38", engine_family="credit", has_golden=True,
                divergences=(StageDivergence(agent="q", old_decision="a", new_decision="b", severity="decision"),)),
        ]
        out = format_drift_report(reports)
        assert "1 stable" in out
        assert "1 DECISION DRIFT" in out

    def test_format_accepts_drift_summary(self):
        summary = DriftSummary(reports=(
            TaskDriftReport(task_id="T01", engine_family="lattice", has_golden=True),
        ))
        out = format_drift_report(summary)
        assert "T01" in out
        assert "1 stable" in out
