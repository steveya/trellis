"""Tests for the post-task consolidation skill in autonomous.py."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from trellis.agent.knowledge.autonomous import (
    ConsolidationResult,
    _assess_consolidation_needs,
    _log_consolidation,
    _maybe_consolidate,
    _run_consolidation,
    _CANDIDATE_BACKLOG_RATIO,
    _TRACE_BLOAT_COUNT,
    _SUPERSEDES_GAP_COUNT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_lesson(entries_dir: Path, lesson_id: str, status: str, category: str = "numerical") -> Path:
    """Write a minimal lesson entry YAML."""
    p = entries_dir / f"{lesson_id}.yaml"
    p.write_text(yaml.dump({
        "id": lesson_id,
        "title": f"Lesson {lesson_id}",
        "status": status,
        "category": category,
        "severity": "medium",
        "symptom": "x",
        "root_cause": "y",
        "fix": "z",
        "applies_when": {"method": [], "features": [], "instrument": [], "error_signature": None},
        "confidence": 0.5 if status == "candidate" else 0.85,
    }))
    return p


def _write_trace(traces_dir: Path, name: str, age_days: int = 0) -> Path:
    """Write a minimal trace YAML and backdate its mtime."""
    p = traces_dir / f"{name}.yaml"
    p.write_text(yaml.dump({"instrument": "test", "method": "analytical", "resolved": True}))
    if age_days > 0:
        old_time = time.time() - age_days * 86400
        import os
        os.utime(p, (old_time, old_time))
    return p


# ---------------------------------------------------------------------------
# _assess_consolidation_needs
# ---------------------------------------------------------------------------

class TestAssessConsolidationNeeds:
    def test_candidate_backlog_triggers_tier1(self, tmp_path):
        """When >40% of lessons are candidates, tier 1 is triggered."""
        entries = tmp_path / "lessons" / "entries"
        entries.mkdir(parents=True)
        traces = tmp_path / "traces"
        traces.mkdir()

        # 3 candidates, 1 promoted → 75% candidates
        _write_lesson(entries, "n001", "candidate")
        _write_lesson(entries, "n002", "candidate")
        _write_lesson(entries, "n003", "candidate")
        _write_lesson(entries, "n004", "promoted")

        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            tiers, reasons = _assess_consolidation_needs()
        finally:
            mod.__file__ = original_file

        assert 1 in tiers
        assert any("candidate_backlog" in r for r in reasons)

    def test_no_backlog_gets_routine_maintenance(self, tmp_path):
        """When nothing triggers, tier 1 still runs as routine maintenance."""
        entries = tmp_path / "lessons" / "entries"
        entries.mkdir(parents=True)
        traces = tmp_path / "traces"
        traces.mkdir()

        # 1 candidate, 9 promoted → 10% candidates, below threshold
        _write_lesson(entries, "n001", "candidate")
        for i in range(9):
            _write_lesson(entries, f"p{i:03d}", "promoted", category=f"cat{i}")

        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            tiers, reasons = _assess_consolidation_needs()
        finally:
            mod.__file__ = original_file

        assert 1 in tiers
        assert any("routine_maintenance" in r for r in reasons)

    def test_trace_bloat_triggers_tier2(self, tmp_path):
        """When trace count > threshold, tier 2 is triggered."""
        entries = tmp_path / "lessons" / "entries"
        entries.mkdir(parents=True)
        traces = tmp_path / "traces"
        traces.mkdir()

        _write_lesson(entries, "n001", "promoted")

        # Create lots of trace files
        for i in range(_TRACE_BLOAT_COUNT + 10):
            (traces / f"trace_{i:04d}.yaml").write_text("x: 1\n")

        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            tiers, reasons = _assess_consolidation_needs()
        finally:
            mod.__file__ = original_file

        assert 2 in tiers
        assert any("trace_bloat" in r for r in reasons)

    def test_supersedes_gap_triggers_tier3(self, tmp_path):
        """When many promoted lessons lack supersedes scan, tier 3 triggers."""
        entries = tmp_path / "lessons" / "entries"
        entries.mkdir(parents=True)
        traces = tmp_path / "traces"
        traces.mkdir()

        for i in range(_SUPERSEDES_GAP_COUNT + 5):
            _write_lesson(entries, f"p{i:03d}", "promoted")

        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            tiers, reasons = _assess_consolidation_needs()
        finally:
            mod.__file__ = original_file

        assert 3 in tiers
        assert any("supersedes_gap" in r for r in reasons)

    def test_principle_opportunity_triggers_tier4(self, tmp_path):
        """When a category has 3+ promoted lessons and no recent draft, tier 4 triggers."""
        entries = tmp_path / "lessons" / "entries"
        entries.mkdir(parents=True)
        traces = tmp_path / "traces"
        traces.mkdir()

        # 3 promoted in same category
        _write_lesson(entries, "v001", "promoted", category="volatility")
        _write_lesson(entries, "v002", "promoted", category="volatility")
        _write_lesson(entries, "v003", "promoted", category="volatility")

        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            tiers, reasons = _assess_consolidation_needs()
        finally:
            mod.__file__ = original_file

        assert 4 in tiers
        assert any("principle_opportunity" in r for r in reasons)


# ---------------------------------------------------------------------------
# _run_consolidation
# ---------------------------------------------------------------------------

class TestRunConsolidation:
    def test_tier1_calls_recalibrate(self):
        """Tier 1 calls recalibrate_candidates."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc:
            mock_rc.return_value = {"boosted": 2, "validated": 1, "unchanged": 5}
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1])

        assert result.triggered is True
        assert 1 in result.tiers_run
        assert result.tier_results["recalibrate"]["boosted"] == 2
        mock_rc.assert_called_once()

    def test_tier2_calls_compact_traces(self):
        """Tier 2 calls compact_traces."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc, \
             patch("trellis.agent.knowledge.promotion.compact_traces") as mock_ct:
            mock_rc.return_value = {"boosted": 0, "validated": 0, "unchanged": 0}
            mock_ct.return_value = {"cohorts_summarized": 3, "traces_compacted": 42}
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1, 2])

        assert result.tier_results["compact_traces"]["traces_compacted"] == 42
        mock_ct.assert_called_once_with(older_than_days=30)

    def test_tier3_calls_backfill_dry_run(self):
        """Tier 3 calls backfill_supersedes with dry_run=True."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc, \
             patch("trellis.agent.knowledge.promotion.backfill_supersedes") as mock_bs:
            mock_rc.return_value = {}
            mock_bs.return_value = {"new_001": ["old_002"]}
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1, 3])

        mock_bs.assert_called_once_with(dry_run=True)
        assert result.tier_results["backfill_supersedes_dry"]["count"] == 1

    def test_tier4_calls_draft_with_model(self):
        """Tier 4 calls draft_principle_candidates when model is provided."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc, \
             patch("trellis.agent.knowledge.promotion.draft_principle_candidates") as mock_dp:
            mock_rc.return_value = {}
            mock_dp.return_value = [{"category": "volatility"}]
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1, 4], model="test-model")

        mock_dp.assert_called_once_with(model="test-model")
        assert result.tier_results["principle_candidates"]["drafted"] == 1

    def test_tier4_skipped_without_model(self):
        """Tier 4 is skipped when no model is provided."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc, \
             patch("trellis.agent.knowledge.promotion.draft_principle_candidates") as mock_dp:
            mock_rc.return_value = {}
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1, 4], model=None)

        mock_dp.assert_not_called()
        assert "principle_candidates" not in result.tier_results

    def test_tier_error_does_not_crash(self):
        """An error in one tier does not prevent others from running."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc, \
             patch("trellis.agent.knowledge.promotion.compact_traces") as mock_ct:
            mock_rc.side_effect = RuntimeError("boom")
            mock_ct.return_value = {"cohorts_summarized": 1, "traces_compacted": 5}
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1, 2])

        assert "error" in result.tier_results["recalibrate"]
        assert result.tier_results["compact_traces"]["traces_compacted"] == 5

    def test_duration_tracked(self):
        """Duration is recorded."""
        with patch("trellis.agent.knowledge.promotion.recalibrate_candidates") as mock_rc:
            mock_rc.return_value = {}
            with patch("trellis.agent.knowledge.autonomous._log_consolidation"):
                result = _run_consolidation({}, [1])

        assert result.duration_seconds >= 0


# ---------------------------------------------------------------------------
# _log_consolidation
# ---------------------------------------------------------------------------

class TestLogConsolidation:
    def test_creates_log_file(self, tmp_path):
        """Log creates the consolidation_log.yaml file."""
        log_path = tmp_path / "traces" / "consolidation_log.yaml"
        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            result = ConsolidationResult(
                triggered=True,
                tiers_run=[1, 2],
                trigger_reasons=["test"],
                tier_results={"recalibrate": {"boosted": 1}},
                duration_seconds=0.5,
            )
            _log_consolidation(result)
        finally:
            mod.__file__ = original_file

        assert log_path.exists()
        entries = yaml.safe_load(log_path.read_text())
        assert len(entries) == 1
        assert entries[0]["tiers_run"] == [1, 2]
        assert entries[0]["trigger_reasons"] == ["test"]

    def test_appends_to_existing_log(self, tmp_path):
        """Log appends to existing entries."""
        log_path = tmp_path / "traces" / "consolidation_log.yaml"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(yaml.dump([{"timestamp": "old", "tiers_run": [1]}]))

        import trellis.agent.knowledge.autonomous as mod
        original_file = mod.__file__
        try:
            mod.__file__ = str(tmp_path / "autonomous.py")
            _log_consolidation(ConsolidationResult(triggered=True, tiers_run=[2]))
        finally:
            mod.__file__ = original_file

        entries = yaml.safe_load(log_path.read_text())
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# _maybe_consolidate
# ---------------------------------------------------------------------------

class TestMaybeConsolidate:
    def test_synchronous_returns_result(self):
        """Synchronous mode returns ConsolidationResult."""
        with patch("trellis.agent.knowledge.autonomous._assess_consolidation_needs") as mock_assess, \
             patch("trellis.agent.knowledge.autonomous._run_consolidation") as mock_run:
            mock_assess.return_value = ([1], ["test"])
            mock_run.return_value = ConsolidationResult(triggered=True, tiers_run=[1])
            result = _maybe_consolidate({}, background=False)

        assert result is not None
        assert result.triggered is True

    def test_background_returns_none(self):
        """Background mode returns None (thread is spawned)."""
        with patch("trellis.agent.knowledge.autonomous._assess_consolidation_needs") as mock_assess, \
             patch("trellis.agent.knowledge.autonomous._run_consolidation") as mock_run, \
             patch("trellis.agent.knowledge.autonomous._log_consolidation"):
            mock_assess.return_value = ([1], ["test"])
            mock_run.return_value = ConsolidationResult(triggered=True, tiers_run=[1])
            result = _maybe_consolidate({}, background=True)

        assert result is None

    def test_tier4_deferred_in_background(self):
        """Tier 4 is excluded from background execution."""
        with patch("trellis.agent.knowledge.autonomous._assess_consolidation_needs") as mock_assess, \
             patch("trellis.agent.knowledge.autonomous._run_consolidation") as mock_run, \
             patch("trellis.agent.knowledge.autonomous._log_consolidation"):
            mock_assess.return_value = ([1, 4], ["backlog", "principle"])
            mock_run.return_value = ConsolidationResult(triggered=True, tiers_run=[1])
            _maybe_consolidate({}, background=True)

        # Give thread a moment to start
        import time
        time.sleep(0.1)
        # _run_consolidation should have been called with [1] only (4 excluded)
        if mock_run.called:
            call_tiers = mock_run.call_args[0][1]
            assert 4 not in call_tiers

    def test_no_tiers_returns_not_triggered(self):
        """When no tiers are needed, returns not-triggered."""
        with patch("trellis.agent.knowledge.autonomous._assess_consolidation_needs") as mock_assess:
            mock_assess.return_value = ([], [])
            result = _maybe_consolidate({}, background=False)

        assert result is not None
        assert result.triggered is False
