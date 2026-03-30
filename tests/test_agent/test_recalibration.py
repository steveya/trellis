"""Tests for recalibrate_candidates() in the promotion pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest


@pytest.fixture()
def knowledge_tree(tmp_path, monkeypatch):
    """Build a minimal knowledge directory tree for recalibration tests."""
    lessons_dir = tmp_path / "lessons"
    entries_dir = lessons_dir / "entries"
    entries_dir.mkdir(parents=True)
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(parents=True)
    index_path = lessons_dir / "index.yaml"

    # Patch module-level paths
    import trellis.agent.knowledge.promotion as promo

    monkeypatch.setattr(promo, "_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(promo, "_LESSONS_DIR", lessons_dir)
    monkeypatch.setattr(promo, "_TRACES_DIR", traces_dir)
    monkeypatch.setattr(promo, "_INDEX_PATH", index_path)

    return {
        "tmp_path": tmp_path,
        "lessons_dir": lessons_dir,
        "entries_dir": entries_dir,
        "traces_dir": traces_dir,
        "index_path": index_path,
    }


def _write_lesson(entries_dir: Path, lesson_id: str, *, confidence: float = 0.4,
                   status: str = "candidate", source_trace: str | None = None,
                   features: list[str] | None = None) -> Path:
    """Write a minimal valid lesson YAML."""
    data = {
        "id": lesson_id,
        "title": f"Test lesson {lesson_id}",
        "severity": "high",
        "category": "monte_carlo",
        "status": status,
        "confidence": confidence,
        "created": "2026-03-20T10:00:00",
        "version": "",
        "source_trace": source_trace,
        "applies_when": {
            "method": ["monte_carlo"],
            "features": features or ["early_exercise"],
            "instrument": [],
            "error_signature": None,
        },
        "symptom": "Test symptom",
        "root_cause": "Test root cause",
        "fix": "Test fix",
        "validation": "",
    }
    path = entries_dir / f"{lesson_id}.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def _write_trace(traces_dir: Path, name: str, *, resolved: bool = True,
                 features: list[str] | None = None) -> Path:
    """Write a minimal trace YAML."""
    data = {
        "timestamp": "2026-03-23T22:55:10",
        "instrument": "test_instrument",
        "method": "monte_carlo",
        "description": "test trace",
        "pricing_plan": {
            "method": "monte_carlo",
            "features": features or ["early_exercise"],
        },
        "attempt": 0,
        "code_hash": "abc123",
        "validation_failures": [],
        "diagnosis": None,
        "resolved": resolved,
        "lesson_id": None,
        "duration_seconds": 1.0,
    }
    path = traces_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


class TestRecalibrateCandidates:
    """Tests for the recalibrate_candidates function."""

    def test_boost_from_resolved_source_trace(self, knowledge_tree):
        """Candidate with a resolved source_trace gets +0.15 boost."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "source_trace_1", resolved=True)
        _write_lesson(entries_dir, "mc_099", confidence=0.4,
                      source_trace=str(trace_path))

        stats = recalibrate_candidates(dry_run=True)

        assert stats["boosted"] == 1
        # 0.4 + 0.15 = 0.55 < 0.6, so not yet validated
        assert stats["validated"] == 0
        assert stats["unchanged"] == 0

    def test_boost_from_feature_cross_match(self, knowledge_tree):
        """Candidate with 3+ matching resolved traces gets +0.10 boost."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        _write_lesson(entries_dir, "mc_098", confidence=0.4,
                      features=["early_exercise"])

        # Write 3 resolved traces with matching features
        for i in range(3):
            _write_trace(traces_dir, f"matching_trace_{i}", resolved=True,
                         features=["early_exercise", "discounting"])

        stats = recalibrate_candidates(dry_run=True)

        assert stats["boosted"] == 1
        # 0.4 + 0.10 = 0.50 < 0.6, not validated
        assert stats["validated"] == 0

    def test_boost_combined_crosses_validation_threshold(self, knowledge_tree):
        """Source trace resolved + feature match boosts past 0.6 threshold."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "source_trace_combo", resolved=True,
                                  features=["early_exercise"])
        _write_lesson(entries_dir, "mc_097", confidence=0.4,
                      source_trace=str(trace_path),
                      features=["early_exercise"])

        # Write 3 more resolved traces with matching features
        for i in range(3):
            _write_trace(traces_dir, f"combo_trace_{i}", resolved=True,
                         features=["early_exercise"])

        stats = recalibrate_candidates(dry_run=True)

        assert stats["boosted"] == 1
        # 0.4 + 0.15 + 0.10 = 0.65 >= 0.6 → validated
        assert stats["validated"] == 1
        assert stats["promoted"] == 0  # 0.65 < 0.8

    def test_boost_combined_crosses_promotion_threshold(self, knowledge_tree):
        """High initial confidence + both boosts crosses 0.8 promotion threshold."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "source_trace_promo", resolved=True,
                                  features=["early_exercise"])
        _write_lesson(entries_dir, "mc_096", confidence=0.58,
                      source_trace=str(trace_path),
                      features=["early_exercise"])

        for i in range(3):
            _write_trace(traces_dir, f"promo_trace_{i}", resolved=True,
                         features=["early_exercise"])

        stats = recalibrate_candidates(dry_run=True)

        assert stats["boosted"] == 1
        # 0.58 + 0.15 + 0.10 = 0.83 >= 0.8 → promoted
        assert stats["validated"] == 1
        assert stats["promoted"] == 1

    def test_dry_run_no_mutations(self, knowledge_tree):
        """dry_run=True does not modify any files."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "source_dry", resolved=True,
                                  features=["early_exercise"])
        lesson_path = _write_lesson(entries_dir, "mc_095", confidence=0.4,
                                    source_trace=str(trace_path),
                                    features=["early_exercise"])

        for i in range(3):
            _write_trace(traces_dir, f"dry_trace_{i}", resolved=True,
                         features=["early_exercise"])

        original_content = lesson_path.read_text()

        stats = recalibrate_candidates(dry_run=True)
        assert stats["boosted"] == 1

        # Lesson file should be unchanged
        assert lesson_path.read_text() == original_content

    def test_no_source_trace_no_matching_traces(self, knowledge_tree):
        """Candidate with no source trace and no matching resolved traces is unchanged."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]

        _write_lesson(entries_dir, "mc_094", confidence=0.4,
                      features=["exotic_feature_xyz"])

        stats = recalibrate_candidates(dry_run=True)

        assert stats["unchanged"] == 1
        assert stats["boosted"] == 0

    def test_unresolved_source_trace_no_boost(self, knowledge_tree):
        """Source trace with resolved=False gives no boost."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "unresolved_trace",
                                  resolved=False)
        _write_lesson(entries_dir, "mc_093", confidence=0.4,
                      source_trace=str(trace_path))

        stats = recalibrate_candidates(dry_run=True)

        assert stats["unchanged"] == 1
        assert stats["boosted"] == 0

    def test_non_candidate_lessons_ignored(self, knowledge_tree):
        """Only candidate lessons are processed; promoted/validated are skipped."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "source_promoted", resolved=True)
        _write_lesson(entries_dir, "mc_092", confidence=0.9, status="promoted",
                      source_trace=str(trace_path))

        stats = recalibrate_candidates(dry_run=True)

        assert stats["boosted"] == 0
        assert stats["unchanged"] == 0

    def test_real_mutations_applied_when_not_dry_run(self, knowledge_tree):
        """Without dry_run, lesson files are actually mutated."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        trace_path = _write_trace(traces_dir, "source_real", resolved=True,
                                  features=["early_exercise"])
        lesson_path = _write_lesson(entries_dir, "mc_091", confidence=0.5,
                                    source_trace=str(trace_path),
                                    features=["early_exercise"])

        for i in range(3):
            _write_trace(traces_dir, f"real_trace_{i}", resolved=True,
                         features=["early_exercise"])

        stats = recalibrate_candidates(dry_run=False)

        assert stats["boosted"] == 1
        # 0.5 + 0.15 + 0.10 = 0.75 >= 0.6 → validated
        assert stats["validated"] == 1

        # Verify lesson file was mutated
        updated = yaml.safe_load(lesson_path.read_text())
        assert updated["confidence"] >= 0.75
        assert updated["status"] == "validated"

    def test_recalibration_log_written(self, knowledge_tree):
        """Recalibration results are logged to traces/recalibration_log.yaml."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        entries_dir = knowledge_tree["entries_dir"]
        traces_dir = knowledge_tree["traces_dir"]

        _write_lesson(entries_dir, "mc_090", confidence=0.4,
                      features=["exotic_feature_xyz"])

        recalibrate_candidates(dry_run=True)

        log_path = traces_dir / "recalibration_log.yaml"
        assert log_path.exists()
        log_data = yaml.safe_load(log_path.read_text())
        assert isinstance(log_data, list)
        assert len(log_data) == 1
        assert log_data[0]["dry_run"] is True
        assert "stats" in log_data[0]

    def test_empty_entries_dir(self, knowledge_tree):
        """No lesson entries produces zero stats."""
        from trellis.agent.knowledge.promotion import recalibrate_candidates

        stats = recalibrate_candidates(dry_run=True)

        assert stats["boosted"] == 0
        assert stats["unchanged"] == 0
