"""Tests for automatic lesson supersedes detection."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from trellis.agent.knowledge.promotion import (
    _fix_word_overlap,
    _detect_supersedes,
    promote_lesson,
    backfill_supersedes,
)


# ---------------------------------------------------------------------------
# _fix_word_overlap
# ---------------------------------------------------------------------------


class TestFixWordOverlap:
    def test_exact_match(self):
        assert _fix_word_overlap("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        score = _fix_word_overlap("hello world foo", "hello world bar")
        # intersection = {hello, world}, union = {hello, world, foo, bar}
        assert score == pytest.approx(2.0 / 4.0)

    def test_no_overlap(self):
        assert _fix_word_overlap("alpha beta", "gamma delta") == 0.0

    def test_empty_string(self):
        assert _fix_word_overlap("", "hello") == 0.0
        assert _fix_word_overlap("hello", "") == 0.0
        assert _fix_word_overlap("", "") == 0.0

    def test_case_insensitive(self):
        assert _fix_word_overlap("Hello World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# _detect_supersedes
# ---------------------------------------------------------------------------


def _write_entry(entries_dir: Path, lesson_id: str, **kwargs) -> None:
    """Helper to write a minimal lesson YAML entry."""
    data = {
        "id": lesson_id,
        "title": kwargs.get("title", f"Lesson {lesson_id}"),
        "severity": kwargs.get("severity", "medium"),
        "category": kwargs.get("category", "convention"),
        "status": kwargs.get("status", "promoted"),
        "confidence": kwargs.get("confidence", 1.0),
        "created": kwargs.get("created", ""),
        "version": kwargs.get("version", ""),
        "applies_when": kwargs.get("applies_when", {
            "method": [],
            "features": [],
            "instrument": [],
            "error_signature": None,
        }),
        "symptom": kwargs.get("symptom", "Some symptom"),
        "root_cause": kwargs.get("root_cause", "Some root cause"),
        "fix": kwargs.get("fix", "Some fix text"),
        "validation": kwargs.get("validation", ""),
    }
    data.update({k: v for k, v in kwargs.items() if k not in data})
    path = entries_dir / f"{lesson_id}.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


class TestDetectSupersedes:
    def test_finds_matching_lesson(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        _write_entry(
            entries_dir, "old_001",
            category="convention",
            fix="Always use the pinned interpreter and explicit shell for pricing runs",
        )
        result = _detect_supersedes(
            "new_001",
            "convention",
            "Always use the pinned interpreter and explicit shell for pricing runs to avoid failures",
            entries_dir,
        )
        assert result == "old_001"

    def test_ignores_different_category(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        _write_entry(
            entries_dir, "old_001",
            category="monte_carlo",
            fix="Always use the pinned interpreter and explicit shell for pricing runs",
        )
        result = _detect_supersedes(
            "new_001",
            "convention",
            "Always use the pinned interpreter and explicit shell for pricing runs",
            entries_dir,
        )
        assert result is None

    def test_ignores_below_threshold(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        _write_entry(
            entries_dir, "old_001",
            category="convention",
            fix="alpha beta gamma delta epsilon",
        )
        result = _detect_supersedes(
            "new_001",
            "convention",
            "zeta eta theta iota kappa",
            entries_dir,
        )
        assert result is None

    def test_ignores_non_promoted(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        _write_entry(
            entries_dir, "old_001",
            category="convention",
            status="candidate",
            fix="Always use the pinned interpreter and explicit shell",
        )
        result = _detect_supersedes(
            "new_001",
            "convention",
            "Always use the pinned interpreter and explicit shell",
            entries_dir,
        )
        assert result is None

    def test_ignores_self(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        _write_entry(
            entries_dir, "new_001",
            category="convention",
            fix="Always use the pinned interpreter and explicit shell",
        )
        result = _detect_supersedes(
            "new_001",
            "convention",
            "Always use the pinned interpreter and explicit shell",
            entries_dir,
        )
        assert result is None

    def test_empty_directory(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        result = _detect_supersedes("new_001", "convention", "some fix", entries_dir)
        assert result is None

    def test_nonexistent_directory(self, tmp_path):
        result = _detect_supersedes(
            "new_001", "convention", "some fix", tmp_path / "nonexistent",
        )
        assert result is None


# ---------------------------------------------------------------------------
# promote_lesson with auto-detect
# ---------------------------------------------------------------------------


class TestPromoteLessonSupersedes:
    def test_promote_detects_supersedes(self, tmp_path, monkeypatch):
        """Promoting a lesson auto-detects and archives the superseded one."""
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()

        # Create an already-promoted old lesson
        _write_entry(
            entries_dir, "con_001",
            category="convention",
            status="promoted",
            fix="Always use the pinned interpreter and explicit shell for pricing",
        )

        # Create a validated new lesson with similar fix
        _write_entry(
            entries_dir, "con_002",
            category="convention",
            status="validated",
            confidence=0.9,
            fix="Always use the pinned interpreter and explicit shell for pricing runs to avoid errors",
        )

        # Monkeypatch the module-level paths so promote_lesson uses tmp_path
        import trellis.agent.knowledge.promotion as promo_mod
        monkeypatch.setattr(promo_mod, "_LESSONS_DIR", tmp_path)
        # Suppress index rebuild (no real index file in tmp_path)
        monkeypatch.setattr(promo_mod, "_INDEX_REBUILD_SUPPRESS_DEPTH", 1)

        result = promote_lesson("con_002")
        assert result is True

        # Verify the new lesson has supersedes field
        new_data = yaml.safe_load((entries_dir / "con_002.yaml").read_text())
        assert "con_001" in new_data.get("supersedes", [])

        # Verify the old lesson is archived
        old_data = yaml.safe_load((entries_dir / "con_001.yaml").read_text())
        assert old_data["status"] == "archived"
        assert old_data.get("archive_reason") == "superseded_by_con_002"


# ---------------------------------------------------------------------------
# backfill_supersedes
# ---------------------------------------------------------------------------


class TestBackfillSupersedes:
    def test_dry_run_detects_relationships(self, tmp_path, monkeypatch):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()

        # Two promoted lessons in same category with overlapping fix
        _write_entry(
            entries_dir, "mc_001",
            category="monte_carlo",
            fix="Use antithetic variates for variance reduction in Monte Carlo paths",
            created="2026-03-01",
        )
        _write_entry(
            entries_dir, "mc_002",
            category="monte_carlo",
            fix="Use antithetic variates for variance reduction in Monte Carlo simulation paths to improve convergence",
            created="2026-03-15",
        )

        import trellis.agent.knowledge.promotion as promo_mod
        monkeypatch.setattr(promo_mod, "_LESSONS_DIR", tmp_path)

        result = backfill_supersedes(dry_run=True)
        assert "mc_002" in result
        assert "mc_001" in result["mc_002"]

        # Dry run should NOT mutate files
        old_data = yaml.safe_load((entries_dir / "mc_001.yaml").read_text())
        assert old_data["status"] == "promoted"

    def test_backfill_ignores_different_categories(self, tmp_path, monkeypatch):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()

        _write_entry(
            entries_dir, "mc_001",
            category="monte_carlo",
            fix="Use antithetic variates for variance reduction",
        )
        _write_entry(
            entries_dir, "con_001",
            category="convention",
            fix="Use antithetic variates for variance reduction",
        )

        import trellis.agent.knowledge.promotion as promo_mod
        monkeypatch.setattr(promo_mod, "_LESSONS_DIR", tmp_path)

        result = backfill_supersedes(dry_run=True)
        assert result == {}

    def test_backfill_mutates_when_not_dry_run(self, tmp_path, monkeypatch):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()

        _write_entry(
            entries_dir, "fd_001",
            category="finite_diff",
            fix="Apply Crank-Nicolson theta method for PDE stability",
            created="2026-03-01",
        )
        _write_entry(
            entries_dir, "fd_002",
            category="finite_diff",
            fix="Apply Crank-Nicolson theta method for PDE stability and convergence",
            created="2026-03-15",
        )

        import trellis.agent.knowledge.promotion as promo_mod
        monkeypatch.setattr(promo_mod, "_LESSONS_DIR", tmp_path)
        monkeypatch.setattr(promo_mod, "_INDEX_REBUILD_SUPPRESS_DEPTH", 1)

        result = backfill_supersedes(dry_run=False)
        assert "fd_002" in result

        # Old lesson should be archived
        old_data = yaml.safe_load((entries_dir / "fd_001.yaml").read_text())
        assert old_data["status"] == "archived"
        assert old_data.get("archive_reason") == "superseded_by_fd_002"

        # New lesson should have supersedes field
        new_data = yaml.safe_load((entries_dir / "fd_002.yaml").read_text())
        assert "fd_001" in new_data.get("supersedes", [])
