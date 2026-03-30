"""Tests for LLM-assisted principle extraction from promoted lessons."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from trellis.agent.knowledge.promotion import (
    _draft_principle_prompt,
    adopt_principle_candidate,
    draft_principle_candidates,
    review_principle_candidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(entry_id: str, category: str, status: str = "promoted") -> dict:
    """Return a minimal lesson entry dict."""
    return {
        "id": entry_id,
        "title": f"Lesson {entry_id}",
        "fix": f"Fix for {entry_id}",
        "category": category,
        "status": status,
        "severity": "medium",
        "applies_when": {"method": [], "features": [], "instrument": []},
    }


def _populate_entries(tmp_path: Path, entries: list[dict]) -> Path:
    """Write lesson entry files and return the entries directory."""
    entries_dir = tmp_path / "lessons" / "entries"
    entries_dir.mkdir(parents=True)
    for e in entries:
        path = entries_dir / f"{e['id']}.yaml"
        path.write_text(yaml.dump(e, default_flow_style=False))
    return entries_dir


def _mock_llm_fn(prompt: str) -> dict:
    """Canned LLM response for testing."""
    return {
        "rule": "Always do the right thing",
        "rationale": "Because all lessons agree on this",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDraftPrinciplePrompt:
    def test_basic_format(self):
        lessons = [
            {"id": "mc_001", "title": "Use many paths", "fix": "Increase path count"},
            {"id": "mc_002", "title": "Check SE", "fix": "Verify standard error"},
            {"id": "mc_003", "title": "Basis risk", "fix": "Use Laguerre basis"},
        ]
        prompt = _draft_principle_prompt("monte_carlo", lessons)
        assert "monte_carlo" in prompt
        assert "3 lessons" in prompt
        assert "mc_001" in prompt
        assert "mc_002" in prompt
        assert "mc_003" in prompt
        assert "JSON" in prompt

    def test_missing_fields_use_defaults(self):
        lessons = [{"id": "x"}, {}, {"title": "T"}]
        prompt = _draft_principle_prompt("cat", lessons)
        assert "?" in prompt  # fallback for missing fields


class TestDraftPrincipleCandidates:
    def test_drafts_candidates_for_eligible_categories(self, tmp_path, monkeypatch):
        entries = [
            _make_entry("mc_001", "monte_carlo"),
            _make_entry("mc_002", "monte_carlo"),
            _make_entry("mc_003", "monte_carlo"),
            _make_entry("cal_001", "calibration"),  # only 1 — should be skipped
        ]
        _populate_entries(tmp_path, entries)

        # Patch the module-level paths
        import trellis.agent.knowledge.promotion as promo
        monkeypatch.setattr(promo, "_LESSONS_DIR", tmp_path / "lessons")
        traces_dir = tmp_path / "traces"
        monkeypatch.setattr(promo, "_TRACES_DIR", traces_dir)

        candidates = draft_principle_candidates(_llm_fn=_mock_llm_fn)

        assert len(candidates) == 1
        cp = candidates[0]["candidate_principle"]
        assert cp["rule"] == "Always do the right thing"
        assert cp["category"] == "monte_carlo"
        assert cp["status"] == "candidate"
        assert cp["confidence"] == 0.7
        assert "mc_001" in cp["derived_from"]
        assert "drafted_at" in cp

        # Check file was written
        out_dir = traces_dir / "principle_candidates"
        assert out_dir.exists()
        files = list(out_dir.glob("monte_carlo_*.yaml"))
        assert len(files) == 1

    def test_skips_non_promoted(self, tmp_path, monkeypatch):
        entries = [
            _make_entry("mc_001", "monte_carlo", status="candidate"),
            _make_entry("mc_002", "monte_carlo", status="candidate"),
            _make_entry("mc_003", "monte_carlo", status="candidate"),
        ]
        _populate_entries(tmp_path, entries)

        import trellis.agent.knowledge.promotion as promo
        monkeypatch.setattr(promo, "_LESSONS_DIR", tmp_path / "lessons")
        monkeypatch.setattr(promo, "_TRACES_DIR", tmp_path / "traces")

        candidates = draft_principle_candidates(_llm_fn=_mock_llm_fn)
        assert len(candidates) == 0

    def test_handles_llm_failure(self, tmp_path, monkeypatch):
        entries = [
            _make_entry("mc_001", "monte_carlo"),
            _make_entry("mc_002", "monte_carlo"),
            _make_entry("mc_003", "monte_carlo"),
        ]
        _populate_entries(tmp_path, entries)

        import trellis.agent.knowledge.promotion as promo
        monkeypatch.setattr(promo, "_LESSONS_DIR", tmp_path / "lessons")
        monkeypatch.setattr(promo, "_TRACES_DIR", tmp_path / "traces")

        def failing_llm(prompt):
            raise RuntimeError("LLM unavailable")

        candidates = draft_principle_candidates(_llm_fn=failing_llm)
        assert len(candidates) == 0


class TestReviewPrincipleCandidate:
    def test_loads_valid_candidate(self, tmp_path):
        candidate = {
            "candidate_principle": {
                "rule": "Do X",
                "rationale": "Because Y",
                "derived_from": ["a", "b"],
                "category": "test",
                "confidence": 0.7,
                "status": "candidate",
                "drafted_at": "2026-03-29T00:00:00",
            }
        }
        path = tmp_path / "candidate.yaml"
        path.write_text(yaml.dump(candidate))

        result = review_principle_candidate(path)
        assert result["candidate_principle"]["rule"] == "Do X"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            review_principle_candidate(tmp_path / "nonexistent.yaml")

    def test_raises_on_invalid_format(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("not_a_candidate: true")
        with pytest.raises(ValueError):
            review_principle_candidate(path)


class TestAdoptPrincipleCandidate:
    def test_appends_to_principles(self, tmp_path, monkeypatch):
        # Set up existing principles
        import trellis.agent.knowledge.promotion as promo
        monkeypatch.setattr(promo, "_KNOWLEDGE_DIR", tmp_path)

        canonical_dir = tmp_path / "canonical"
        canonical_dir.mkdir()
        existing = [
            {"id": "P1", "rule": "Existing rule", "derived_from": ["x"], "category": "test"},
        ]
        principles_path = canonical_dir / "principles.yaml"
        principles_path.write_text(yaml.dump(existing))

        # Write candidate
        candidate = {
            "candidate_principle": {
                "rule": "New rule from lessons",
                "rationale": "Good reason",
                "derived_from": ["mc_001", "mc_002", "mc_003"],
                "category": "monte_carlo",
                "confidence": 0.7,
                "status": "candidate",
                "drafted_at": "2026-03-29T00:00:00",
            }
        }
        cand_path = tmp_path / "candidate.yaml"
        cand_path.write_text(yaml.dump(candidate))

        result = adopt_principle_candidate(cand_path)
        assert result is True

        updated = yaml.safe_load(principles_path.read_text())
        assert len(updated) == 2
        new = updated[1]
        assert new["id"] == "P2"
        assert new["rule"] == "New rule from lessons"
        assert new["category"] == "monte_carlo"
        assert "mc_001" in new["derived_from"]

    def test_creates_principles_file_if_missing(self, tmp_path, monkeypatch):
        import trellis.agent.knowledge.promotion as promo
        monkeypatch.setattr(promo, "_KNOWLEDGE_DIR", tmp_path)

        canonical_dir = tmp_path / "canonical"
        canonical_dir.mkdir()

        candidate = {
            "candidate_principle": {
                "rule": "First rule ever",
                "rationale": "Reason",
                "derived_from": ["a"],
                "category": "test",
            }
        }
        cand_path = tmp_path / "candidate.yaml"
        cand_path.write_text(yaml.dump(candidate))

        result = adopt_principle_candidate(cand_path)
        assert result is True

        principles_path = canonical_dir / "principles.yaml"
        updated = yaml.safe_load(principles_path.read_text())
        assert len(updated) == 1
        assert updated[0]["id"] == "P1"
