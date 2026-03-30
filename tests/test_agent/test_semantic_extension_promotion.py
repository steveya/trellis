"""Tests for semantic-DSL extension trace and lesson promotion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml


def _patch_extension_paths(monkeypatch, root: Path) -> None:
    import trellis.agent.knowledge.promotion as promotion_mod

    monkeypatch.setattr(promotion_mod, "_KNOWLEDGE_DIR", root)
    monkeypatch.setattr(promotion_mod, "_LESSONS_DIR", root / "lessons")
    monkeypatch.setattr(promotion_mod, "_TRACES_DIR", root / "traces")
    monkeypatch.setattr(
        promotion_mod,
        "_SEMANTIC_EXTENSION_TRACES_DIR",
        root / "traces" / "semantic_extensions",
    )
    monkeypatch.setattr(promotion_mod, "_INDEX_PATH", root / "lessons" / "index.yaml")
    monkeypatch.setattr(promotion_mod, "_REPO_ROOT", root.parents[2])


def test_record_semantic_extension_trace_is_stable_and_promotes_on_repeat(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import record_semantic_extension_trace
    from trellis.agent.semantic_contract_validation import (
        classify_semantic_gap,
        propose_semantic_extension,
        semantic_extension_summary,
        semantic_gap_summary,
    )

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_extension_paths(monkeypatch, knowledge_root)

    gap = classify_semantic_gap(
        "Price a resettable memory note with a holiday-adjusted schedule and monthly coupons.",
        instrument_type="structured_note",
    )
    proposal = replace(propose_semantic_extension(gap), confidence=0.9)
    gap_summary = semantic_gap_summary(gap)
    proposal_summary = semantic_extension_summary(proposal)

    trace_path = record_semantic_extension_trace(
        request_id="executor_build_test",
        request_text=gap.request_text,
        instrument_type=gap.instrument_type,
        semantic_gap=gap_summary,
        semantic_extension=proposal_summary,
        route_method="build_then_price",
    )
    trace_path_again = record_semantic_extension_trace(
        request_id="executor_build_test",
        request_text=gap.request_text,
        instrument_type=gap.instrument_type,
        semantic_gap=gap_summary,
        semantic_extension=proposal_summary,
        route_method="build_then_price",
    )

    assert trace_path == trace_path_again
    trace_data = yaml.safe_load(Path(trace_path).read_text())
    assert trace_data["trace_key"] == proposal.trace_key
    assert trace_data["occurrences"] == 2
    assert trace_data["lesson_id"] is not None
    assert trace_data["lesson_contract"]["valid"] is True
    assert trace_data["lesson_promotion_outcome"] == "promoted"

    lesson_path = knowledge_root / "lessons" / "entries" / f"{trace_data['lesson_id']}.yaml"
    assert lesson_path.exists()
    lesson_data = yaml.safe_load(lesson_path.read_text())
    assert lesson_data["status"] == "promoted"
    assert lesson_data["source_trace"] == trace_path
    assert "generate_schedule" in lesson_data["applies_when"]["features"]
