from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def _patch_promotion_paths(monkeypatch, root: Path) -> None:
    import trellis.agent.knowledge.promotion as promotion_mod

    monkeypatch.setattr(promotion_mod, "_KNOWLEDGE_DIR", root)
    monkeypatch.setattr(promotion_mod, "_LESSONS_DIR", root / "lessons")
    monkeypatch.setattr(promotion_mod, "_TRACES_DIR", root / "traces")
    monkeypatch.setattr(promotion_mod, "_INDEX_PATH", root / "lessons" / "index.yaml")
    monkeypatch.setattr(promotion_mod, "_REPO_ROOT", root.parents[2])


def _write_candidate(
    root: Path,
    name: str,
    *,
    module_path: str = "trellis.instruments._agent._fresh.demo",
    comparison_target: str = "demo",
    cross_status: str = "passed",
    deviation: float = 0.2,
    code: str | None = None,
) -> Path:
    traces = root / "traces"
    candidate_dir = traces / "promotion_candidates"
    platform_dir = traces / "platform"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    platform_dir.mkdir(parents=True, exist_ok=True)
    platform_trace = platform_dir / "executor_build_demo.yaml"
    platform_trace.write_text("status: completed\n")
    candidate_path = candidate_dir / name
    source = code or (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class DemoSpec:\n"
        "    strike: float = 100.0\n\n"
        "class DemoPayoff:\n"
        "    def evaluate(self, market_state):\n"
        "        return 1.0\n"
    )
    candidate_path.write_text(yaml.safe_dump({
        "timestamp": "2026-03-26T20:00:00",
        "task_id": "T999",
        "task_title": "Demo candidate",
        "instrument_type": "quanto_option",
        "comparison_target": comparison_target,
        "preferred_method": "analytical",
        "reference_target": False,
        "payoff_class": "DemoPayoff",
        "module_path": module_path,
        "attempts": 2,
        "platform_request_id": "executor_build_demo",
        "platform_trace_path": str(platform_trace),
        "market_context": {"source": "mock"},
        "cross_validation": {
            "status": cross_status,
            "prices": {comparison_target: 101.0},
            "price_errors": {},
            "tolerance_pct": 5.0,
            "deviations_pct": {comparison_target: deviation},
            "passed_targets": [comparison_target] if cross_status == "passed" else [],
            "failed_targets": [] if cross_status == "passed" else [comparison_target],
            "successful_targets": [comparison_target] if cross_status == "passed" else [],
        },
        "code_hash": hashlib.sha256(source.strip().encode()).hexdigest()[:12],
        "code": source,
    }, sort_keys=False))
    return candidate_path


def _write_adapter_pair(
    root: Path,
    name: str,
    *,
    checked_in_code: str,
    fresh_code: str,
) -> tuple[Path, Path]:
    repo_root = root.parents[2]
    agent_dir = repo_root / "trellis" / "instruments" / "_agent"
    checked_in_path = agent_dir / f"{name}.py"
    fresh_path = agent_dir / "_fresh" / f"{name}.py"
    checked_in_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_path.parent.mkdir(parents=True, exist_ok=True)
    checked_in_path.write_text(checked_in_code)
    fresh_path.write_text(fresh_code)
    return checked_in_path, fresh_path


def test_review_promotion_candidate_approves_fresh_cross_validated_candidate(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import review_promotion_candidate

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    candidate_path = _write_candidate(knowledge_root, "candidate.yaml")

    review = review_promotion_candidate(candidate_path)

    assert review["status"] == "approved"
    assert review["approved"] is True
    assert review["recommended_module_path"] == "trellis.instruments._agent.demo"
    assert review["recommended_file_path"].endswith("trellis/instruments/_agent/demo.py")
    assert Path(review["review_path"]).exists()
    assert all(check["passed"] for check in review["checks"] if check["blocking"])


def test_review_promotion_candidate_rejects_when_cross_validation_did_not_pass(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import review_promotion_candidate

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    candidate_path = _write_candidate(
        knowledge_root,
        "candidate_failed.yaml",
        cross_status="failed",
        deviation=12.0,
    )

    review = review_promotion_candidate(candidate_path)

    assert review["status"] == "rejected"
    assert review["approved"] is False
    failed = {check["name"] for check in review["checks"] if not check["passed"] and check["blocking"]}
    assert "cross_validation_passed" in failed
    assert "target_within_tolerance" in failed


def test_review_promotion_candidate_rejects_non_fresh_module_path(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import review_promotion_candidate

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    candidate_path = _write_candidate(
        knowledge_root,
        "candidate_existing.yaml",
        module_path="trellis.instruments._agent.demo",
    )

    review = review_promotion_candidate(candidate_path)

    assert review["status"] == "rejected"
    failed = {check["name"] for check in review["checks"] if not check["passed"] and check["blocking"]}
    assert "fresh_module_path" in failed


def test_list_promotion_candidate_paths_returns_latest_first(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import list_promotion_candidate_paths

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    first = _write_candidate(knowledge_root, "20260326_195934_t105_quanto_bs.yaml")
    second = _write_candidate(knowledge_root, "20260326_195935_t105_mc_quanto.yaml")

    paths = list_promotion_candidate_paths(limit=2)

    assert paths == [str(second), str(first)]


def test_adopt_promotion_candidate_writes_approved_candidate_to_target(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import (
        adopt_promotion_candidate,
        review_promotion_candidate,
    )

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    candidate_path = _write_candidate(knowledge_root, "candidate.yaml")
    review = review_promotion_candidate(candidate_path)
    target_path = Path(review["recommended_file_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("# old deterministic route\n")

    adoption = adopt_promotion_candidate(review["review_path"])

    assert adoption["status"] == "adopted"
    assert adoption["adopted"] is True
    assert adoption["changed"] is True
    assert target_path.read_text() == (yaml.safe_load(candidate_path.read_text())["code"]).rstrip() + "\n"
    assert Path(adoption["adoption_path"]).exists()


def test_adopt_promotion_candidate_blocks_rejected_review(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import (
        adopt_promotion_candidate,
        review_promotion_candidate,
    )

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    candidate_path = _write_candidate(
        knowledge_root,
        "candidate_failed.yaml",
        cross_status="failed",
        deviation=12.0,
    )
    review = review_promotion_candidate(candidate_path)

    adoption = adopt_promotion_candidate(review["review_path"])

    assert adoption["status"] == "blocked"
    assert adoption["adopted"] is False
    assert Path(adoption["adoption_path"]).exists()
    failed = {check["name"] for check in adoption["checks"] if not check["passed"] and check["blocking"]}
    assert "review_approved" in failed


def test_adopt_promotion_candidate_dry_run_preserves_target_file(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import (
        adopt_promotion_candidate,
        review_promotion_candidate,
    )

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    candidate_path = _write_candidate(knowledge_root, "candidate.yaml")
    review = review_promotion_candidate(candidate_path)
    target_path = Path(review["recommended_file_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("# keep existing route\n")

    adoption = adopt_promotion_candidate(review["review_path"], dry_run=True)

    assert adoption["status"] == "ready"
    assert adoption["adopted"] is False
    assert target_path.read_text() == "# keep existing route\n"
    assert Path(adoption["adoption_path"]).exists()


def test_detect_adapter_lifecycle_records_flags_stale_checked_in_adapter(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import (
        AdapterLifecycleStatus,
        detect_adapter_lifecycle_records,
    )

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    _write_adapter_pair(
        knowledge_root,
        "example_adapter",
        checked_in_code="class ExampleAdapter:\n    value = 1\n",
        fresh_code="class ExampleAdapter:\n    value = 2\n",
    )

    records = detect_adapter_lifecycle_records()
    stale = [record for record in records if record.status == AdapterLifecycleStatus.STALE]
    fresh = [record for record in records if record.status == AdapterLifecycleStatus.FRESH]

    assert len(stale) == 1
    assert len(fresh) == 1
    assert stale[0].adapter_id == "trellis.instruments._agent.example_adapter"
    assert stale[0].replacement == "trellis.instruments._agent._fresh.example_adapter"
    assert fresh[0].supersedes == (stale[0].adapter_id,)
    assert fresh[0].module_path == "trellis.instruments._agent._fresh.example_adapter"
    assert "validated fresh-build replacement" not in fresh[0].reason.lower()
    assert "await" in fresh[0].reason.lower() or "pending" in fresh[0].reason.lower()


def test_detect_adapter_lifecycle_records_keeps_fx_and_quanto_shells_fresh():
    from trellis.agent.knowledge.promotion import (
        AdapterLifecycleStatus,
        detect_adapter_lifecycle_records,
    )

    stale_ids = {
        record.adapter_id
        for record in detect_adapter_lifecycle_records()
        if record.status == AdapterLifecycleStatus.STALE
    }

    assert "trellis.instruments._agent.fxvanillaanalytical" not in stale_ids
    assert "trellis.instruments._agent.fxvanillamontecarlo" not in stale_ids
    assert "trellis.instruments._agent.quantooptionanalytical" not in stale_ids
    assert "trellis.instruments._agent.quantooptionmontecarlo" not in stale_ids


def test_review_and_adopt_promotion_candidate_propagate_adapter_lifecycle_state(monkeypatch, tmp_path):
    from trellis.agent.knowledge.promotion import (
        adopt_promotion_candidate,
        review_promotion_candidate,
    )
    from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload, format_knowledge_for_prompt

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    _, fresh_path = _write_adapter_pair(
        knowledge_root,
        "demo",
        checked_in_code="class DemoPayoff:\n    value = 1\n",
        fresh_code="class DemoPayoff:\n    value = 2\n",
    )
    candidate_path = _write_candidate(
        knowledge_root,
        "candidate.yaml",
        code=fresh_path.read_text(),
    )

    review = review_promotion_candidate(candidate_path)
    review_lifecycle = review["adapter_lifecycle"]

    assert review_lifecycle["stage"] == "deprecated"
    assert review_lifecycle["raw"]["summary"]["stale_adapter_count"] == 1
    assert review_lifecycle["resolved"]["summary"]["deprecated_adapter_count"] == 1
    assert review_lifecycle["resolved"]["summary"]["stale_adapter_count"] == 0

    review_payload = build_shared_knowledge_payload({})
    assert review_payload["summary"]["deprecated_adapter_count"] == 1
    assert review_payload["summary"]["stale_adapter_count"] == 0
    review_prompt = format_knowledge_for_prompt({}, compact=True)
    assert "DEPRECATED" in review_prompt
    assert "STALE" not in review_prompt

    adoption = adopt_promotion_candidate(review["review_path"])
    adoption_lifecycle = adoption["adapter_lifecycle"]

    assert adoption_lifecycle["stage"] == "archived"
    assert adoption_lifecycle["raw"]["summary"]["stale_adapter_count"] == 1
    assert adoption_lifecycle["resolved"]["summary"]["archived_adapter_count"] == 1
    assert adoption_lifecycle["resolved"]["summary"]["stale_adapter_count"] == 0

    adoption_payload = build_shared_knowledge_payload({})
    assert adoption_payload["summary"]["archived_adapter_count"] == 1
    assert adoption_payload["summary"]["stale_adapter_count"] == 0
    adoption_prompt = format_knowledge_for_prompt({}, compact=True)
    assert "STALE" not in adoption_prompt
    assert "DEPRECATED" not in adoption_prompt


def test_format_knowledge_for_prompt_surfaces_stale_adapter_warning(monkeypatch, tmp_path):
    from trellis.agent.knowledge.retrieval import build_shared_knowledge_payload, format_knowledge_for_prompt

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    _write_adapter_pair(
        knowledge_root,
        "example_adapter",
        checked_in_code="class ExampleAdapter:\n    value = 1\n",
        fresh_code="class ExampleAdapter:\n    value = 2\n",
    )

    prompt = format_knowledge_for_prompt({}, compact=True)
    payload = build_shared_knowledge_payload({})

    assert "Adapter Freshness" in prompt
    assert "STALE" in prompt
    assert "example_adapter" in prompt
    assert payload["summary"]["stale_adapter_count"] == 1
    assert payload["summary"]["stale_adapter_ids"] == [
        "trellis.instruments._agent.example_adapter"
    ]
