"""Tests for the explicit post-benchmark _agent adapter promotion workflow.

The FinancePy pilot (QUA-864) treats benchmark execution and `_agent` admission
as separate responsibilities.  ``QUA-866`` fails the benchmark closed when it
leaks into the admitted surface.  ``QUA-867`` adds the opposite direction: an
explicit admission step that copies a validated fresh-build artifact into
``trellis/instruments/_agent`` only when every provenance link (benchmark run
id, git sha, knowledge revision, code hash) matches.  This file exercises the
fail-closed contract for that workflow.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


def _patch_promotion_paths(monkeypatch, knowledge_root: Path) -> None:
    import trellis.agent.knowledge.promotion as promotion_mod

    repo_root = knowledge_root.parents[2]
    monkeypatch.setattr(promotion_mod, "_KNOWLEDGE_DIR", knowledge_root)
    monkeypatch.setattr(promotion_mod, "_LESSONS_DIR", knowledge_root / "lessons")
    monkeypatch.setattr(promotion_mod, "_TRACES_DIR", knowledge_root / "traces")
    monkeypatch.setattr(
        promotion_mod, "_INDEX_PATH", knowledge_root / "lessons" / "index.yaml"
    )
    monkeypatch.setattr(promotion_mod, "_REPO_ROOT", repo_root)


def _prepare_record_and_candidate(
    tmp_path: Path,
    monkeypatch,
    *,
    task_id: str = "F009",
) -> tuple[Path, dict[str, object]]:
    from trellis.agent.knowledge.promotion import record_benchmark_promotion_candidate

    knowledge_root = tmp_path / "trellis" / "agent" / "knowledge"
    _patch_promotion_paths(monkeypatch, knowledge_root)
    repo_root = tmp_path

    report_root = repo_root / "task_runs" / "financepy_benchmarks"
    generated_dir = report_root / "generated" / task_id.lower() / "analytical"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_file = generated_dir / "barrieroption.py"
    source = (
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\n"
        "class BarrierOptionSpec:\n"
        "    strike: float = 100.0\n\n"
        "class BarrierOptionPayoff:\n"
        "    def evaluate(self, market_state):\n"
        "        return 1.0\n"
    )
    generated_file.write_text(source)

    record = {
        "task_id": task_id,
        "title": "FinancePy parity benchmark",
        "instrument_type": "barrier_option",
        "preferred_method": "analytical",
        "benchmark_execution_policy": "fresh_generated",
        "benchmark_campaign_id": "pilot",
        "run_id": f"{task_id}_20260415T120000000000Z",
        "status": "priced",
        "git_sha": "deadbeef",
        "knowledge_revision": "cafebabe",
        "comparison_summary": {
            "status": "passed",
            "tolerance_pct": 2.0,
            "compared_outputs": ("price",),
            "output_deviation_pct": {"price": 0.1},
        },
        "generated_artifact": {
            "module_name": f"trellis_benchmarks._fresh.{task_id.lower()}.analytical.barrieroption",
            "class_name": "BarrierOptionPayoff",
            "file_path": str(generated_file),
            "module_path": (
                f"task_runs/financepy_benchmarks/generated/{task_id.lower()}/analytical/"
                "barrieroption.py"
            ),
            "code_hash": hashlib.sha256(source.strip().encode()).hexdigest(),
            "is_fresh_build": True,
            "admission_target_module_name": "trellis.instruments._agent.barrieroption",
        },
    }
    history_dir = report_root / "history" / task_id
    latest_dir = report_root / "latest"
    history_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{record['run_id']}.json"
    latest_path = latest_dir / f"{task_id}.json"
    history_path.write_text(json.dumps(record, indent=2))
    latest_path.write_text(json.dumps(record, indent=2))
    record["history_path"] = str(history_path)
    record["latest_path"] = str(latest_path)

    candidate_path = Path(record_benchmark_promotion_candidate(benchmark_record=record))
    return candidate_path, record


def test_promote_benchmark_candidate_writes_agent_adapter_with_admission_log(
    tmp_path, monkeypatch
):
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, record = _prepare_record_and_candidate(tmp_path, monkeypatch)

    result = promote_benchmark_candidate(candidate_path, repo_root=tmp_path)

    assert result["status"] == "promoted"
    assert result["dry_run"] is False

    target_path = Path(result["admission_target_file_path"])
    assert target_path.exists()
    promoted_source = target_path.read_text()
    assert "class BarrierOptionPayoff" in promoted_source

    admission_log = Path(result["admission_log_path"])
    assert admission_log.exists()
    payload = yaml.safe_load(admission_log.read_text())
    assert payload["benchmark_run_id"] == record["run_id"]
    assert payload["git_sha"] == record["git_sha"]
    assert payload["knowledge_revision"] == record["knowledge_revision"]
    assert payload["admission_target_module_name"] == (
        "trellis.instruments._agent.barrieroption"
    )
    assert payload["code_hash"] == hashlib.sha256(promoted_source.strip().encode()).hexdigest()


def test_promote_benchmark_candidate_dry_run_leaves_filesystem_untouched(
    tmp_path, monkeypatch
):
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    target_path = tmp_path / "trellis" / "instruments" / "_agent" / "barrieroption.py"

    result = promote_benchmark_candidate(
        candidate_path, repo_root=tmp_path, dry_run=True
    )

    assert result["status"] == "would_promote"
    assert result["dry_run"] is True
    assert result["admission_target_file_path"].endswith(
        "trellis/instruments/_agent/barrieroption.py"
    )
    assert not target_path.exists()


def test_promote_benchmark_candidate_fails_closed_on_hash_mismatch(tmp_path, monkeypatch):
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    candidate = yaml.safe_load(candidate_path.read_text())
    candidate["code_hash"] = "badbadbadbad"
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    with pytest.raises(PromotionAdmissionError) as exc_info:
        promote_benchmark_candidate(candidate_path, repo_root=tmp_path)

    assert "hash" in str(exc_info.value).lower()
    target_path = tmp_path / "trellis" / "instruments" / "_agent" / "barrieroption.py"
    assert not target_path.exists()


def test_promote_benchmark_candidate_fails_closed_when_benchmark_record_missing(
    tmp_path, monkeypatch
):
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    candidate_path, record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    history_path = Path(record["history_path"])
    history_path.unlink()

    with pytest.raises(PromotionAdmissionError) as exc_info:
        promote_benchmark_candidate(candidate_path, repo_root=tmp_path)

    message = str(exc_info.value).lower()
    assert "benchmark_record_exists" in message or "benchmark record" in message


def test_promote_benchmark_candidate_fails_closed_on_stale_git_sha(tmp_path, monkeypatch):
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    candidate_path, record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    history_path = Path(record["history_path"])
    mutated = json.loads(history_path.read_text())
    mutated["git_sha"] = "different_sha"
    history_path.write_text(json.dumps(mutated, indent=2))

    with pytest.raises(PromotionAdmissionError) as exc_info:
        promote_benchmark_candidate(candidate_path, repo_root=tmp_path)

    assert "git_sha" in str(exc_info.value).lower() or "provenance" in str(exc_info.value).lower()


def test_promote_benchmark_candidate_refuses_when_review_rejects(tmp_path, monkeypatch):
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    candidate_path, record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    history_path = Path(record["history_path"])
    mutated = json.loads(history_path.read_text())
    mutated["comparison_summary"]["status"] = "failed"
    history_path.write_text(json.dumps(mutated, indent=2))

    with pytest.raises(PromotionAdmissionError) as exc_info:
        promote_benchmark_candidate(candidate_path, repo_root=tmp_path)

    message = str(exc_info.value).lower()
    assert any(
        token in message
        for token in ("review", "rejected", "passed", "comparison_summary")
    )


def test_promote_agent_adapter_cli_invokes_promotion_function(tmp_path, monkeypatch, capsys):
    from scripts import promote_agent_adapter

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)

    exit_code = promote_agent_adapter.main(
        [
            "--candidate",
            str(candidate_path),
            "--repo-root",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "would_promote"
    assert payload["dry_run"] is True


def test_promote_agent_adapter_cli_surfaces_failure_with_nonzero_exit(
    tmp_path, monkeypatch, capsys
):
    from scripts import promote_agent_adapter

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    candidate = yaml.safe_load(candidate_path.read_text())
    candidate["code_hash"] = "badbadbadbad"
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    exit_code = promote_agent_adapter.main(
        [
            "--candidate",
            str(candidate_path),
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert exit_code != 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "rejected"
    assert "hash" in payload["error"].lower()
