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
            # Mirror the live runner: `_generated_artifact_from_result` hashes
            # `read_bytes()` (no `.strip()`).  The test fixture must use the
            # same scheme so admission's hash check sees the same bytes.
            "code_hash": hashlib.sha256(generated_file.read_bytes()).hexdigest(),
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
    assert payload["code_hash"] == hashlib.sha256(target_path.read_bytes()).hexdigest()


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


def test_record_benchmark_promotion_candidate_is_idempotent_for_same_code_hash(
    tmp_path, monkeypatch
):
    from trellis.agent.knowledge.promotion import record_benchmark_promotion_candidate

    candidate_path_first, record = _prepare_record_and_candidate(
        tmp_path, monkeypatch, task_id="F009"
    )
    second_path = record_benchmark_promotion_candidate(benchmark_record=record)

    assert Path(second_path).resolve() == Path(candidate_path_first).resolve(), (
        "expected dedup to return the existing candidate path when the code "
        "hash matches a prior emission"
    )
    candidate_dir = candidate_path_first.parent
    matches = sorted(candidate_dir.glob("*_f009_financepy_benchmark.yaml"))
    assert len(matches) == 1, (
        "rerunning the benchmark with byte-identical generated code must not "
        "produce a second candidate file"
    )


def test_promote_agent_adapter_cli_batch_dry_run_runs_all_candidates(
    tmp_path, monkeypatch, capsys
):
    from scripts import promote_agent_adapter

    f009_path, _ = _prepare_record_and_candidate(tmp_path, monkeypatch, task_id="F009")
    f001_path, _ = _prepare_record_and_candidate(tmp_path, monkeypatch, task_id="F001")
    candidate_root = f009_path.parent
    assert f001_path.parent == candidate_root

    exit_code = promote_agent_adapter.main(
        [
            "--candidate-glob",
            "*.yaml",
            "--candidate-root",
            str(candidate_root),
            "--repo-root",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "would_promote_all"
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 2
    assert payload["processed_count"] == 2
    statuses = {entry["status"] for entry in payload["results"]}
    assert statuses == {"would_promote"}


def test_promote_agent_adapter_cli_batch_apply_halts_on_first_rejection(
    tmp_path, monkeypatch, capsys
):
    from scripts import promote_agent_adapter

    f001_path, _ = _prepare_record_and_candidate(tmp_path, monkeypatch, task_id="F001")
    f009_path, _ = _prepare_record_and_candidate(tmp_path, monkeypatch, task_id="F009")
    candidate_root = f001_path.parent

    # Corrupt the candidate that sorts first lexically so the batch hits a
    # rejection before touching the second.
    [first_path] = sorted(p for p in candidate_root.glob("*.yaml") if p.name.endswith("_f001_financepy_benchmark.yaml"))
    candidate = yaml.safe_load(first_path.read_text())
    candidate["code_hash"] = "badbadbadbad"
    first_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    exit_code = promote_agent_adapter.main(
        [
            "--candidate-glob",
            "*.yaml",
            "--candidate-root",
            str(candidate_root),
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "rejected"
    assert payload["candidate_count"] == 2
    assert payload["processed_count"] == 1
    assert payload["results"][0]["status"] == "rejected"

    # The second candidate's admission target was not written because the
    # batch halted on the first rejection.
    f009_target = tmp_path / "trellis" / "instruments" / "_agent" / "barrieroption.py"
    assert not f009_target.exists()


def test_promote_agent_adapter_cli_requires_exactly_one_candidate_source(capsys):
    from scripts import promote_agent_adapter

    exit_code = promote_agent_adapter.main([])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "candidate" in payload["error"].lower()


def test_promote_benchmark_candidate_resolves_relative_record_path(
    tmp_path, monkeypatch
):
    """Relative benchmark_record_path resolves against the candidate file.

    PR #590 Copilot review: a candidate that stored a repo-relative
    `benchmark_record_path` would fail admission when the CLI was invoked
    from a different CWD.  Resolve relative paths against `path.parent`.
    """
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    history_path = Path(record["history_path"])
    candidate = yaml.safe_load(candidate_path.read_text())
    import os as _os
    relative = _os.path.relpath(history_path, candidate_path.parent.resolve())
    candidate["benchmark_provenance"]["benchmark_record_path"] = str(relative)
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    result = promote_benchmark_candidate(
        candidate_path, repo_root=tmp_path, dry_run=True
    )

    assert result["status"] == "would_promote"


def test_promote_benchmark_candidate_rejects_agent_fresh_admission_target(
    tmp_path, monkeypatch
):
    """`_agent._fresh.*` is the scratch namespace, not the admitted surface.

    PR #590 Copilot review: a candidate naming the isolation namespace as
    its admission target should fail closed.
    """
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    candidate = yaml.safe_load(candidate_path.read_text())
    candidate["admission_target_module_name"] = (
        "trellis.instruments._agent._fresh.barrieroption"
    )
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    with pytest.raises(PromotionAdmissionError) as exc_info:
        promote_benchmark_candidate(candidate_path, repo_root=tmp_path)
    assert "_fresh" in str(exc_info.value)


def test_promote_benchmark_candidate_rejects_yaml_target_file_path_mismatch(
    tmp_path, monkeypatch
):
    """A YAML-supplied admission_target_file_path that disagrees with the
    module-derived path is rejected -- otherwise a tampered candidate could
    overwrite arbitrary files in the repo (PR #590 Copilot/Codex P1)."""
    from trellis.agent.knowledge.promotion import (
        PromotionAdmissionError,
        promote_benchmark_candidate,
    )

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    candidate = yaml.safe_load(candidate_path.read_text())
    candidate["admission_target_file_path"] = str(
        tmp_path / "trellis" / "core" / "evil_overwrite_target.py"
    )
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    with pytest.raises(PromotionAdmissionError) as exc_info:
        promote_benchmark_candidate(candidate_path, repo_root=tmp_path)
    message = str(exc_info.value).lower()
    assert "admission_target_file_path" in message or "disagrees" in message


def test_promote_benchmark_candidate_admission_target_is_under_agent_tree(
    tmp_path, monkeypatch
):
    """The resolved admission target must be inside trellis/instruments/_agent/."""
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    result = promote_benchmark_candidate(
        candidate_path, repo_root=tmp_path, dry_run=True
    )
    target = Path(result["admission_target_file_path"])
    expected_root = (tmp_path / "trellis" / "instruments" / "_agent").resolve()
    target.relative_to(expected_root)  # raises ValueError if outside


def test_promote_benchmark_candidate_admission_timestamp_is_utc(
    tmp_path, monkeypatch
):
    """Admission log timestamp is UTC (PR #590 Copilot review)."""
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, _record = _prepare_record_and_candidate(tmp_path, monkeypatch)
    result = promote_benchmark_candidate(
        candidate_path, repo_root=tmp_path, dry_run=True
    )
    timestamp = result["admission_timestamp"]
    assert isinstance(timestamp, str)
    # UTC ISO format ends in `+00:00` or `Z`.
    assert timestamp.endswith("+00:00") or timestamp.endswith("Z")


def test_promotion_candidate_code_hash_matches_benchmark_record_artifact_hash(
    tmp_path, monkeypatch
):
    """Candidate's `code_hash` must match the benchmark record's
    `generated_artifact.code_hash` exactly so admission's hash check accepts
    the genuine candidate.  The benchmark record uses
    `sha256(read_bytes()).hexdigest()`; the candidate emitter must use the
    same scheme (preferring the record's hash directly when present).
    (PR #590 Copilot review.)"""
    from trellis.agent.knowledge.promotion import record_benchmark_promotion_candidate

    candidate_path, record = _prepare_record_and_candidate(
        tmp_path, monkeypatch, task_id="F009"
    )
    # _prepare_record_and_candidate populates the record with a full SHA-256
    # over the source bytes, matching what `_generated_artifact_from_result`
    # produces in the live runner.
    record_hash = record["generated_artifact"]["code_hash"]
    candidate = yaml.safe_load(candidate_path.read_text())
    assert candidate["code_hash"] == record_hash, (
        "candidate hash diverged from benchmark record hash; admission would reject"
    )


def test_promote_benchmark_candidate_accepts_full_hex_hash_without_strip(
    tmp_path, monkeypatch
):
    """Admission's `computed_hash` must use the same byte-faithful scheme
    as candidate emission so trailing whitespace cannot cause a hash
    mismatch on a genuine candidate.  (PR #590 Copilot review.)"""
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, _record = _prepare_record_and_candidate(
        tmp_path, monkeypatch, task_id="F009"
    )
    candidate = yaml.safe_load(candidate_path.read_text())
    # Force a code body with trailing newline (the common case): the
    # admission flow must still treat the candidate hash as equivalent.
    code_with_trailing_newline = candidate["code"]
    if not code_with_trailing_newline.endswith("\n"):
        code_with_trailing_newline += "\n"
    candidate["code"] = code_with_trailing_newline
    candidate["code_hash"] = hashlib.sha256(
        code_with_trailing_newline.encode("utf-8")
    ).hexdigest()
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    result = promote_benchmark_candidate(
        candidate_path, repo_root=tmp_path, dry_run=True
    )
    assert result["status"] == "would_promote"


def test_record_benchmark_promotion_candidate_emits_module_name_field(
    tmp_path, monkeypatch
):
    """Benchmark candidates should write `module_name` (the unambiguous
    import-name field) in addition to legacy `module_path` so consumers
    don't have to overload `module_path`'s meaning.  (PR #590 round-3.)"""
    candidate_path, record = _prepare_record_and_candidate(
        tmp_path, monkeypatch, task_id="F009"
    )
    candidate = yaml.safe_load(candidate_path.read_text())
    assert "module_name" in candidate
    assert candidate["module_name"] == record["generated_artifact"]["module_name"]
    # `module_path` retained for backward compatibility.
    assert candidate.get("module_path") == candidate["module_name"]


def test_promote_benchmark_candidate_prefers_module_name_over_module_path(
    tmp_path, monkeypatch
):
    """Admission reads `module_name` first; missing `module_path` is fine."""
    from trellis.agent.knowledge.promotion import promote_benchmark_candidate

    candidate_path, _record = _prepare_record_and_candidate(
        tmp_path, monkeypatch, task_id="F009"
    )
    candidate = yaml.safe_load(candidate_path.read_text())
    candidate.pop("module_path", None)
    # module_name must remain present and unambiguous.
    assert "module_name" in candidate
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False))

    result = promote_benchmark_candidate(
        candidate_path, repo_root=tmp_path, dry_run=True
    )
    assert result["status"] == "would_promote"
