"""Tests for the fresh-generated FinancePy pilot parity scorecard (QUA-868).

The scorecard consumes append-only history records from
``task_runs/financepy_benchmarks/history``, restricts them to the pilot
subset (F001/F002/F003/F007/F009/F012), verifies each run's provenance is
fresh-generated (QUA-866), and emits a timestamped JSON+Markdown artifact
summarizing parity outcomes.  Residual misses escalate as shared
root-cause follow-ons, not as task-specific patches.
"""

from __future__ import annotations

import json
from pathlib import Path

from trellis.agent.financepy_benchmark import (
    FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS,
)
from trellis.agent.pilot_parity_scorecard import (
    PILOT_SCORECARD_TASK_IDS,
    build_pilot_parity_scorecard,
    load_pilot_benchmark_records,
    render_pilot_parity_scorecard,
    save_pilot_parity_scorecard,
)


def _write_history_record(root: Path, record: dict) -> Path:
    task_id = record["task_id"]
    run_id = record["run_id"]
    history_dir = root / "history" / task_id
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def _fresh_record(task_id: str, *, passed: bool = True, run_id_suffix: str = "A") -> dict:
    return {
        "task_id": task_id,
        "title": f"{task_id} pilot",
        "instrument_type": "equity_vanilla",
        "preferred_method": "analytical",
        "benchmark_execution_policy": "fresh_generated",
        "benchmark_campaign_id": "pilot",
        "execution_mode": "cold_agent_plus_financepy_reference",
        "run_id": f"{task_id}_20260415T{run_id_suffix}",
        "run_started_at": f"2026-04-15T12:00:{run_id_suffix[-2:]}Z",
        "run_completed_at": f"2026-04-15T12:05:{run_id_suffix[-2:]}Z",
        "git_sha": "deadbeef",
        "knowledge_revision": "cafebabe",
        "status": "priced",
        "cold_agent_elapsed_seconds": 1.5,
        "cold_agent_token_usage": {"total_tokens": 100},
        "comparison_summary": {
            "status": "passed" if passed else "failed",
            "tolerance_pct": 2.0,
            "compared_outputs": ["price"],
            "output_deviation_pct": {"price": 0.05 if passed else 3.5},
        },
        "generated_artifact": {
            "module_name": (
                f"trellis_benchmarks._fresh.{task_id.lower()}.analytical.europeanoptionanalytical"
            ),
            "class_name": "EuropeanOptionPayoff",
            "module_path": (
                f"task_runs/financepy_benchmarks/generated/{task_id.lower()}/analytical/"
                "europeanoptionanalytical.py"
            ),
            "is_fresh_build": True,
        },
        "fresh_generated_boundary": {
            "status": "enforced",
            "policy": "fresh_generated",
            "task_id": task_id,
            "reason": "fresh-generated artifact verified outside admitted _agent tree",
            "violations": [],
            "generated_module": (
                f"trellis_benchmarks._fresh.{task_id.lower()}.analytical.europeanoptionanalytical"
            ),
            "generated_module_path": (
                f"task_runs/financepy_benchmarks/generated/{task_id.lower()}/analytical/"
                "europeanoptionanalytical.py"
            ),
            "inspected_imports": [],
        },
    }


def _violated_record(task_id: str, reason: str = "boundary breach") -> dict:
    record = _fresh_record(task_id, run_id_suffix="B")
    record["status"] = "failed"
    record["comparison_summary"] = {
        "status": "benchmark_boundary_violation",
        "boundary_error": reason,
        "boundary_violations": [reason],
    }
    record["fresh_generated_boundary"] = {
        "status": "violated",
        "policy": "fresh_generated",
        "task_id": task_id,
        "reason": reason,
        "violations": [reason],
        "generated_module": "trellis.instruments._agent.europeanoptionanalytical",
        "generated_module_path": "trellis/instruments/_agent/europeanoptionanalytical.py",
        "inspected_imports": [],
    }
    record["generated_artifact"] = {
        "module_name": "trellis.instruments._agent.europeanoptionanalytical",
        "is_fresh_build": False,
    }
    return record


def test_pilot_scorecard_task_ids_match_financepy_benchmark_pilot_subset():
    assert set(PILOT_SCORECARD_TASK_IDS) == set(FRESH_GENERATED_FINANCEPY_PILOT_TASK_IDS)


def test_load_pilot_benchmark_records_filters_to_pilot_subset(tmp_path):
    for task_id in ("F001", "F002", "F010"):
        _write_history_record(tmp_path, _fresh_record(task_id))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)

    assert {record["task_id"] for record in records} == {"F001", "F002"}


def test_build_pilot_parity_scorecard_emits_per_task_rows_and_summary(tmp_path):
    for task_id in sorted(PILOT_SCORECARD_TASK_IDS):
        _write_history_record(tmp_path, _fresh_record(task_id))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    assert scorecard["scorecard_name"] == "financepy_pilot"
    assert scorecard["pilot_task_ids"] == sorted(PILOT_SCORECARD_TASK_IDS)
    assert scorecard["task_count"] == len(PILOT_SCORECARD_TASK_IDS)
    assert scorecard["pilot_summary"]["fresh_generated_enforced_count"] == len(
        PILOT_SCORECARD_TASK_IDS
    )
    assert scorecard["pilot_summary"]["boundary_violation_count"] == 0
    assert scorecard["pilot_summary"]["missing_run_count"] == 0
    assert scorecard["pilot_summary"]["latest_pass_count"] == len(PILOT_SCORECARD_TASK_IDS)
    assert scorecard["residual_misses"] == []
    for task in scorecard["tasks"]:
        assert task["fresh_generated_boundary_status"] == "enforced"
        assert task["latest"]["passed"] is True


def test_build_pilot_parity_scorecard_flags_missing_pilot_tasks(tmp_path):
    for task_id in ("F001", "F002", "F003"):
        _write_history_record(tmp_path, _fresh_record(task_id))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    assert scorecard["pilot_summary"]["missing_run_count"] == 3
    missing_ids = {
        miss["task_id"]
        for miss in scorecard["residual_misses"]
        if miss["category"] == "missing_run"
    }
    assert missing_ids == {"F007", "F009", "F012"}


def test_build_pilot_parity_scorecard_escalates_boundary_violations(tmp_path):
    for task_id in ("F001", "F002", "F003", "F007", "F009"):
        _write_history_record(tmp_path, _fresh_record(task_id))
    _write_history_record(tmp_path, _violated_record("F012", reason="admitted _agent import"))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    assert scorecard["pilot_summary"]["boundary_violation_count"] == 1
    assert scorecard["pilot_summary"]["fresh_generated_enforced_count"] == 5
    misses = {
        miss["task_id"]: miss
        for miss in scorecard["residual_misses"]
    }
    assert "F012" in misses
    assert misses["F012"]["category"] == "boundary_violation"
    assert "admitted _agent import" in misses["F012"]["reason"]


def test_build_pilot_parity_scorecard_tracks_parity_failures_as_residual_misses(tmp_path):
    for task_id in ("F001", "F002", "F003", "F007", "F012"):
        _write_history_record(tmp_path, _fresh_record(task_id))
    _write_history_record(tmp_path, _fresh_record("F009", passed=False))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    assert scorecard["pilot_summary"]["fresh_generated_enforced_count"] == 6
    assert scorecard["pilot_summary"]["boundary_violation_count"] == 0
    misses = {miss["task_id"]: miss for miss in scorecard["residual_misses"]}
    assert "F009" in misses
    assert misses["F009"]["category"] == "parity_failure"


def test_render_pilot_parity_scorecard_includes_pilot_summary_block():
    scorecard = {
        "scorecard_name": "financepy_pilot",
        "created_at": "2026-04-15T12:00:00Z",
        "pilot_task_ids": sorted(PILOT_SCORECARD_TASK_IDS),
        "task_count": 6,
        "run_count": 6,
        "pilot_summary": {
            "fresh_generated_enforced_count": 5,
            "boundary_violation_count": 1,
            "missing_run_count": 0,
            "latest_pass_count": 5,
        },
        "residual_misses": [
            {
                "task_id": "F012",
                "category": "boundary_violation",
                "reason": "admitted _agent import",
            }
        ],
        "tasks": [],
        "notes": [],
    }

    text = render_pilot_parity_scorecard(scorecard)
    assert "Fresh-generated enforced" in text
    assert "Boundary violations" in text
    assert "Residual Misses" in text
    assert "F012" in text


def test_save_pilot_parity_scorecard_writes_timestamped_stem(tmp_path):
    scorecard = {
        "scorecard_name": "financepy_pilot",
        "created_at": "2026-04-15T12:00:00Z",
        "pilot_task_ids": sorted(PILOT_SCORECARD_TASK_IDS),
        "task_count": 0,
        "run_count": 0,
        "pilot_summary": {
            "fresh_generated_enforced_count": 0,
            "boundary_violation_count": 0,
            "missing_run_count": 6,
            "latest_pass_count": 0,
        },
        "residual_misses": [],
        "tasks": [],
        "notes": [],
    }

    artifacts = save_pilot_parity_scorecard(
        scorecard,
        reports_root=tmp_path,
        stem="financepy_pilot",
        timestamp="20260415T120000Z",
    )

    assert artifacts.json_path.exists()
    assert artifacts.text_path.exists()
    assert "20260415T120000Z" in artifacts.json_path.name
    assert artifacts.json_path.suffix == ".json"
    assert artifacts.text_path.suffix == ".md"


def test_pilot_parity_scorecard_cli_writes_artifacts(tmp_path, monkeypatch, capsys):
    from scripts import pilot_parity_scorecard as cli

    benchmark_root = tmp_path / "task_runs" / "financepy_benchmarks"
    for task_id in sorted(PILOT_SCORECARD_TASK_IDS):
        _write_history_record(benchmark_root, _fresh_record(task_id))

    reports_root = tmp_path / "reports"

    exit_code = cli.main(
        [
            "--benchmark-root",
            str(benchmark_root),
            "--reports-root",
            str(reports_root),
            "--scorecard-name",
            "financepy_pilot",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["scorecard_json"]).exists()
    assert Path(payload["scorecard_md"]).exists()
    assert payload["pilot_summary"]["fresh_generated_enforced_count"] == len(
        PILOT_SCORECARD_TASK_IDS
    )
