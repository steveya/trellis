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
            "deviation_median_pct": 0.0,
            "deviation_outlier_count": 0,
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
            "deviation_median_pct": 0.0,
            "deviation_outlier_count": 0,
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


def _record_with_deviation(task_id: str, deviation_pct: float) -> dict:
    record = _fresh_record(task_id)
    record["comparison_summary"] = dict(record["comparison_summary"])
    record["comparison_summary"]["output_deviation_pct"] = {"price": deviation_pct}
    return record


def test_build_pilot_parity_scorecard_flags_deviation_outlier(tmp_path):
    devs = {
        "F001": 0.005,
        "F002": 0.0002,
        "F003": 0.001,
        "F007": 0.93,
        "F009": 0.003,
        "F012": 0.10,
    }
    for task_id, dev in devs.items():
        _write_history_record(tmp_path, _record_with_deviation(task_id, dev))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    summary = scorecard["pilot_summary"]
    # Both F007 (0.93%) and F012 (0.10%) are >10x the pilot median (~0.004%)
    # and above the suppression floor.  Both deserve a human look before
    # promotion -- exactly the F007-shape signal QUA-873 was built to surface.
    assert summary["deviation_outlier_count"] == 2
    assert 0.0 < summary["deviation_median_pct"] < 0.93

    by_task = {task["task_id"]: task for task in scorecard["tasks"]}
    assert by_task["F007"]["outlier_flag"] is True
    assert by_task["F012"]["outlier_flag"] is True
    assert by_task["F007"]["max_output_deviation_pct"] == 0.93
    assert by_task["F007"]["deviation_vs_pilot_median"] is not None
    assert by_task["F007"]["deviation_vs_pilot_median"] >= 10.0
    # Tasks at or near the median are not flagged.
    for other in ("F001", "F002", "F003", "F009"):
        assert by_task[other]["outlier_flag"] is False

    outlier_misses = [
        miss
        for miss in scorecard["residual_misses"]
        if miss["category"] == "deviation_outlier"
    ]
    assert {miss["task_id"] for miss in outlier_misses} == {"F007", "F012"}


def test_build_pilot_parity_scorecard_no_outliers_when_all_within_ratio(tmp_path):
    devs = {
        "F001": 0.005,
        "F002": 0.004,
        "F003": 0.006,
        "F007": 0.007,
        "F009": 0.003,
        "F012": 0.005,
    }
    for task_id, dev in devs.items():
        _write_history_record(tmp_path, _record_with_deviation(task_id, dev))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    summary = scorecard["pilot_summary"]
    assert summary["deviation_outlier_count"] == 0
    assert all(task["outlier_flag"] is False for task in scorecard["tasks"])


def test_render_pilot_parity_scorecard_emits_deviation_outlier_section(tmp_path):
    devs = {
        "F001": 0.005,
        "F002": 0.0002,
        "F003": 0.001,
        "F007": 0.93,
        "F009": 0.003,
        "F012": 0.10,
    }
    for task_id, dev in devs.items():
        _write_history_record(tmp_path, _record_with_deviation(task_id, dev))

    records = load_pilot_benchmark_records(benchmark_root=tmp_path)
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=records,
    )

    text = render_pilot_parity_scorecard(scorecard)
    assert "## Deviation Outliers" in text
    assert "F007" in text
    assert "Deviation median" in text


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


def _record_with_greek_coverage(task_id, *, trellis_greeks, financepy_greeks, greek_parity):
    """Build a record whose comparison_summary reports QUA-861 Greek fields."""
    record = _fresh_record(task_id)
    missing_trellis = [g for g in financepy_greeks if g not in trellis_greeks]
    missing_financepy = [g for g in trellis_greeks if g not in financepy_greeks]
    compared = [g for g in trellis_greeks if g in financepy_greeks]
    record["comparison_summary"] = dict(record["comparison_summary"])
    record["comparison_summary"].update(
        {
            "missing_trellis_outputs": missing_trellis,
            "missing_financepy_outputs": missing_financepy,
            "greek_coverage": {
                "trellis_greek_count": len(trellis_greeks),
                "financepy_greek_count": len(financepy_greeks),
                "compared_greek_count": len(compared),
            },
            "greek_parity": greek_parity,
            "greek_failures": [],
        }
    )
    return record


def test_pilot_scorecard_aggregates_greek_coverage_across_tasks(tmp_path):
    # Two tasks expose Trellis Greeks, four do not; five have financepy Greeks.
    records = {
        "F001": _record_with_greek_coverage(
            "F001",
            trellis_greeks=["delta", "vega"],
            financepy_greeks=["delta", "vega", "gamma"],
            greek_parity="passed",
        ),
        "F002": _record_with_greek_coverage(
            "F002",
            trellis_greeks=["delta"],
            financepy_greeks=["delta"],
            greek_parity="passed",
        ),
        "F003": _record_with_greek_coverage(
            "F003",
            trellis_greeks=[],
            financepy_greeks=["delta"],
            greek_parity="not_applicable",
        ),
        "F007": _record_with_greek_coverage(
            "F007",
            trellis_greeks=[],
            financepy_greeks=[],
            greek_parity="not_applicable",
        ),
        "F009": _record_with_greek_coverage(
            "F009",
            trellis_greeks=[],
            financepy_greeks=["delta", "gamma"],
            greek_parity="not_applicable",
        ),
        "F012": _record_with_greek_coverage(
            "F012",
            trellis_greeks=[],
            financepy_greeks=[],
            greek_parity="not_applicable",
        ),
    }
    for rec in records.values():
        _write_history_record(tmp_path, rec)

    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=load_pilot_benchmark_records(benchmark_root=tmp_path),
    )
    summary = scorecard["pilot_summary"]
    assert summary["tasks_with_trellis_greek_coverage"] == 2
    assert summary["tasks_with_greek_overlap"] == 2
    assert summary["greek_parity_passed_count"] == 2
    assert summary["greek_parity_failed_count"] == 0

    # Tasks where financepy declared Greeks but Trellis emitted none become
    # residual misses with category `missing_greek_coverage`.
    missing_coverage_misses = {
        miss["task_id"]: miss
        for miss in scorecard["residual_misses"]
        if miss["category"] == "missing_greek_coverage"
    }
    assert set(missing_coverage_misses) == {"F003", "F009"}


def test_pilot_scorecard_does_not_flag_missing_greeks_when_binding_has_no_greek_overlap(tmp_path):
    for task_id in sorted(PILOT_SCORECARD_TASK_IDS):
        _write_history_record(
            tmp_path,
            _record_with_greek_coverage(
                task_id,
                trellis_greeks=[],
                financepy_greeks=[],
                greek_parity="not_applicable",
            ),
        )
    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=load_pilot_benchmark_records(benchmark_root=tmp_path),
    )
    assert scorecard["pilot_summary"]["tasks_with_trellis_greek_coverage"] == 0
    assert [
        miss for miss in scorecard["residual_misses"]
        if miss["category"] == "missing_greek_coverage"
    ] == []


def test_pilot_scorecard_render_surfaces_greek_coverage_fields(tmp_path):
    for task_id in sorted(PILOT_SCORECARD_TASK_IDS):
        record = _record_with_greek_coverage(
            task_id,
            trellis_greeks=["delta"] if task_id == "F001" else [],
            financepy_greeks=["delta", "gamma"] if task_id == "F001" else [],
            greek_parity="passed" if task_id == "F001" else "not_applicable",
        )
        _write_history_record(tmp_path, record)

    scorecard = build_pilot_parity_scorecard(
        scorecard_name="financepy_pilot",
        benchmark_runs=load_pilot_benchmark_records(benchmark_root=tmp_path),
    )
    text = render_pilot_parity_scorecard(scorecard)
    assert "Tasks with Trellis Greek coverage" in text
    assert "Tasks with Greek overlap" in text
    assert "Greek parity passed" in text
    assert "Greek coverage: trellis=" in text
    # F001 is the only task with a compared Greek; its Greek parity should be
    # surfaced per-task.
    assert "Greek parity: `passed`" in text
