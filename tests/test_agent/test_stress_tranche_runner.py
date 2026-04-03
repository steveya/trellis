from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _task(task_id: str) -> dict[str, object]:
    return {
        "id": task_id,
        "title": f"{task_id} title",
        "cross_validate": {"internal": ["fft"], "analytical": "black_scholes"},
    }


def _manifest(task_id: str) -> dict[str, object]:
    return {
        task_id: {
            "outcome_class": "compare_ready",
            "required_mock_capabilities": ["discount_curve"],
            "comparison_targets": ["fft", "black_scholes"],
            "reference_target": "black_scholes",
        }
    }


def test_run_stress_tranche_blocks_on_preflight_and_writes_report(monkeypatch, tmp_path):
    import run_stress_tranche as runner

    task_id = "E21"
    monkeypatch.setattr(runner, "load_stress_task_manifest", lambda: _manifest(task_id))
    monkeypatch.setattr(runner, "_load_stress_tasks", lambda task_ids: {task_id: _task(task_id)})
    monkeypatch.setattr(
        runner,
        "grade_stress_task_preflight",
        lambda task, expectation: {
            "task_contract_alignment": runner.GradeResult(
                False,
                ("required mock capabilities drift: manifest=('discount_curve',) task_contract=('spot',)",),
            )
        },
    )
    monkeypatch.setattr(
        runner,
        "summarize_stress_preflight",
        lambda tasks, manifest=None: {
            "totals": {"tasks": 1, "passed": 0, "failed": 1},
            "failed_tasks": [task_id],
            "by_task": {
                task_id: {
                    "title": "E21 title",
                    "checks": {
                        "task_contract_alignment": {
                            "passed": False,
                            "details": ["required mock capabilities drift"],
                        }
                    },
                }
            },
        },
    )
    monkeypatch.setattr(runner, "render_stress_tranche_report", lambda report: "# report\n")

    called = {"run_task": 0}

    def _run_task(*args, **kwargs):
        called["run_task"] += 1
        return {}

    monkeypatch.setattr(runner, "run_task", _run_task)

    output_path = tmp_path / "stress.json"
    report_json_path = tmp_path / "stress_report.json"
    report_md_path = tmp_path / "stress_report.md"
    exit_code = runner.run_stress_tranche(
        model="gpt-5.4-mini",
        validation="standard",
        task_ids=[task_id],
        force_rebuild=False,
        preflight_only=False,
        output_path=output_path,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
    )

    assert exit_code == 1
    assert called["run_task"] == 0
    assert report_json_path.exists()
    assert report_md_path.exists()
    payload = json.loads(report_json_path.read_text())
    assert payload["status"] == "preflight_failed"
    assert payload["preflight_summary"]["failed_tasks"] == [task_id]


def test_run_stress_tranche_preflight_only_skips_live_execution(monkeypatch, tmp_path):
    import run_stress_tranche as runner

    task_id = "E21"
    monkeypatch.setattr(runner, "load_stress_task_manifest", lambda: _manifest(task_id))
    monkeypatch.setattr(runner, "_load_stress_tasks", lambda task_ids: {task_id: _task(task_id)})
    monkeypatch.setattr(
        runner,
        "grade_stress_task_preflight",
        lambda task, expectation: {
            "task_contract_alignment": runner.GradeResult(True, ()),
        },
    )
    monkeypatch.setattr(
        runner,
        "summarize_stress_preflight",
        lambda tasks, manifest=None: {
            "totals": {"tasks": 1, "passed": 1, "failed": 0},
            "failed_tasks": [],
            "by_task": {},
        },
    )
    monkeypatch.setattr(runner, "render_stress_tranche_report", lambda report: "# report\n")

    called = {"run_task": 0}
    monkeypatch.setattr(runner, "run_task", lambda *args, **kwargs: called.__setitem__("run_task", 1))

    exit_code = runner.run_stress_tranche(
        model="gpt-5.4-mini",
        validation="standard",
        task_ids=[task_id],
        force_rebuild=False,
        preflight_only=True,
        output_path=tmp_path / "stress.json",
        report_json_path=tmp_path / "stress_report.json",
        report_md_path=tmp_path / "stress_report.md",
    )

    assert exit_code == 0
    assert called["run_task"] == 0
    payload = json.loads((tmp_path / "stress_report.json").read_text())
    assert payload["status"] == "preflight_only"


def test_run_stress_tranche_passes_force_rebuild_to_live_runner(monkeypatch, tmp_path):
    import run_stress_tranche as runner

    task_id = "E21"
    monkeypatch.setattr(runner, "load_stress_task_manifest", lambda: _manifest(task_id))
    monkeypatch.setattr(runner, "_load_stress_tasks", lambda task_ids: {task_id: _task(task_id)})
    monkeypatch.setattr(
        runner,
        "grade_stress_task_preflight",
        lambda task, expectation: {
            "task_contract_alignment": runner.GradeResult(True, ()),
        },
    )
    monkeypatch.setattr(
        runner,
        "summarize_stress_preflight",
        lambda tasks, manifest=None: {
            "totals": {"tasks": 1, "passed": 1, "failed": 0},
            "failed_tasks": [],
            "by_task": {},
        },
    )
    monkeypatch.setattr(runner, "render_stress_tranche_report", lambda report: "# report\n")
    monkeypatch.setattr(runner, "build_market_state", lambda: object())
    monkeypatch.setattr(
        runner,
        "summarize_task_results",
        lambda results: {"totals": {"tasks": len(results), "successes": len(results), "failures": 0}},
    )
    monkeypatch.setattr(
        runner,
        "summarize_stress_tranche",
        lambda tasks, results, manifest=None: {
            "totals": {
                "tasks": 1,
                "passed_gate": 1,
                "failed_gate": 0,
                "compare_ready": 1,
                "honest_block": 0,
            },
            "by_task": {},
            "follow_on_candidates": [],
        },
    )

    observed: dict[str, object] = {}

    def _run_task(task, market_state, **kwargs):
        observed.update(kwargs)
        return {
            "task_id": task_id,
            "success": True,
            "cross_validation": {"status": "passed", "reference_target": "black_scholes"},
            "task_run_latest_path": "/tmp/task_runs/latest/E21.json",
            "task_diagnosis_latest_packet_path": "/tmp/task_runs/diagnostics/latest/E21.json",
            "task_diagnosis_latest_dossier_path": "/tmp/task_runs/diagnostics/latest/E21.md",
        }

    monkeypatch.setattr(runner, "run_task", _run_task)

    exit_code = runner.run_stress_tranche(
        model="gpt-5.4-mini",
        validation="standard",
        task_ids=[task_id],
        force_rebuild=True,
        preflight_only=False,
        output_path=tmp_path / "stress.json",
        report_json_path=tmp_path / "stress_report.json",
        report_md_path=tmp_path / "stress_report.md",
    )

    assert exit_code == 0
    assert observed["force_rebuild"] is True
    assert observed["fresh_build"] is True
    assert observed["validation"] == "standard"
