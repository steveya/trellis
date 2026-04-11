from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_knowledge_light_proving.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_knowledge_light_proving",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_proving_tasks_contains_expected_case_ids():
    module = _load_module()

    tasks = module.build_proving_tasks()

    assert [task["id"] for task in tasks] == ["KL01", "KL02", "KL03"]
    assert tasks[0]["construct"] == ["analytical", "monte_carlo"]
    assert tasks[1]["construct"] == ["rate_tree"]
    assert tasks[2]["construct"] == ["monte_carlo", "credit"]


def test_select_cases_filters_requested_ids():
    module = _load_module()
    tasks = module.build_proving_tasks()

    selected = module._select_cases(tasks, ["KL02"])

    assert [task["id"] for task in selected] == ["KL02"]


def test_run_proving_set_writes_first_pass_metrics_and_retry_taxonomy(tmp_path, monkeypatch, capsys):
    module = _load_module()

    def _write_platform_trace(name: str, *events: dict[str, object]) -> str:
        trace_path = tmp_path / f"{name}.yaml"
        trace_path.write_text(
            yaml.safe_dump(
                {
                    "request_id": name,
                    "status": "succeeded",
                    "outcome": "build_completed",
                },
                sort_keys=False,
            )
        )
        events_path = trace_path.with_suffix(".events.ndjson")
        if events:
            events_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n"
            )
        return str(trace_path)

    fake_results = iter(
        [
            {
                "task_id": "KL01",
                "success": True,
                "attempts": 1,
                "platform_trace_path": _write_platform_trace("kl01"),
            },
            {
                "task_id": "KL02",
                "success": True,
                "attempts": 2,
                "platform_trace_path": _write_platform_trace(
                    "kl02",
                    {
                        "event": "builder_attempt_failed",
                        "status": "error",
                        "timestamp": "2026-04-10T15:00:00+00:00",
                        "details": {"reason": "validation"},
                    }
                ),
            },
        ]
    )

    monkeypatch.setattr(module, "build_market_state", lambda: object())
    monkeypatch.setattr(module, "run_task", lambda *args, **kwargs: next(fake_results))

    output_file = tmp_path / "knowledge_light_results.json"
    tasks = [
        {"id": "KL01", "title": "Case 1"},
        {"id": "KL02", "title": "Case 2"},
    ]

    summary = module.run_proving_set(
        tasks,
        str(output_file),
        model="test-model",
        validation="standard",
    )

    summary_path = tmp_path / "knowledge_light_results_summary.json"
    persisted = json.loads(summary_path.read_text())
    stdout = capsys.readouterr().out

    assert summary["first_pass"]["rate"] == 0.5
    assert summary["attempts_to_success"]["distribution"] == {"1": 1, "2": 1}
    assert summary["retry_taxonomy"]["by_stage"]["validation"]["task_ids"] == ["KL02"]
    assert persisted["first_pass"]["first_pass_successes"] == 1
    assert persisted["attempts_to_success"]["average"] == 1.5
    assert persisted["retry_taxonomy"]["recovered_successes"] == 1
    assert "First pass:" in stdout
    assert "Attempts to success:" in stdout
    assert "Retry taxonomy:" in stdout
