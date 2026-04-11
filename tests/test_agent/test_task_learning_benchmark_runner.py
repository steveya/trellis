from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_task_learning_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_task_learning_benchmark",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_learning_benchmark_writes_pass_outputs_and_report(tmp_path, monkeypatch, capsys):
    module = _load_module()

    fake_results = iter(
        [
            {
                "task_id": "T13",
                "success": False,
                "attempts": 2,
                "elapsed_seconds": 9.0,
                "error": "build failed",
                "knowledge_gaps": ["missing_cookbook"],
                "knowledge_summary": {"lesson_ids": []},
                "reflection": {},
                "token_usage_summary": {"total_tokens": 200},
            },
            {
                "task_id": "T14",
                "success": False,
                "attempts": 1,
                "elapsed_seconds": 6.0,
                "error": "MissingCapabilityError: missing market data discount_curve",
                "knowledge_gaps": [],
                "knowledge_summary": {"lesson_ids": []},
                "reflection": {},
                "token_usage_summary": {"total_tokens": 110},
            },
            {
                "task_id": "T13",
                "success": True,
                "attempts": 1,
                "elapsed_seconds": 4.0,
                "knowledge_gaps": [],
                "knowledge_summary": {"lesson_ids": ["lesson-13"]},
                "reflection": {
                    "lesson_captured": ["lesson-13"],
                    "cookbook_enriched": True,
                },
                "token_usage_summary": {"total_tokens": 90},
            },
            {
                "task_id": "T14",
                "success": False,
                "attempts": 1,
                "elapsed_seconds": 5.0,
                "error": "MissingCapabilityError: missing market data discount_curve",
                "knowledge_gaps": [],
                "knowledge_summary": {"lesson_ids": []},
                "reflection": {},
                "token_usage_summary": {"total_tokens": 100},
            },
        ]
    )

    monkeypatch.setattr(module, "build_market_state", lambda: object())
    monkeypatch.setattr(module, "run_task", lambda *args, **kwargs: next(fake_results))
    monkeypatch.setattr(module, "_git_revision", lambda: "abc1234")

    tasks = [
        {"id": "T13", "title": "European call PDE", "status": "pending"},
        {"id": "T14", "title": "American put", "status": "pending"},
    ]

    artifacts = module.run_learning_benchmark(
        tasks,
        benchmark_name="non_canary_task_learning",
        output_root=tmp_path,
        passes=2,
        model="test-model",
        validation="standard",
        fresh_build=True,
        knowledge_light=False,
    )

    report = json.loads(artifacts["report_json_path"].read_text())
    pass_one_results = json.loads((tmp_path / "raw" / "non_canary_task_learning_pass_1.json").read_text())
    pass_two_summary = json.loads((tmp_path / "raw" / "non_canary_task_learning_pass_2_summary.json").read_text())
    stdout = capsys.readouterr().out

    assert len(pass_one_results) == 2
    assert report["git_revision"] == "abc1234"
    assert report["passes"][0]["fresh_build"] is True
    assert report["passes"][1]["summary"]["first_pass"]["first_pass_successes"] == 1
    assert pass_two_summary["totals"]["successes"] == 1
    assert "Pass 1/2" in stdout
    assert "Pass 2/2" in stdout
    assert "Saved learning benchmark report" in stdout
