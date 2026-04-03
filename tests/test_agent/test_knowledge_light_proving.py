from __future__ import annotations

import importlib.util
from pathlib import Path


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
