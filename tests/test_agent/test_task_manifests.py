from __future__ import annotations

from pathlib import Path

import yaml


def test_load_pricing_tasks_aggregates_new_corpora_and_preserves_legacy_ids():
    from trellis.agent.task_manifests import load_pricing_tasks

    tasks = load_pricing_tasks(root=Path(__file__).resolve().parents[2])
    task_ids = {task["id"] for task in tasks}

    assert "F001" in task_ids
    assert "P001" in task_ids
    assert "T01" in task_ids
    assert "E21" in task_ids


def test_load_negative_tasks_reads_dedicated_negative_corpus():
    from trellis.agent.task_manifests import load_negative_tasks

    tasks = load_negative_tasks(root=Path(__file__).resolve().parents[2])
    task_ids = {task["id"] for task in tasks}

    assert {"N001", "N002", "N003"} <= task_ids


def test_load_task_manifest_materializes_market_from_scenario(tmp_path):
    from trellis.agent.task_manifests import load_task_manifest

    (tmp_path / "MARKET_SCENARIOS.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "scenarios": {
                    "flat_case": {
                        "source": "mock",
                        "as_of": "2024-11-15",
                        "selected_components": {
                            "discount_curve": "usd_ois",
                            "vol_surface": "flat_20pct",
                        },
                    }
                },
            },
            sort_keys=False,
        )
    )
    (tmp_path / "TASKS_BENCHMARK_FINANCEPY.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "tasks": [
                    {
                        "id": "F900",
                        "title": "Parity task",
                        "status": "pending",
                        "market_scenario_id": "flat_case",
                    }
                ],
            },
            sort_keys=False,
        )
    )

    tasks = load_task_manifest("TASKS_BENCHMARK_FINANCEPY.yaml", root=tmp_path)
    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_definition_version"] == 3
    assert task["task_corpus"] == "benchmark_financepy"
    assert task["market"] == {
        "source": "mock",
        "as_of": "2024-11-15",
        "discount_curve": "usd_ois",
        "vol_surface": "flat_20pct",
    }
