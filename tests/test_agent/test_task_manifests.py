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
                "version": 2,
                "scenarios": {
                    "flat_case": {
                        "source": "mock",
                        "as_of": "2024-11-15",
                        "description": "Flat single-asset test scenario.",
                        "selected_components": {
                            "discount_curve": "usd_ois",
                            "vol_surface": "spx_heston_implied_vol",
                        },
                        "constructor": {
                            "kind": "single_asset_equity",
                            "valuation_date": "2024-11-15",
                            "domestic_rate": 0.05,
                            "black_vol": 0.2,
                            "underlier": {
                                "name": "SPX",
                                "spot": 100.0,
                                "volatility": 0.2,
                                "carry": {
                                    "rate": 0.0,
                                    "curve_name": "SPX-DISC",
                                },
                            },
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
    assert task["market"]["source"] == "mock"
    assert task["market"]["as_of"] == "2024-11-15"
    assert task["market"]["discount_curve"] == "usd_ois"
    assert task["market"]["vol_surface"] == "spx_heston_implied_vol"
    assert task["market"]["scenario_constructor_kind"] == "single_asset_equity"
    assert task["market"]["scenario_schema_version"] == 2
    assert task["market"]["benchmark_inputs"]["stock_price"] == 100.0
    assert task["market"]["benchmark_inputs"]["domestic_rate"] == 0.05
    assert task["market"]["scenario_digest"]


def test_filter_loaded_tasks_supports_corpus_and_exact_id_selection():
    from trellis.agent.task_manifests import filter_loaded_tasks

    tasks = [
        {"id": "F001", "status": "pending", "task_corpus": "benchmark_financepy"},
        {"id": "P004", "status": "pending", "task_corpus": "extension"},
        {"id": "P007", "status": "done", "task_corpus": "extension"},
    ]

    selected = filter_loaded_tasks(
        tasks,
        status="all",
        corpora=("extension",),
        task_ids=("P007", "P004"),
    )

    assert [task["id"] for task in selected] == ["P007", "P004"]


def test_filter_loaded_tasks_respects_status_before_exact_id_selection():
    from trellis.agent.task_manifests import filter_loaded_tasks

    tasks = [
        {"id": "P004", "status": "pending", "task_corpus": "extension"},
        {"id": "P007", "status": "done", "task_corpus": "extension"},
    ]

    selected = filter_loaded_tasks(
        tasks,
        status="pending",
        task_ids=("P007", "P004"),
    )

    assert [task["id"] for task in selected] == ["P004"]
