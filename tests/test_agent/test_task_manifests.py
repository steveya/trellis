from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml


def test_load_pricing_tasks_aggregates_new_corpora_and_preserves_legacy_ids():
    from trellis.agent.task_manifests import load_pricing_tasks

    tasks = load_pricing_tasks(root=Path(__file__).resolve().parents[2])
    task_ids = {task["id"] for task in tasks}

    assert "F001" in task_ids
    assert "P001" in task_ids
    assert "T01" in task_ids
    assert "E21" in task_ids
    assert "FPC001" in task_ids


def test_load_fpml_conformance_tasks_reads_dedicated_corpus():
    from trellis.agent.task_manifests import load_fpml_conformance_tasks

    tasks = load_fpml_conformance_tasks(root=Path(__file__).resolve().parents[2])
    task_ids = {task["id"] for task in tasks}

    assert {"FPC001", "FPC002", "FPC003", "FPC101", "FPC106"} <= task_ids


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


def test_load_task_manifest_materializes_one_named_proof_fixture(tmp_path):
    from trellis.agent.task_manifests import load_task_manifest

    fixture = {
        "instrument_type": "callable_bond",
        "benchmark_contract": {
            "product": "callable_bond",
            "notional": 100.0,
            "coupon": 0.05,
        },
    }
    (tmp_path / "TASKS_PROOF_LEGACY.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proof_fixtures": {"callable_fixed_v1": fixture},
                "tasks": [
                    {
                        "id": "T900",
                        "title": "Display-only title",
                        "status": "pending",
                        "proof_fixture_id": "callable_fixed_v1",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    task = load_task_manifest("TASKS_PROOF_LEGACY.yaml", root=tmp_path)[0]

    assert task["instrument_type"] == "callable_bond"
    assert task["benchmark_contract"] == fixture["benchmark_contract"]
    assert task["proof_fixture_id"] == "callable_fixed_v1"
    assert task["proof_fixture_schema_version"] == 1
    assert task["proof_fixture_digest"]


def test_named_proof_fixture_rejects_task_level_economic_overrides(tmp_path):
    from trellis.agent.task_manifests import load_task_manifest

    fixture = {
        "instrument_type": "callable_bond",
        "benchmark_contract": {
            "product": "callable_bond",
            "notional": 100.0,
            "coupon": 0.05,
        },
    }
    task = {
        "id": "T900",
        "title": "Display-only title",
        "status": "pending",
        "proof_fixture_id": "callable_fixed_v1",
        "benchmark_contract": deepcopy(fixture["benchmark_contract"]),
    }
    task["benchmark_contract"]["coupon"] = 0.06
    (tmp_path / "TASKS_PROOF_LEGACY.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proof_fixtures": {"callable_fixed_v1": fixture},
                "tasks": [task],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot override proof fixture"):
        load_task_manifest("TASKS_PROOF_LEGACY.yaml", root=tmp_path)


def test_named_proof_fixture_is_not_materialized_for_modern_corpora(tmp_path):
    from trellis.agent.task_manifests import load_task_manifest

    (tmp_path / "TASKS_EXTENSION.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proof_fixtures": {
                    "unvalidated_fixture": {
                        "benchmark_contract": {
                            "product": "callable_bond",
                            "coupon": 0.05,
                        }
                    }
                },
                "tasks": [
                    {
                        "id": "P900",
                        "title": "Modern schema boundary",
                        "status": "pending",
                        "proof_fixture_id": "unvalidated_fixture",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    task = load_task_manifest("TASKS_EXTENSION.yaml", root=tmp_path)[0]

    assert "benchmark_contract" not in task
    assert "proof_fixture_digest" not in task


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


def test_filter_loaded_tasks_supports_multi_character_id_prefix_ranges():
    from trellis.agent.task_manifests import filter_loaded_tasks

    tasks = [
        {"id": "FPC001", "status": "pending"},
        {"id": "FPC002", "status": "pending"},
        {"id": "FPC101", "status": "pending"},
        {"id": "F001", "status": "pending"},
    ]

    selected = filter_loaded_tasks(
        tasks,
        start_id="FPC001",
        end_id="FPC002",
    )

    assert [task["id"] for task in selected] == ["FPC001", "FPC002"]
