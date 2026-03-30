from __future__ import annotations

import json

import scripts.remediate as remediate


def test_load_all_results_handles_list_and_summary_index_formats(tmp_path, monkeypatch):
    list_path = tmp_path / "task_results_a.json"
    index_path = tmp_path / "task_results_b.json"
    summary_path = tmp_path / "task_results_c.json"

    with open(list_path, "w") as fh:
        json.dump(
            [
                {"task_id": "T001", "success": True, "failures": []},
                {"task_id": "T002", "success": False, "failures": ["boom"]},
            ],
            fh,
        )

    with open(index_path, "w") as fh:
        json.dump(
            {
                "T100": {
                    "result": {"task_id": "T100", "success": True, "failures": []}
                },
                "T101": {
                    "result": {"task_id": "T101", "success": False, "failures": ["bad"]}
                },
            },
            fh,
        )

    with open(summary_path, "w") as fh:
        json.dump({"totals": {"success": 2, "failure": 1}}, fh)

    monkeypatch.setattr(remediate, "ROOT", tmp_path)

    results = remediate.load_all_results()

    assert [r["task_id"] for r in results] == ["T001", "T002", "T100", "T101"]
    assert all(isinstance(r, dict) for r in results)
    assert all("success" in r for r in results)


def test_analyze_failures_uses_nested_method_failure_text():
    nested_result = {
        "task_id": "T900",
        "success": False,
        "method_results": {
            "psor_pde": {
                "success": False,
                "failures": [
                    "OpenAI json request failed after 3 attempts for model 'gpt-5-mini': TimeoutError: OpenAI request exceeded 30.0s",
                ],
            },
            "lsm_mc": {
                "success": False,
                "failures": ["name 'AMERICAN' is not defined"],
            },
        },
    }

    categories = remediate.analyze_failures([nested_result])

    assert [r["task_id"] for r in categories["timeout"]] == ["T900"]


def test_analyze_failures_buckets_missing_capabilities_as_market_data():
    result = {
        "task_id": "T901",
        "success": False,
        "failures": [
            "Cannot build payoff: missing capabilities ['spot', 'discount_curve']. Available: ['black_vol']",
        ],
    }

    categories = remediate.analyze_failures([result])

    assert [r["task_id"] for r in categories["missing_market_data"]] == ["T901"]
