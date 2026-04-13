from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_binding_first_exotic_closeout.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_binding_first_exotic_closeout",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_binding_first_exotic_closeout_writes_outputs(tmp_path):
    module = _load_module()

    report_one = tmp_path / "qua808_report.json"
    report_two = tmp_path / "qua809_report.json"
    report_one.write_text(
        json.dumps(
            {
                "cohort": "event_control_schedule",
                "status": "failed_gate",
                "task_ids": ["T105"],
                "report_json_path": str(report_one),
                "report_md_path": str(tmp_path / "qua808_report.md"),
                "raw_results_path": str(tmp_path / "qua808_results.json"),
                "proof_summary": {
                    "totals": {"tasks": 1, "passed_gate": 1, "failed_gate": 0, "proved": 1, "honest_block": 0},
                    "failure_buckets": {"success": 1},
                    "by_task": {
                        "T105": {
                            "title": "Quanto option",
                            "cohort": "event_control_schedule",
                            "outcome_class": "proved",
                            "passed_gate": True,
                            "failure_bucket": "success",
                            "comparison_status": "passed",
                            "binding_ids": ["binding.quanto"],
                            "route_ids": [],
                            "first_pass": True,
                            "attempts_to_success": 1,
                            "retry_taxonomy": [],
                            "elapsed_seconds": 10.0,
                            "token_usage": {"total_tokens": 100, "call_count": 2},
                        }
                    },
                },
                "task_summary": {},
            }
        )
    )
    report_two.write_text(
        json.dumps(
            {
                "cohort": "basket_credit_loss",
                "status": "failed_gate",
                "task_ids": ["T50"],
                "report_json_path": str(report_two),
                "report_md_path": str(tmp_path / "qua809_report.md"),
                "raw_results_path": str(tmp_path / "qua809_results.json"),
                "proof_summary": {
                    "totals": {"tasks": 1, "passed_gate": 0, "failed_gate": 1, "proved": 1, "honest_block": 0},
                    "failure_buckets": {"comparison_insufficient_results": 1},
                    "by_task": {
                        "T50": {
                            "title": "Nth-to-default",
                            "cohort": "basket_credit_loss",
                            "outcome_class": "proved",
                            "passed_gate": False,
                            "failure_bucket": "comparison_insufficient_results",
                            "comparison_status": "insufficient_results",
                            "binding_ids": ["binding.ntd"],
                            "route_ids": ["unknown"],
                            "first_pass": False,
                            "attempts_to_success": 3,
                            "retry_taxonomy": ["code_generation"],
                            "elapsed_seconds": 20.0,
                            "token_usage": {"total_tokens": 200, "call_count": 3},
                        }
                    },
                },
                "task_summary": {},
            }
        )
    )

    output_json = tmp_path / "closeout.json"
    output_md = tmp_path / "closeout.md"
    exit_code = module.run_binding_first_exotic_closeout(
        report_json_paths=[report_one, report_two],
        output_json_path=output_json,
        output_md_path=output_md,
    )

    assert exit_code == 0
    summary = json.loads(output_json.read_text())
    assert summary["totals"]["tasks"] == 2
    assert summary["totals"]["passed_gate"] == 1
    assert summary["unknown_route_tasks"] == ["T50"]
    assert "Binding-First Exotic Program Closeout" in output_md.read_text()
