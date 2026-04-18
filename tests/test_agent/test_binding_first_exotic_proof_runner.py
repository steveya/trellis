from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_binding_first_exotic_proof.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_binding_first_exotic_proof",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_binding_first_exotic_proof_writes_report_and_summary(tmp_path, monkeypatch):
    module = _load_module()

    manifest = {
        "T105": {
            "cohort": "event_control_schedule",
            "outcome_class": "proved",
            "required_mock_capabilities": ["discount_curve", "forward_curve", "fx_rates", "spot"],
            "comparison_targets": ["quanto_bs", "mc_quanto"],
            "requires_binding_ids": True,
        },
        "E27": {
            "cohort": "event_control_schedule",
            "outcome_class": "honest_block",
            "required_mock_capabilities": ["discount_curve", "black_vol_surface", "model_parameters", "spot"],
            "comparison_targets": ["american_pathdep_pde", "american_pathdep_mc", "american_pathdep_fft"],
            "expected_blocker_categories": ["unsupported_composite"],
            "requires_binding_ids": False,
        },
    }

    def _task_run_record(path: Path, *, binding_id: str | None = None) -> str:
        telemetry = {"binding_observations": []}
        if binding_id:
            telemetry["binding_observations"].append(
                {
                    "binding_id": binding_id,
                    "binding_family": "analytical",
                    "route_id": "equity_quanto",
                    "route_family": "analytical",
                }
            )
        path.write_text(json.dumps({"telemetry": telemetry}))
        return str(path)

    t105_latest = _task_run_record(
        tmp_path / "T105_latest.json",
        binding_id="trellis.models.fx_vanilla.price_quanto_option_analytical_from_market_state",
    )
    e27_latest = _task_run_record(tmp_path / "E27_latest.json")

    tasks = {
        "T105": {
            "id": "T105",
            "title": "Quanto option: quanto-adjusted BS vs MC cross-currency",
            "cross_validate": {"internal": ["quanto_bs", "mc_quanto"]},
            "market_assertions": {
                "requires": ["discount_curve", "forward_curve", "fx_rates", "spot"]
            },
        },
        "E27": {
            "id": "E27",
            "title": "American Asian barrier under Heston",
            "cross_validate": {"internal": ["american_pathdep_pde", "american_pathdep_mc", "american_pathdep_fft"]},
            "market_assertions": {
                "requires": ["discount_curve", "black_vol_surface", "model_parameters", "spot"]
            },
        },
    }
    fake_results = iter(
        [
            {
                "task_id": "T105",
                "success": True,
                "attempts": 1,
                "elapsed_seconds": 10.0,
                "cross_validation": {"status": "passed"},
                "task_run_latest_path": t105_latest,
                "token_usage_summary": {"total_tokens": 100, "call_count": 2},
            },
            {
                "task_id": "E27",
                "success": False,
                "attempts": 1,
                "elapsed_seconds": 3.0,
                "cross_validation": {"status": "insufficient_results"},
                "blocker_details": {"blocker_report": {"blockers": [{"category": "unsupported_composite"}]}},
                "task_run_latest_path": e27_latest,
                "token_usage_summary": {"total_tokens": 10, "call_count": 1},
            },
        ]
    )

    monkeypatch.setattr(module, "load_binding_first_exotic_proof_manifest", lambda: manifest)
    monkeypatch.setattr(module, "_load_selected_tasks", lambda task_ids: {task_id: tasks[task_id] for task_id in task_ids})
    monkeypatch.setattr(
        module,
        "grade_binding_first_exotic_proof_preflight",
        lambda *args, **kwargs: {
            "task_contract_alignment": SimpleNamespace(passed=True, details=()),
            "market_capability_alignment": SimpleNamespace(passed=True, details=()),
            "comparison_target_inventory": SimpleNamespace(passed=True, details=()),
            "comparison_target_separation": SimpleNamespace(passed=True, details=()),
        },
    )
    monkeypatch.setattr(module, "run_task", lambda *args, **kwargs: next(fake_results))

    output_file = tmp_path / "proof_results.json"
    report_json = tmp_path / "proof_report.json"
    report_md = tmp_path / "proof_report.md"

    exit_code = module.run_binding_first_exotic_proof(
        cohort="event_control_schedule",
        task_ids=[],
        model="test-model",
        validation="standard",
        fresh_build=True,
        preflight_only=False,
        output_path=output_file,
        report_json_path=report_json,
        report_md_path=report_md,
    )

    assert exit_code == 0
    report = json.loads(report_json.read_text())
    assert report["proof_summary"]["totals"]["passed_gate"] == 2
    assert report["proof_summary"]["by_task"]["T105"]["binding_ids"] == [
        "trellis.models.fx_vanilla.price_quanto_option_analytical_from_market_state"
    ]
    assert report["proof_summary"]["by_task"]["E27"]["outcome_class"] == "honest_block"
    assert "Binding-First Exotic Proof Cohort" in report_md.read_text()


def test_run_binding_first_exotic_proof_returns_nonzero_when_live_gate_fails(tmp_path, monkeypatch):
    module = _load_module()

    manifest = {
        "T105": {
            "cohort": "event_control_schedule",
            "outcome_class": "proved",
            "required_mock_capabilities": ["discount_curve", "forward_curve", "fx_rates", "spot"],
            "comparison_targets": ["quanto_bs", "mc_quanto"],
            "requires_binding_ids": True,
        },
    }
    tasks = {
        "T105": {
            "id": "T105",
            "title": "Quanto option: quanto-adjusted BS vs MC cross-currency",
            "cross_validate": {"internal": ["quanto_bs", "mc_quanto"]},
            "market_assertions": {
                "requires": ["discount_curve", "forward_curve", "fx_rates", "spot"]
            },
        }
    }

    monkeypatch.setattr(module, "load_binding_first_exotic_proof_manifest", lambda: manifest)
    monkeypatch.setattr(module, "_load_selected_tasks", lambda task_ids: {task_id: tasks[task_id] for task_id in task_ids})
    monkeypatch.setattr(
        module,
        "grade_binding_first_exotic_proof_preflight",
        lambda *args, **kwargs: {
            "task_contract_alignment": SimpleNamespace(passed=True, details=()),
            "market_capability_alignment": SimpleNamespace(passed=True, details=()),
            "comparison_target_inventory": SimpleNamespace(passed=True, details=()),
            "comparison_target_separation": SimpleNamespace(passed=True, details=()),
        },
    )
    monkeypatch.setattr(
        module,
        "run_task",
        lambda *args, **kwargs: {
            "task_id": "T105",
            "success": True,
            "attempts": 1,
            "elapsed_seconds": 10.0,
            "cross_validation": {"status": "passed"},
            "token_usage_summary": {"total_tokens": 100, "call_count": 2},
        },
    )

    exit_code = module.run_binding_first_exotic_proof(
        cohort="event_control_schedule",
        task_ids=[],
        model="test-model",
        validation="standard",
        fresh_build=True,
        preflight_only=False,
        output_path=tmp_path / "proof_results.json",
        report_json_path=tmp_path / "proof_report.json",
        report_md_path=tmp_path / "proof_report.md",
    )

    assert exit_code == 1


def test_run_binding_first_exotic_proof_rejects_unknown_selected_task_ids(tmp_path, monkeypatch):
    module = _load_module()

    monkeypatch.setattr(
        module,
        "load_binding_first_exotic_proof_manifest",
        lambda: {
            "T105": {
                "cohort": "event_control_schedule",
                "outcome_class": "proved",
                "required_mock_capabilities": [],
                "comparison_targets": [],
            }
        },
    )

    exit_code = module.run_binding_first_exotic_proof(
        cohort="event_control_schedule",
        task_ids=["T105", "BAD_ID"],
        model="test-model",
        validation="standard",
        fresh_build=True,
        preflight_only=True,
        output_path=tmp_path / "proof_results.json",
        report_json_path=tmp_path / "proof_report.json",
        report_md_path=tmp_path / "proof_report.md",
    )

    assert exit_code == 1


def test_run_binding_first_exotic_proof_freezes_post_build_learning_surface(tmp_path, monkeypatch):
    module = _load_module()

    manifest = {
        "T105": {
            "cohort": "event_control_schedule",
            "outcome_class": "proved",
            "required_mock_capabilities": [],
            "comparison_targets": [],
        }
    }
    tasks = {
        "T105": {
            "id": "T105",
            "title": "Quanto option: quanto-adjusted BS vs MC cross-currency",
            "cross_validate": {"internal": []},
            "market_assertions": {"requires": []},
        }
    }
    observed_env: list[tuple[str | None, str | None]] = []

    monkeypatch.setattr(module, "load_binding_first_exotic_proof_manifest", lambda: manifest)
    monkeypatch.setattr(module, "_load_selected_tasks", lambda task_ids: {task_id: tasks[task_id] for task_id in task_ids})
    monkeypatch.setattr(
        module,
        "grade_binding_first_exotic_proof_preflight",
        lambda *args, **kwargs: {
            "task_contract_alignment": SimpleNamespace(passed=True, details=()),
            "market_capability_alignment": SimpleNamespace(passed=True, details=()),
            "comparison_target_inventory": SimpleNamespace(passed=True, details=()),
            "comparison_target_separation": SimpleNamespace(passed=True, details=()),
        },
    )

    monkeypatch.setenv("TRELLIS_SKIP_POST_BUILD_REFLECTION", "0")
    monkeypatch.delenv("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION", raising=False)

    def _fake_run_task(*args, **kwargs):
        import os

        observed_env.append(
            (
                os.environ.get("TRELLIS_SKIP_POST_BUILD_REFLECTION"),
                os.environ.get("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION"),
            )
        )
        return {
            "task_id": "T105",
            "success": True,
            "attempts": 1,
            "elapsed_seconds": 1.0,
            "cross_validation": {"status": "passed"},
            "token_usage_summary": {"total_tokens": 1, "call_count": 1},
        }

    monkeypatch.setattr(module, "run_task", _fake_run_task)

    exit_code = module.run_binding_first_exotic_proof(
        cohort="event_control_schedule",
        task_ids=[],
        model="test-model",
        validation="standard",
        fresh_build=True,
        preflight_only=False,
        output_path=tmp_path / "proof_results.json",
        report_json_path=tmp_path / "proof_report.json",
        report_md_path=tmp_path / "proof_report.md",
    )

    assert exit_code == 1
    assert observed_env == [("1", "1")]
    import os
    assert os.environ.get("TRELLIS_SKIP_POST_BUILD_REFLECTION") == "0"
    assert os.environ.get("TRELLIS_SKIP_POST_BUILD_CONSOLIDATION") is None
