from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml


def _write_yaml(root: Path, name: str, payload) -> None:
    (root / name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _codes(report) -> set[str]:
    return {issue.code for issue in report.blocking_issues}


def test_missing_and_malformed_manifests_fail_closed(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    missing = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )
    assert _codes(missing) == {"manifest.missing"}

    _write_yaml(tmp_path, "TASKS_EXTENSION.yaml", "not-a-task-manifest")
    malformed = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml",),
    )
    assert "manifest.invalid_top_level" in _codes(malformed)


def test_duplicate_task_ids_are_rejected_across_manifests(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    task = {"id": "P900", "title": "Duplicate", "status": "pending"}
    _write_yaml(tmp_path, "TASKS_EXTENSION.yaml", {"version": 1, "tasks": [task]})
    _write_yaml(tmp_path, "TASKS_NEGATIVE.yaml", {"version": 1, "tasks": [task]})

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_EXTENSION.yaml", "TASKS_NEGATIVE.yaml"),
    )

    duplicates = [
        issue for issue in report.blocking_issues if issue.code == "task.duplicate_id"
    ]
    assert len(duplicates) == 2
    assert {issue.manifest for issue in duplicates} == {
        "TASKS_EXTENSION.yaml",
        "TASKS_NEGATIVE.yaml",
    }


def test_financepy_contract_requires_references_economics_and_acceptance(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(tmp_path, "MARKET_SCENARIOS.yaml", {"version": 1, "scenarios": {}})
    _write_yaml(tmp_path, "FINANCEPY_BINDINGS.yaml", {"version": 1, "bindings": {}})
    _write_yaml(
        tmp_path,
        "TASKS_BENCHMARK_FINANCEPY.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "F900",
                    "title": "Incomplete parity task",
                    "status": "pending",
                    "description": "Price a product.",
                    "instrument_type": "european_option",
                    "construct": "analytical",
                    "market_scenario_id": "missing-market",
                    "financepy_binding_id": "missing-binding",
                    "benchmark_contract": {},
                    "cross_validate": {"external": ["financepy"]},
                }
            ],
        },
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_BENCHMARK_FINANCEPY.yaml",),
    )

    assert {
        "reference.unknown_market_scenario",
        "reference.unknown_financepy_binding",
        "financepy.missing_economic_contract",
        "financepy.missing_tolerance",
    } <= _codes(report)


def test_contract_envelopes_require_typed_meaningful_content(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {"version": 1, "scenarios": {"scenario": {"description": "test"}}},
    )
    _write_yaml(
        tmp_path,
        "FINANCEPY_BINDINGS.yaml",
        {"version": 1, "bindings": {"binding": {"description": "test"}}},
    )
    _write_yaml(
        tmp_path,
        "TASKS_BENCHMARK_FINANCEPY.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "F900",
                    "title": "Near-empty contract",
                    "status": "pending",
                    "description": "Price it.",
                    "instrument_type": "option",
                    "construct": ["analytical"],
                    "market_scenario_id": "scenario",
                    "financepy_binding_id": "binding",
                    "benchmark_contract": {
                        "product": "equity_vanilla",
                        "terms": {"strike": ""},
                    },
                    "cross_validate": {"tolerance_pct": 1.0},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "P900",
                    "title": "Bad method contract",
                    "status": "pending",
                    "description": "Price it.",
                    "instrument_type": "option",
                    "construct": ["analytical", ""],
                    "market_scenario_id": "scenario",
                    "extension_contract": {
                        "product": "equity_vanilla",
                        "terms": [""],
                    },
                    "validation_policy": "invented_policy",
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_NEGATIVE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "N900",
                    "title": "Empty clarification",
                    "status": "pending",
                    "description": "Price it.",
                    "market_scenario_id": "scenario",
                    "expected_outcome": "clarification_requested",
                    "clarification_contract": {"missing_fields": [""]},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_FPML_CONFORMANCE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "FPC900",
                    "title": "Wrong task kind",
                    "status": "pending",
                    "task_kind": "other",
                    "expected_outcome": "honest_block",
                    "market_scenario_id": "scenario",
                    "fpml": {
                        "fixture": "fixture.xml",
                        "source_view": "confirmation",
                        "source_version": "5-12",
                    },
                    "expected_blocker_ids": [""],
                }
            ],
        },
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=(
            "TASKS_BENCHMARK_FINANCEPY.yaml",
            "TASKS_EXTENSION.yaml",
            "TASKS_NEGATIVE.yaml",
            "TASKS_FPML_CONFORMANCE.yaml",
        ),
    )

    assert {
        "financepy.invalid_construct",
        "financepy.missing_economic_contract",
        "extension.invalid_construct",
        "extension.missing_economic_contract",
        "extension.invalid_validation_policy",
        "negative.missing_fields",
        "fpml.invalid_task_kind",
        "fpml.missing_blockers",
    } <= _codes(report)


def test_tolerances_accept_exact_zero_and_reject_non_finite_values(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {"version": 1, "scenarios": {"scenario": {"description": "test"}}},
    )
    _write_yaml(
        tmp_path,
        "FINANCEPY_BINDINGS.yaml",
        {"version": 1, "bindings": {"binding": {"description": "test"}}},
    )
    base = {
        "title": "Tolerance contract",
        "status": "pending",
        "description": "Price it.",
        "instrument_type": "option",
        "construct": "analytical",
        "market_scenario_id": "scenario",
        "financepy_binding_id": "binding",
        "benchmark_contract": {"product": "equity_vanilla", "spot": 100.0},
    }
    _write_yaml(
        tmp_path,
        "TASKS_BENCHMARK_FINANCEPY.yaml",
        {
            "version": 1,
            "tasks": [
                {"id": "F900", **base, "cross_validate": {"tolerance_pct": 0.0}},
                {"id": "F901", **base, "cross_validate": {"tolerance_pct": math.nan}},
                {"id": "F902", **base, "cross_validate": {"tolerance_pct": math.inf}},
                {
                    "id": "F903",
                    **base,
                    "benchmark_contract": {
                        "product": "equity_vanilla",
                        "spot": 100.0,
                        "strike": math.nan,
                    },
                    "cross_validate": {"tolerance_pct": 0.0},
                },
            ],
        },
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_BENCHMARK_FINANCEPY.yaml",),
    )

    tolerance_issues = [
        issue for issue in report.blocking_issues if issue.code == "financepy.missing_tolerance"
    ]
    assert {issue.task_id for issue in tolerance_issues} == {"F901", "F902"}
    economic_issues = [
        issue
        for issue in report.blocking_issues
        if issue.code == "financepy.missing_economic_contract"
    ]
    assert {issue.task_id for issue in economic_issues} == {"F903"}


def test_extension_negative_fpml_and_framework_contract_shapes_are_strict(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "MARKET_SCENARIOS.yaml",
        {
            "version": 1,
            "scenarios": {
                "scenario": {
                    "source": "mock",
                    "as_of": "2024-11-15",
                    "description": "Test scenario.",
                    "selected_components": {},
                    "constructor": {"kind": "test"},
                }
            },
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_EXTENSION.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "P900",
                    "title": "Incomplete extension",
                    "status": "pending",
                    "description": "Price it.",
                    "instrument_type": "option",
                    "construct": "analytical",
                    "market_scenario_id": "scenario",
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_NEGATIVE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "N900",
                    "title": "Bad negative row",
                    "status": "pending",
                    "description": "Price an option.",
                    "market_scenario_id": "scenario",
                    "expected_outcome": "clarification_requested",
                    "clarification_contract": {},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "TASKS_FPML_CONFORMANCE.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "FPC900",
                    "title": "Bad FpML row",
                    "status": "pending",
                    "task_kind": "fpml_conformance",
                    "market_scenario_id": "scenario",
                    "expected_outcome": "pricing_success",
                    "fpml": {},
                }
            ],
        },
    )
    _write_yaml(
        tmp_path,
        "FRAMEWORK_TASKS.yaml",
        [
            {
                "id": "E900",
                "title": "Bad framework row",
                "status": "pending",
                "construct": "framework",
                "trigger_after": [""],
            }
        ],
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=(
            "TASKS_EXTENSION.yaml",
            "TASKS_NEGATIVE.yaml",
            "TASKS_FPML_CONFORMANCE.yaml",
            "FRAMEWORK_TASKS.yaml",
        ),
    )

    assert {
        "extension.missing_economic_contract",
        "extension.missing_validation_policy",
        "negative.missing_fields",
        "fpml.missing_source_contract",
        "fpml.missing_native_contract",
        "fpml.missing_tolerance",
        "framework.missing_delivery_contract",
    } <= _codes(report)


def test_legacy_debt_must_match_checked_baseline(tmp_path):
    from trellis.agent.task_manifest_validation import (
        audit_task_manifests,
        legacy_issue_digest,
        legacy_task_fingerprint,
    )

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY.yaml",
        {
            "version": 1,
            "tasks": [
                {
                    "id": "T900",
                    "title": "Title-only proof",
                    "status": "pending",
                    "construct": "analytical",
                    "new_component": "example",
                    "cross_validate": {"analytical": "example"},
                }
            ],
        },
    )

    first = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
    )
    assert "legacy.baseline_missing" in _codes(first)
    assert first.legacy_issues

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY_BASELINE.yaml",
        {
            "version": 1,
            "manifest": "TASKS_PROOF_LEGACY.yaml",
            "issue_count": len(first.legacy_issues),
            "issue_digest": legacy_issue_digest(first.legacy_issues),
            "task_fingerprint": legacy_task_fingerprint(
                yaml.safe_load(
                    (tmp_path / "TASKS_PROOF_LEGACY.yaml").read_text()
                )["tasks"]
            ),
            "policy": "exact_issue_identity_and_task_content",
        },
    )
    accepted = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
    )
    assert not accepted.blocking_issues
    assert accepted.legacy_issues

    payload = yaml.safe_load((tmp_path / "TASKS_PROOF_LEGACY.yaml").read_text())
    payload["tasks"][0]["title"] = "Semantically changed title-only proof"
    _write_yaml(tmp_path, "TASKS_PROOF_LEGACY.yaml", payload)
    changed = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
    )
    assert "legacy.baseline_mismatch" in _codes(changed)


def test_manifest_paths_cannot_escape_the_repository_root(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("../TASKS_EXTENSION.yaml",),
    )

    assert _codes(report) == {"manifest.outside_root"}


def test_legacy_baseline_path_cannot_escape_the_repository_root(tmp_path):
    from trellis.agent.task_manifest_validation import audit_task_manifests

    _write_yaml(
        tmp_path,
        "TASKS_PROOF_LEGACY.yaml",
        {"version": 1, "tasks": []},
    )

    report = audit_task_manifests(
        root=tmp_path,
        manifest_names=("TASKS_PROOF_LEGACY.yaml",),
        legacy_baseline_name="../TASKS_PROOF_LEGACY_BASELINE.yaml",
    )

    assert _codes(report) == {"legacy.baseline_outside_root"}


def test_incomplete_selected_legacy_task_stops_before_market_or_build(monkeypatch, tmp_path):
    from scripts import run_tasks
    from trellis.agent.task_manifest_validation import TaskManifestValidationError

    task = {
        "id": "T900",
        "title": "Sparse legacy request",
        "status": "pending",
        "task_definition_manifest": "TASKS_PROOF_LEGACY.yaml",
    }
    calls: list[str] = []

    def unexpected_market():
        calls.append("market")
        return object()

    def unexpected_build(*args, **kwargs):
        calls.append("build")
        return {}

    monkeypatch.setattr(run_tasks, "build_market_state", unexpected_market)
    monkeypatch.setattr(run_tasks, "run_task", unexpected_build)

    with pytest.raises(TaskManifestValidationError, match="legacy.missing_disposition"):
        run_tasks.run_block([task], str(tmp_path / "results.json"))

    assert calls == []


def test_assert_valid_task_manifests_raises_with_actionable_issue_keys(tmp_path):
    from trellis.agent.task_manifest_validation import (
        TaskManifestValidationError,
        assert_valid_task_manifests,
    )

    with pytest.raises(TaskManifestValidationError, match="manifest.missing") as exc:
        assert_valid_task_manifests(
            root=tmp_path,
            manifest_names=("TASKS_NEGATIVE.yaml",),
        )

    assert exc.value.report.blocking_issues


def test_pricing_loader_runs_manifest_preflight_before_loading(monkeypatch, tmp_path):
    from trellis.agent import task_manifests

    called = []

    def fail_preflight(**kwargs):
        called.append(kwargs)
        raise RuntimeError("preflight stopped loading")

    monkeypatch.setattr(task_manifests, "assert_valid_task_manifests", fail_preflight)

    with pytest.raises(RuntimeError, match="preflight stopped loading"):
        task_manifests.load_pricing_tasks(root=tmp_path)

    assert called == [{"root": tmp_path, "manifest_names": task_manifests.PRICING_TASK_CORPORA}]


def test_active_lookup_preflights_pricing_and_negative_corpora_together(monkeypatch, tmp_path):
    from trellis.agent import task_manifests

    called = []

    def record_preflight(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(task_manifests, "assert_valid_task_manifests", record_preflight)
    monkeypatch.setattr(task_manifests, "load_task_manifest", lambda *args, **kwargs: [])

    assert task_manifests.load_active_task_lookup(root=tmp_path) == {}
    assert called == [
        {
            "root": tmp_path,
            "manifest_names": (
                *task_manifests.PRICING_TASK_CORPORA,
                task_manifests.NEGATIVE_TASKS_MANIFEST,
            ),
        }
    ]


def test_financepy_benchmark_loader_uses_fail_closed_preflight(tmp_path):
    from trellis.agent.financepy_benchmark import load_financepy_benchmark_tasks
    from trellis.agent.task_manifest_validation import TaskManifestValidationError

    with pytest.raises(TaskManifestValidationError, match="manifest.missing"):
        load_financepy_benchmark_tasks(root=tmp_path)


def test_checked_in_task_manifests_pass_fail_closed_gate():
    from trellis.agent.task_manifest_validation import assert_valid_task_manifests
    from trellis.agent.task_manifests import ALL_TASK_CORPORA, ROOT

    report = assert_valid_task_manifests(root=ROOT, manifest_names=ALL_TASK_CORPORA)

    assert not report.blocking_issues
    assert report.legacy_issues
