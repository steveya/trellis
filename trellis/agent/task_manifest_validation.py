"""Fail-closed validation for task manifests and their external references."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trellis.agent.knowledge.methods import is_known_method


LEGACY_TASKS_MANIFEST = "TASKS_PROOF_LEGACY.yaml"
LEGACY_BASELINE_MANIFEST = "TASKS_PROOF_LEGACY_BASELINE.yaml"

_COMMON_REQUIRED_FIELDS = ("id", "title", "status")
_VALID_STATUSES = frozenset({"pending", "done", "blocked"})
_VALID_LEGACY_DISPOSITIONS = frozenset(
    {
        "executable_pricing",
        "named_proof_fixture",
        "expected_honest_block",
        "research_hold",
        "proof_hold",
        "rewrite_candidate",
    }
)
_NON_PRICING_LEGACY_DISPOSITIONS = frozenset(
    {"expected_honest_block", "research_hold", "proof_hold", "rewrite_candidate"}
)
_VALID_EXTENSION_POLICIES = frozenset({"invariants_and_cross_method"})


@dataclass(frozen=True, order=True)
class TaskManifestIssue:
    """One stable, machine-readable task-manifest validation finding."""

    manifest: str
    code: str
    message: str
    task_id: str = ""
    path: str = ""

    @property
    def key(self) -> str:
        """Return a stable identity suitable for checked debt baselines."""
        return ":".join(
            (
                self.manifest,
                self.task_id or "<manifest>",
                self.code,
                self.path or "<root>",
            )
        )


@dataclass(frozen=True)
class TaskManifestValidationReport:
    """Strict failures plus visible, checked legacy contract debt."""

    blocking_issues: tuple[TaskManifestIssue, ...]
    legacy_issues: tuple[TaskManifestIssue, ...]
    legacy_task_fingerprint: str = ""

    @property
    def issue_count(self) -> int:
        return len(self.blocking_issues) + len(self.legacy_issues)


class TaskManifestValidationError(ValueError):
    """Raised when task selection would cross an invalid manifest boundary."""

    def __init__(self, report: TaskManifestValidationReport):
        self.report = report
        preview = "\n".join(
            f"- {issue.key}: {issue.message}" for issue in report.blocking_issues[:12]
        )
        remaining = len(report.blocking_issues) - 12
        suffix = f"\n- ... and {remaining} more" if remaining > 0 else ""
        super().__init__(
            f"task-manifest validation failed with {len(report.blocking_issues)} "
            f"blocking issue(s):\n{preview}{suffix}"
        )


def legacy_issue_digest(issues: Sequence[TaskManifestIssue]) -> str:
    """Digest the exact stable identity set for known legacy contract debt."""
    payload = "\n".join(sorted(issue.key for issue in issues)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def legacy_task_fingerprint(tasks: Sequence[Mapping[str, Any]]) -> str:
    """Digest normalized legacy task content, including authored semantics."""
    payload = yaml.safe_dump(
        [dict(task) for task in tasks],
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_task_manifests(
    *,
    root: Path,
    manifest_names: Sequence[str],
    legacy_baseline_name: str = LEGACY_BASELINE_MANIFEST,
) -> TaskManifestValidationReport:
    """Audit selected corpora without hiding known legacy incompleteness."""
    blocking: list[TaskManifestIssue] = []
    legacy: list[TaskManifestIssue] = []
    located_ids: dict[str, list[tuple[str, str]]] = defaultdict(list)
    scenario_ids = _load_registry_ids(root / "MARKET_SCENARIOS.yaml", "scenarios")
    binding_ids = _load_registry_ids(root / "FINANCEPY_BINDINGS.yaml", "bindings")
    legacy_tasks: list[Mapping[str, Any]] = []
    resolved_root = root.resolve()

    for manifest_name in manifest_names:
        manifest_path = (root / manifest_name).resolve()
        try:
            manifest_path.relative_to(resolved_root)
        except ValueError:
            blocking.append(
                _issue(
                    manifest_name,
                    "manifest.outside_root",
                    "task manifest must resolve inside the repository root",
                )
            )
            continue
        tasks, structural_issues = _read_tasks(manifest_path, manifest_name)
        blocking.extend(structural_issues)
        for index, task in tasks:
            path = f"tasks[{index}]"
            task_id = _text(task.get("id"))
            if task_id:
                located_ids[task_id].append((manifest_name, path))
            blocking.extend(_validate_common(manifest_name, task, path))
            blocking.extend(
                _validate_references(
                    manifest_name,
                    task,
                    path,
                    scenario_ids=scenario_ids,
                    binding_ids=binding_ids,
                )
            )
            if manifest_name == "TASKS_BENCHMARK_FINANCEPY.yaml":
                blocking.extend(_validate_financepy_task(manifest_name, task, path))
            elif manifest_name == "TASKS_EXTENSION.yaml":
                blocking.extend(_validate_extension_task(manifest_name, task, path))
            elif manifest_name == "TASKS_FPML_CONFORMANCE.yaml":
                blocking.extend(
                    _validate_fpml_task(manifest_name, task, path, root=root)
                )
            elif manifest_name == "TASKS_NEGATIVE.yaml":
                blocking.extend(_validate_negative_task(manifest_name, task, path))
            elif manifest_name == "FRAMEWORK_TASKS.yaml":
                blocking.extend(_validate_framework_task(manifest_name, task, path))
            elif manifest_name == LEGACY_TASKS_MANIFEST:
                legacy_tasks.append(task)
                legacy.extend(_validate_legacy_task(manifest_name, task, path))

    for task_id, locations in located_ids.items():
        if len(locations) < 2:
            continue
        location_text = ", ".join(f"{manifest}:{path}" for manifest, path in locations)
        for manifest_name, path in locations:
            blocking.append(
                _issue(
                    manifest_name,
                    "task.duplicate_id",
                    f"task id {task_id!r} is duplicated at {location_text}",
                    task_id=task_id,
                    path=f"{path}.id",
                )
            )

    if LEGACY_TASKS_MANIFEST in manifest_names:
        fingerprint = legacy_task_fingerprint(legacy_tasks)
        baseline_path = (root / legacy_baseline_name).resolve()
        try:
            baseline_path.relative_to(resolved_root)
        except ValueError:
            blocking.append(
                _issue(
                    legacy_baseline_name,
                    "legacy.baseline_outside_root",
                    "legacy baseline must resolve inside the repository root",
                )
            )
        else:
            blocking.extend(
                _validate_legacy_baseline(
                    baseline_path,
                    legacy,
                    task_fingerprint=fingerprint,
                    baseline_name=legacy_baseline_name,
                )
            )
    else:
        fingerprint = ""

    return TaskManifestValidationReport(
        blocking_issues=tuple(sorted(set(blocking))),
        legacy_issues=tuple(sorted(set(legacy))),
        legacy_task_fingerprint=fingerprint,
    )


def assert_valid_task_manifests(
    *,
    root: Path,
    manifest_names: Sequence[str],
    legacy_baseline_name: str = LEGACY_BASELINE_MANIFEST,
) -> TaskManifestValidationReport:
    """Raise before task loading when selected manifest contracts are invalid."""
    report = audit_task_manifests(
        root=root,
        manifest_names=manifest_names,
        legacy_baseline_name=legacy_baseline_name,
    )
    if report.blocking_issues:
        raise TaskManifestValidationError(report)
    return report


def assert_executable_task_selection(
    tasks: Sequence[Mapping[str, Any]],
) -> None:
    """Reject incomplete or explicitly non-priceable legacy rows before execution."""
    issues: list[TaskManifestIssue] = []
    for index, task in enumerate(tasks):
        if _text(task.get("task_definition_manifest")) != LEGACY_TASKS_MANIFEST:
            continue
        path = f"selected_tasks[{index}]"
        task_issues = _validate_common(LEGACY_TASKS_MANIFEST, task, path)
        task_issues.extend(_validate_legacy_task(LEGACY_TASKS_MANIFEST, task, path))
        disposition = _text(task.get("task_disposition"))
        if disposition in _NON_PRICING_LEGACY_DISPOSITIONS:
            task_issues.append(
                _issue(
                    LEGACY_TASKS_MANIFEST,
                    "legacy.non_executable_disposition",
                    f"legacy task disposition {disposition!r} is not executable pricing",
                    task_id=_text(task.get("id")),
                    path=f"{path}.task_disposition",
                )
            )
        issues.extend(task_issues)
    if issues:
        report = TaskManifestValidationReport(
            blocking_issues=tuple(sorted(set(issues))),
            legacy_issues=(),
        )
        raise TaskManifestValidationError(report)


def _read_tasks(
    path: Path,
    manifest_name: str,
) -> tuple[list[tuple[int, Mapping[str, Any]]], list[TaskManifestIssue]]:
    if not path.exists():
        return [], [
            _issue(manifest_name, "manifest.missing", f"manifest does not exist: {path}")
        ]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [], [
            _issue(manifest_name, "manifest.invalid_yaml", f"cannot parse YAML: {exc}")
        ]

    if isinstance(raw, list) and manifest_name == "FRAMEWORK_TASKS.yaml":
        task_rows = raw
    elif isinstance(raw, Mapping):
        version = raw.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            return [], [
                _issue(
                    manifest_name,
                    "manifest.invalid_version",
                    "versioned task manifests require a positive integer version",
                    path="version",
                )
            ]
        task_rows = raw.get("tasks")
        if not isinstance(task_rows, list):
            return [], [
                _issue(
                    manifest_name,
                    "manifest.invalid_tasks",
                    "top-level 'tasks' must be a list",
                    path="tasks",
                )
            ]
    else:
        return [], [
            _issue(
                manifest_name,
                "manifest.invalid_top_level",
                "manifest must be a task list or a mapping containing a task list",
            )
        ]

    tasks: list[tuple[int, Mapping[str, Any]]] = []
    issues: list[TaskManifestIssue] = []
    for index, row in enumerate(task_rows):
        if not isinstance(row, Mapping):
            issues.append(
                _issue(
                    manifest_name,
                    "task.invalid_row",
                    "task row must be a mapping",
                    path=f"tasks[{index}]",
                )
            )
            continue
        tasks.append((index, row))
    return tasks, issues


def _validate_common(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    issues: list[TaskManifestIssue] = []
    task_id = _text(task.get("id"))
    for field in _COMMON_REQUIRED_FIELDS:
        if not _text(task.get(field)):
            issues.append(
                _issue(
                    manifest_name,
                    "task.missing_field",
                    f"required task field {field!r} is missing",
                    task_id=task_id,
                    path=f"{path}.{field}",
                )
            )
    status = _text(task.get("status"))
    if status and status not in _VALID_STATUSES:
        issues.append(
            _issue(
                manifest_name,
                "task.invalid_status",
                f"unsupported task status {status!r}",
                task_id=task_id,
                path=f"{path}.status",
            )
        )
    return issues


def _validate_references(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
    *,
    scenario_ids: frozenset[str],
    binding_ids: frozenset[str],
) -> list[TaskManifestIssue]:
    issues: list[TaskManifestIssue] = []
    task_id = _text(task.get("id"))
    scenario_id = _text(task.get("market_scenario_id"))
    if scenario_id and scenario_id not in scenario_ids:
        issues.append(
            _issue(
                manifest_name,
                "reference.unknown_market_scenario",
                f"unknown market scenario {scenario_id!r}",
                task_id=task_id,
                path=f"{path}.market_scenario_id",
            )
        )
    binding_id = _text(task.get("financepy_binding_id"))
    if binding_id and binding_id not in binding_ids:
        issues.append(
            _issue(
                manifest_name,
                "reference.unknown_financepy_binding",
                f"unknown FinancePy binding {binding_id!r}",
                task_id=task_id,
                path=f"{path}.financepy_binding_id",
            )
        )
    return issues


def _validate_financepy_task(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    issues = _require_text_fields(
        manifest_name,
        task,
        path,
        ("description", "instrument_type", "market_scenario_id", "financepy_binding_id"),
        code="financepy.missing_field",
    )
    task_id = _text(task.get("id"))
    if not _valid_construct(task.get("construct"), allow_multiple=False):
        issues.append(
            _issue(
                manifest_name,
                "financepy.invalid_construct",
                "construct must name one supported pricing method",
                task_id=task_id,
                path=f"{path}.construct",
            )
        )
    contract = task.get("benchmark_contract")
    if not _authored_economic_contract(contract):
        issues.append(
            _issue(
                manifest_name,
                "financepy.missing_economic_contract",
                "benchmark_contract must name its product and contain authored economics",
                task_id=task_id,
                path=f"{path}.benchmark_contract",
            )
        )
    if not _has_percentage_tolerance(task.get("cross_validate")):
        issues.append(
            _issue(
                manifest_name,
                "financepy.missing_tolerance",
                "cross_validate.tolerance_pct must be an authored finite non-negative number",
                task_id=task_id,
                path=f"{path}.cross_validate.tolerance_pct",
            )
        )
    return issues


def _validate_extension_task(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    issues = _require_text_fields(
        manifest_name,
        task,
        path,
        ("description", "instrument_type", "market_scenario_id"),
        code="extension.missing_field",
    )
    task_id = _text(task.get("id"))
    construct = task.get("construct")
    if not _valid_construct(construct, allow_multiple=True):
        issues.append(
            _issue(
                manifest_name,
                "extension.invalid_construct",
                "construct must name one or more supported pricing methods",
                task_id=task_id,
                path=f"{path}.construct",
            )
        )
    contract = task.get("extension_contract")
    if not _authored_economic_contract(contract):
        issues.append(
            _issue(
                manifest_name,
                "extension.missing_economic_contract",
                "extension_contract must name its product and contain authored economics",
                task_id=task_id,
                path=f"{path}.extension_contract",
            )
        )
    validation_policy = _text(task.get("validation_policy"))
    if not validation_policy:
        issues.append(
            _issue(
                manifest_name,
                "extension.missing_validation_policy",
                "validation_policy must state the acceptance strategy",
                task_id=task_id,
                path=f"{path}.validation_policy",
            )
        )
    elif validation_policy not in _VALID_EXTENSION_POLICIES:
        issues.append(
            _issue(
                manifest_name,
                "extension.invalid_validation_policy",
                f"unsupported validation policy {validation_policy!r}",
                task_id=task_id,
                path=f"{path}.validation_policy",
            )
        )
    return issues


def _validate_negative_task(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    issues = _require_text_fields(
        manifest_name,
        task,
        path,
        ("description", "market_scenario_id", "expected_outcome"),
        code="negative.missing_field",
    )
    task_id = _text(task.get("id"))
    outcome = _text(task.get("expected_outcome"))
    contract = task.get("clarification_contract")
    if outcome not in {"clarification_requested", "honest_block"}:
        issues.append(
            _issue(
                manifest_name,
                "negative.invalid_outcome",
                "expected_outcome must be clarification_requested or honest_block",
                task_id=task_id,
                path=f"{path}.expected_outcome",
            )
        )
    if outcome == "clarification_requested":
        missing_fields = contract.get("missing_fields") if isinstance(contract, Mapping) else None
        if not _nonempty_string_sequence(missing_fields):
            issues.append(
                _issue(
                    manifest_name,
                    "negative.missing_fields",
                    "clarification requests must enumerate missing_fields",
                    task_id=task_id,
                    path=f"{path}.clarification_contract.missing_fields",
                )
            )
    elif outcome == "honest_block":
        reason = contract.get("unsupported_reason") if isinstance(contract, Mapping) else None
        blockers = contract.get("expected_blockers") if isinstance(contract, Mapping) else None
        if not _text(reason) or not _nonempty_string_sequence(blockers):
            issues.append(
                _issue(
                    manifest_name,
                    "negative.missing_block_contract",
                    "honest blocks require unsupported_reason and expected_blockers",
                    task_id=task_id,
                    path=f"{path}.clarification_contract",
                )
            )
    return issues


def _validate_fpml_task(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
    *,
    root: Path,
) -> list[TaskManifestIssue]:
    issues = _require_text_fields(
        manifest_name,
        task,
        path,
        ("task_kind", "expected_outcome", "market_scenario_id"),
        code="fpml.missing_field",
    )
    task_id = _text(task.get("id"))
    if _text(task.get("task_kind")) != "fpml_conformance":
        issues.append(
            _issue(
                manifest_name,
                "fpml.invalid_task_kind",
                "task_kind must be exactly 'fpml_conformance'",
                task_id=task_id,
                path=f"{path}.task_kind",
            )
        )
    fpml = task.get("fpml")
    required_source_fields = ("fixture", "source_view", "source_version")
    if not isinstance(fpml, Mapping) or any(not _text(fpml.get(field)) for field in required_source_fields):
        issues.append(
            _issue(
                manifest_name,
                "fpml.missing_source_contract",
                "fpml must declare fixture, source_view, and source_version",
                task_id=task_id,
                path=f"{path}.fpml",
            )
        )
    else:
        fixture = _text(fpml.get("fixture"))
        fixture_path = (root / fixture).resolve()
        try:
            fixture_path.relative_to(root.resolve())
        except ValueError:
            fixture_path = Path()
        if not fixture_path.is_file():
            issues.append(
                _issue(
                    manifest_name,
                    "fpml.missing_fixture",
                    f"FpML fixture does not exist inside the repository: {fixture!r}",
                    task_id=task_id,
                    path=f"{path}.fpml.fixture",
                )
            )
    outcome = _text(task.get("expected_outcome"))
    if outcome == "pricing_success":
        native_contract = task.get("native_contract")
        if not _authored_contract(native_contract, discriminator="kind"):
            issues.append(
                _issue(
                    manifest_name,
                    "fpml.missing_native_contract",
                    "pricing_success requires a native_contract",
                    task_id=task_id,
                    path=f"{path}.native_contract",
                )
            )
        tolerance = task.get("tolerance")
        if not _has_absolute_relative_tolerance(tolerance):
            issues.append(
                _issue(
                    manifest_name,
                    "fpml.missing_tolerance",
                    "pricing_success requires absolute and relative tolerances",
                    task_id=task_id,
                    path=f"{path}.tolerance",
                )
            )
    elif outcome == "honest_block":
        if not _nonempty_string_sequence(task.get("expected_blocker_ids")):
            issues.append(
                _issue(
                    manifest_name,
                    "fpml.missing_blockers",
                    "honest_block requires expected_blocker_ids",
                    task_id=task_id,
                    path=f"{path}.expected_blocker_ids",
                )
            )
    else:
        issues.append(
            _issue(
                manifest_name,
                "fpml.invalid_outcome",
                "expected_outcome must be pricing_success or honest_block",
                task_id=task_id,
                path=f"{path}.expected_outcome",
            )
        )
    return issues


def _validate_framework_task(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    issues = _require_text_fields(
        manifest_name,
        task,
        path,
        ("construct",),
        code="framework.missing_field",
    )
    task_id = _text(task.get("id"))
    trigger_after = task.get("trigger_after")
    if (
        not _text(task.get("new_component"))
        and not _text(trigger_after)
        and not _nonempty_string_sequence(trigger_after)
    ):
        issues.append(
            _issue(
                manifest_name,
                "framework.missing_delivery_contract",
                "framework tasks require new_component or trigger_after",
                task_id=task_id,
                path=path,
            )
        )
    return issues


def _validate_legacy_task(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    issues: list[TaskManifestIssue] = []
    task_id = _text(task.get("id"))
    disposition = _text(task.get("task_disposition"))
    if disposition not in _VALID_LEGACY_DISPOSITIONS:
        issues.append(
            _issue(
                manifest_name,
                "legacy.missing_disposition",
                "declare executable, named-fixture, honest-block, or hold disposition",
                task_id=task_id,
                path=f"{path}.task_disposition",
            )
        )
        disposition = "executable_pricing"

    if disposition in _NON_PRICING_LEGACY_DISPOSITIONS:
        if not _text(task.get("disposition_reason")):
            issues.append(
                _issue(
                    manifest_name,
                    "legacy.missing_disposition_reason",
                    "non-pricing dispositions require disposition_reason",
                    task_id=task_id,
                    path=f"{path}.disposition_reason",
                )
            )
        return issues

    if not _text(task.get("description")):
        issues.append(
            _issue(
                manifest_name,
                "legacy.missing_description",
                "executable tasks require an authored pricing request",
                task_id=task_id,
                path=f"{path}.description",
            )
        )
    if disposition == "named_proof_fixture":
        has_economics = bool(_text(task.get("proof_fixture_id")))
    else:
        has_economics = any(
            _authored_economic_contract(task.get(field))
            for field in ("benchmark_contract", "extension_contract")
        )
    if not has_economics:
        issues.append(
            _issue(
                manifest_name,
                "legacy.missing_economic_contract",
                "executable tasks require authored economics or a named proof fixture",
                task_id=task_id,
                path=path,
            )
        )
    if not any(
        (
            _text(task.get("market_scenario_id")),
            _nonempty_mapping(task.get("market")),
            _nonempty_mapping(task.get("comparison_regime")),
        )
    ):
        issues.append(
            _issue(
                manifest_name,
                "legacy.missing_market_contract",
                "executable tasks require an authored market contract",
                task_id=task_id,
                path=path,
            )
        )
    if not _has_acceptance_contract(task):
        issues.append(
            _issue(
                manifest_name,
                "legacy.missing_acceptance_contract",
                "executable tasks require an authored tolerance or validation policy",
                task_id=task_id,
                path=path,
            )
        )
    return issues


def _validate_legacy_baseline(
    path: Path,
    legacy_issues: Sequence[TaskManifestIssue],
    *,
    task_fingerprint: str,
    baseline_name: str,
) -> list[TaskManifestIssue]:
    if not path.exists():
        return [
            _issue(
                baseline_name,
                "legacy.baseline_missing",
                f"checked legacy-debt baseline does not exist: {path}",
            )
        ]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [
            _issue(
                baseline_name,
                "legacy.baseline_invalid",
                f"cannot parse legacy baseline: {exc}",
            )
        ]
    if not isinstance(raw, Mapping):
        return [
            _issue(
                baseline_name,
                "legacy.baseline_invalid",
                "legacy baseline must be a mapping",
            )
        ]
    if (
        raw.get("version") != 1
        or _text(raw.get("manifest")) != LEGACY_TASKS_MANIFEST
        or _text(raw.get("policy")) != "exact_issue_identity_and_task_content"
    ):
        return [
            _issue(
                baseline_name,
                "legacy.baseline_invalid",
                "legacy baseline must declare version 1, TASKS_PROOF_LEGACY.yaml, "
                "and exact_issue_identity_and_task_content policy",
            )
        ]
    expected_count = raw.get("issue_count")
    expected_digest = _text(raw.get("issue_digest"))
    expected_fingerprint = _text(raw.get("task_fingerprint"))
    actual_digest = legacy_issue_digest(legacy_issues)
    if (
        expected_count != len(legacy_issues)
        or expected_digest != actual_digest
        or expected_fingerprint != task_fingerprint
    ):
        return [
            _issue(
                baseline_name,
                "legacy.baseline_mismatch",
                "legacy contract debt changed; repair or worsenings require an intentional baseline update "
                f"(expected count={expected_count!r}, digest={expected_digest!r}; "
                f"actual count={len(legacy_issues)}, digest={actual_digest!r}; "
                f"expected task_fingerprint={expected_fingerprint!r}, "
                f"actual task_fingerprint={task_fingerprint!r})",
            )
        ]
    return []


def _load_registry_ids(path: Path, key: str) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return frozenset()
    if not isinstance(raw, Mapping) or not isinstance(raw.get(key), Mapping):
        return frozenset()
    return frozenset(_text(item) for item in raw[key] if _text(item))


def _require_text_fields(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
    fields: Sequence[str],
    *,
    code: str,
) -> list[TaskManifestIssue]:
    task_id = _text(task.get("id"))
    return [
        _issue(
            manifest_name,
            code,
            f"required field {field!r} is missing",
            task_id=task_id,
            path=f"{path}.{field}",
        )
        for field in fields
        if not _text(task.get(field))
    ]


def _has_percentage_tolerance(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    tolerance = value.get("tolerance_pct")
    return _finite_non_negative_number(tolerance)


def _has_absolute_relative_tolerance(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        _finite_non_negative_number(value.get(field))
        for field in ("absolute", "relative")
    )


def _has_acceptance_contract(task: Mapping[str, Any]) -> bool:
    if _text(task.get("validation_policy")):
        return True
    cross_validate = task.get("cross_validate")
    if _has_percentage_tolerance(cross_validate):
        return True
    if isinstance(cross_validate, Mapping):
        tolerance = cross_validate.get("tolerance")
        if _has_absolute_relative_tolerance(tolerance):
            return True
    return False


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _authored_economic_contract(value: Any) -> bool:
    return _authored_contract(value, discriminator="product")


def _authored_contract(value: Any, *, discriminator: str) -> bool:
    if not isinstance(value, Mapping) or not _text(value.get(discriminator)):
        return False
    if _contains_non_finite_number(value):
        return False
    return any(
        key != discriminator and _meaningful_value(item)
        for key, item in value.items()
    )


def _contains_non_finite_number(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_non_finite_number(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_non_finite_number(item) for item in value)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    )


def _meaningful_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_meaningful_value(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_meaningful_value(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return value is not None


def _nonempty_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value)


def _nonempty_string_sequence(value: Any) -> bool:
    return _nonempty_sequence(value) and all(_text(item) for item in value)


def _valid_construct(value: Any, *, allow_multiple: bool) -> bool:
    if isinstance(value, str):
        methods = (value.strip(),)
    elif allow_multiple and _nonempty_string_sequence(value):
        methods = tuple(item.strip() for item in value)
    else:
        return False
    return bool(methods) and all(is_known_method(method) for method in methods)


def _finite_non_negative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _issue(
    manifest: str,
    code: str,
    message: str,
    *,
    task_id: str = "",
    path: str = "",
) -> TaskManifestIssue:
    return TaskManifestIssue(
        manifest=manifest,
        code=code,
        message=message,
        task_id=task_id,
        path=path,
    )
