"""Fail-closed validation for task manifests and their external references."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
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
    """Admit validated pricing/block rows and reject incomplete legacy selections."""
    issues: list[TaskManifestIssue] = []
    for index, task in enumerate(tasks):
        if _text(task.get("task_definition_manifest")) != LEGACY_TASKS_MANIFEST:
            continue
        path = f"selected_tasks[{index}]"
        task_issues = _validate_common(LEGACY_TASKS_MANIFEST, task, path)
        task_issues.extend(_validate_legacy_task(LEGACY_TASKS_MANIFEST, task, path))
        disposition = _text(task.get("task_disposition"))
        if (
            disposition in _NON_PRICING_LEGACY_DISPOSITIONS
            and disposition != "expected_honest_block"
        ):
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
    if task_id == "P003":
        issues.extend(
            _validate_p003_fx_barrier_contract(
                manifest_name,
                contract if isinstance(contract, Mapping) else {},
                path,
            )
        )
    if task_id == "P004":
        issues.extend(
            _validate_p004_callable_collar_contract(
                manifest_name,
                task,
                contract if isinstance(contract, Mapping) else {},
                path,
            )
        )
    return issues


def _validate_p003_fx_barrier_contract(
    manifest_name: str,
    contract: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    task_id = "P003"
    contract_path = f"{path}.extension_contract"
    issues: list[TaskManifestIssue] = []
    required_fields = (
        "monitoring",
        "observations_per_year",
        "rebate",
        "n_paths",
        "n_steps",
        "seed",
    )
    for field in required_fields:
        if not _meaningful_value(contract.get(field)):
            issues.append(
                _issue(
                    manifest_name,
                    "extension.fx_barrier_missing_field",
                    f"P003 FX barrier requires authored {field}",
                    task_id=task_id,
                    path=f"{contract_path}.{field}",
                )
            )

    if _text(contract.get("product")).lower() != "fx_barrier_option":
        issues.append(
            _issue(
                manifest_name,
                "extension.fx_barrier_invalid_contract",
                "P003 must remain an FX barrier option contract",
                task_id=task_id,
                path=f"{contract_path}.product",
            )
        )
    if _text(contract.get("monitoring")).lower() != "discrete":
        issues.append(
            _issue(
                manifest_name,
                "extension.fx_barrier_invalid_monitoring",
                "P003 must declare discrete monitoring",
                task_id=task_id,
                path=f"{contract_path}.monitoring",
            )
        )

    expected_integer_controls = {
        "observations_per_year": 252,
        "n_paths": 120_000,
        "n_steps": 252,
        "seed": 42,
    }
    invalid_integer_control = any(
        not isinstance(contract.get(field), int)
        or isinstance(contract.get(field), bool)
        or contract.get(field) != expected
        for field, expected in expected_integer_controls.items()
    )
    rebate = contract.get("rebate")
    invalid_rebate = not _finite_non_negative_number(rebate) or float(rebate) != 0.0
    if invalid_integer_control or invalid_rebate:
        issues.append(
            _issue(
                manifest_name,
                "extension.fx_barrier_invalid_controls",
                "P003 requires zero rebate, 252 observations/steps, 120000 paths, and seed 42",
                task_id=task_id,
                path=contract_path,
            )
        )
    return issues


def _validate_p004_callable_collar_contract(
    manifest_name: str,
    task: Mapping[str, Any],
    contract: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    task_id = _text(task.get("id"))
    contract_path = f"{path}.extension_contract"
    issues: list[TaskManifestIssue] = []
    required_fields = (
        "floor_strike",
        "cap_strike",
        "start_date",
        "maturity_date",
        "payment_frequency",
        "day_count",
        "rate_index",
        "calendar_name",
        "business_day_adjustment",
        "collar_direction",
        "controller_side",
        "call_action",
        "current_period_treatment",
        "call_settlement",
        "callable_dates",
        "accrual_dates",
        "fixing_dates",
        "payment_dates",
        "notional",
    )
    for field in required_fields:
        if not _meaningful_value(contract.get(field)):
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_missing_field",
                    f"callable collar requires authored {field}",
                    task_id=task_id,
                    path=f"{contract_path}.{field}",
                )
            )

    accrual_dates = _iso_date_sequence(contract.get("accrual_dates"))
    fixing_dates = _iso_date_sequence(contract.get("fixing_dates"))
    payment_dates = _iso_date_sequence(contract.get("payment_dates"))
    callable_dates = _iso_date_sequence(contract.get("callable_dates"))
    for field, dates in (
        ("accrual_dates", accrual_dates),
        ("fixing_dates", fixing_dates),
        ("payment_dates", payment_dates),
        ("callable_dates", callable_dates),
    ):
        if dates is None:
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    f"{field} must be a non-empty ISO-date sequence",
                    task_id=task_id,
                    path=f"{contract_path}.{field}",
                )
            )
    if accrual_dates is not None:
        if len(accrual_dates) < 2 or any(
            right <= left for left, right in zip(accrual_dates, accrual_dates[1:])
        ):
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    "accrual_dates must be strictly increasing boundaries",
                    task_id=task_id,
                    path=f"{contract_path}.accrual_dates",
                )
            )
        expected_periods = len(accrual_dates) - 1
        if fixing_dates is not None and len(fixing_dates) != expected_periods:
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    "fixing_dates must contain one date per accrual period",
                    task_id=task_id,
                    path=f"{contract_path}.fixing_dates",
                )
            )
        if payment_dates is not None and len(payment_dates) != expected_periods:
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    "payment_dates must contain one date per accrual period",
                    task_id=task_id,
                    path=f"{contract_path}.payment_dates",
                )
            )
        if fixing_dates is not None and len(fixing_dates) == expected_periods:
            if any(
                right <= left for left, right in zip(fixing_dates, fixing_dates[1:])
            ) or any(
                fixing_date > accrual_dates[index]
                for index, fixing_date in enumerate(fixing_dates)
            ):
                issues.append(
                    _issue(
                        manifest_name,
                        "extension.callable_collar_invalid_schedule",
                        "fixing dates must be strictly increasing and no later than period start",
                        task_id=task_id,
                        path=f"{contract_path}.fixing_dates",
                    )
                )
        if payment_dates is not None and len(payment_dates) == expected_periods:
            if any(
                right <= left for left, right in zip(payment_dates, payment_dates[1:])
            ) or any(
                payment_date < accrual_dates[index + 1]
                for index, payment_date in enumerate(payment_dates)
            ):
                issues.append(
                    _issue(
                        manifest_name,
                        "extension.callable_collar_invalid_schedule",
                        "payment dates must be strictly increasing and no earlier than period end",
                        task_id=task_id,
                        path=f"{contract_path}.payment_dates",
                    )
                )
        start = _iso_date(contract.get("start_date"))
        maturity = _iso_date(contract.get("maturity_date"))
        if start != accrual_dates[0] or maturity != accrual_dates[-1]:
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    "start_date and maturity_date must match the authored accrual boundaries",
                    task_id=task_id,
                    path=contract_path,
                )
            )
        if callable_dates is not None and any(
            call_date <= accrual_dates[0] or call_date >= accrual_dates[-1]
            for call_date in callable_dates
        ):
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    "callable_dates must fall strictly inside the accrual horizon",
                    task_id=task_id,
                    path=f"{contract_path}.callable_dates",
                )
            )
        if callable_dates is not None and any(
            right <= left for left, right in zip(callable_dates, callable_dates[1:])
        ):
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_schedule",
                    "callable_dates must be strictly increasing",
                    task_id=task_id,
                    path=f"{contract_path}.callable_dates",
                )
            )

    cap_strike = contract.get("cap_strike")
    floor_strike = contract.get("floor_strike")
    if (
        not _finite_non_negative_number(cap_strike)
        or not _finite_non_negative_number(floor_strike)
        or float(cap_strike) <= float(floor_strike)
        or not _finite_non_negative_number(contract.get("notional"))
        or float(contract.get("notional") or 0.0) <= 0.0
    ):
        issues.append(
            _issue(
                manifest_name,
                "extension.callable_collar_invalid_economics",
                "cap_strike and floor_strike must be finite with cap_strike above floor_strike",
                task_id=task_id,
                path=contract_path,
            )
        )

    expected_values = {
        "product": "rate_cap_floor_collar",
        "style": "callable",
        "payment_frequency": "quarterly",
        "day_count": "act/360",
        "calendar_name": "weekend_only",
        "business_day_adjustment": "following",
        "collar_direction": "pay_cap_receive_floor",
        "controller_side": "collar_payer",
        "call_action": "terminate_remaining_strip",
        "current_period_treatment": "fixed_unpaid_cashflow_survives",
        "rate_index": "usd-sofr-3m",
    }
    for field, expected in expected_values.items():
        if _text(contract.get(field)).lower() != expected:
            issues.append(
                _issue(
                    manifest_name,
                    "extension.callable_collar_invalid_semantics",
                    f"{field} must be {expected!r} for the bounded P004 contract",
                    task_id=task_id,
                    path=f"{contract_path}.{field}",
                )
            )
    if contract.get("irregular_schedule") is not True:
        issues.append(
            _issue(
                manifest_name,
                "extension.callable_collar_invalid_semantics",
                "irregular_schedule must be explicitly true",
                task_id=task_id,
                path=f"{contract_path}.irregular_schedule",
            )
        )
    settlement = contract.get("call_settlement")
    if not (
        isinstance(settlement, Mapping)
        and _text(settlement.get("type")).lower() == "cash"
        and _finite_non_negative_number(settlement.get("amount"))
        and _text(settlement.get("currency")).upper() == "USD"
        and _text(settlement.get("timing")).lower() == "exercise_date"
    ):
        issues.append(
            _issue(
                manifest_name,
                "extension.callable_collar_invalid_semantics",
                "call_settlement must author cash amount, USD currency, and exercise-date timing",
                task_id=task_id,
                path=f"{contract_path}.call_settlement",
            )
        )

    required_blockers = {
        "dynamic_composition:callable_collar_control",
        "dynamic_composition:callable_collar_continuation",
    }
    blocker_ids = set(
        str(item).strip()
        for item in (task.get("expected_blocker_ids") or ())
        if isinstance(item, str) and item.strip()
    )
    honest_block = task.get("honest_block_contract")
    missing_capabilities = set(
        str(item).strip()
        for item in (
            honest_block.get("missing_capabilities", ())
            if isinstance(honest_block, Mapping)
            else ()
        )
        if isinstance(item, str) and item.strip()
    )
    if (
        _text(task.get("expected_outcome")) != "honest_block"
        or not required_blockers.issubset(blocker_ids)
        or not isinstance(honest_block, Mapping)
        or not _text(honest_block.get("reason"))
        or not {"callable_collar_control", "callable_collar_continuation"}.issubset(
            missing_capabilities
        )
    ):
        issues.append(
            _issue(
                manifest_name,
                "extension.callable_collar_missing_honest_block",
                "callable collar must declare the control and continuation dynamic-composition blockers",
                task_id=task_id,
                path=path,
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

    if disposition == "expected_honest_block":
        issues.extend(_validate_legacy_expected_honest_block(manifest_name, task, path))
        return issues

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
    if task_id in {"T30", "T96"}:
        issues.extend(
            _validate_legacy_lookback_comparison_contract(
                manifest_name,
                task,
                path,
            )
        )
    return issues


def _validate_legacy_lookback_comparison_contract(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    """Keep the two executable legacy lookback proofs on one bounded contract."""
    contract = task.get("benchmark_contract")
    cross_validate = task.get("cross_validate")
    contract = contract if isinstance(contract, Mapping) else {}
    cross_validate = cross_validate if isinstance(cross_validate, Mapping) else {}
    targets = cross_validate.get("target_contracts")
    targets = targets if isinstance(targets, Mapping) else {}
    market = task.get("market")
    canonical_market_matches = market is None
    if isinstance(market, Mapping):
        from trellis.agent.market_scenarios import load_market_scenario_contracts

        canonical = load_market_scenario_contracts().get("equity_barrier_smile")
        if canonical is not None:
            expected_market = {
                "source": canonical.source,
                "as_of": canonical.as_of.isoformat(),
                **dict(canonical.selected_components),
                "scenario_contract": canonical.to_payload(),
                "scenario_digest": canonical.scenario_digest,
                "scenario_schema_version": canonical.schema_version,
                "scenario_constructor_kind": canonical.constructor_kind,
            }
            benchmark_inputs = canonical.financepy_inputs()
            if benchmark_inputs:
                expected_market["benchmark_inputs"] = benchmark_inputs
            canonical_market_matches = dict(market) == expected_market

    expected_contract = {
        "product": "lookback_option",
        "notional": 1.0,
        "spot": 100.0,
        "strike": 100.0,
        "option_type": "call",
        "lookback_type": "fixed_strike",
        "monitoring_style": "continuous",
        "exercise_style": "european",
        "day_count": "act/365",
        "running_extreme": 100.0,
        "expiry_years": 1.0,
        "n_paths": 80_000,
        "n_steps": 96,
        "seed": 42,
    }
    expected_targets = {
        "mc_lookback": {
            "method": "monte_carlo",
            "route_id": "monte_carlo_paths",
            "route_family": "monte_carlo",
            "validation_bundle_id": "monte_carlo:lookback_option",
            "payoff_family": "lookback_option",
            "exercise_style": "european",
            "model_family": "equity_diffusion",
            "underlying_asset_class": "equity",
            "observation_style": "path_dependent",
        },
        "conze_viswanathan_analytical": {
            "method": "analytical",
            "route_id": "analytical_black76",
            "route_family": "analytical",
            "validation_bundle_id": "analytical:lookback_option",
            "payoff_family": "lookback_option",
            "exercise_style": "european",
            "model_family": "equity_diffusion",
            "underlying_asset_class": "equity",
            "observation_style": "path_dependent",
        },
    }

    construct = task.get("construct")
    construct_methods = (
        tuple(str(item).strip() for item in construct)
        if _nonempty_string_sequence(construct)
        else ()
    )
    internal_targets = cross_validate.get("internal")
    valid = all(
        (
            _text(task.get("task_disposition")) == "executable_pricing",
            _text(task.get("instrument_type")) == "lookback_option",
            _text(task.get("market_scenario_id")) == "equity_barrier_smile",
            canonical_market_matches,
            "comparison_regime" not in task,
            not any(
                field in task
                for field in (
                    "expected_outcome",
                    "expected_blocker_ids",
                    "honest_block_contract",
                )
            ),
            _text(task.get("validation_policy")) == "invariants_and_cross_method",
            construct_methods == ("monte_carlo", "analytical"),
            set(contract) == set(expected_contract),
            all(contract.get(key) == value for key, value in expected_contract.items()),
            set(cross_validate)
            == {
                "internal",
                "reference_target",
                "relations",
                "tolerance_pct",
                "target_contracts",
                "external",
            },
            tuple(internal_targets or ())
            == ("mc_lookback", "conze_viswanathan_analytical"),
            _text(cross_validate.get("reference_target"))
            == "conze_viswanathan_analytical",
            cross_validate.get("relations") == {"mc_lookback": "within_tolerance"},
            cross_validate.get("tolerance_pct") == 1.25,
            set(targets) == set(expected_targets),
            all(
                isinstance(targets.get(target_id), Mapping)
                and set(targets[target_id])
                == {*expected, "variant_parameters"}
                and all(
                    targets[target_id].get(key) == value
                    for key, value in expected.items()
                )
                for target_id, expected in expected_targets.items()
            ),
            isinstance(targets.get("mc_lookback"), Mapping)
            and targets["mc_lookback"].get("variant_parameters")
            == {
                "process": "exact_gbm",
                "extremum_sampling": "conditional_log_bridge",
            },
            isinstance(targets.get("conze_viswanathan_analytical"), Mapping)
            and targets["conze_viswanathan_analytical"].get("variant_parameters")
            == {"formula": "fixed_strike_continuous_lookback"},
        )
    )
    if valid:
        return []
    return [
        _issue(
            manifest_name,
            "legacy.lookback_invalid_contract",
            "T30/T96 require the bounded fixed-strike continuous lookback comparison contract",
            task_id=_text(task.get("id")),
            path=path,
        )
    ]


def _validate_legacy_expected_honest_block(
    manifest_name: str,
    task: Mapping[str, Any],
    path: str,
) -> list[TaskManifestIssue]:
    """Validate one legacy row that is executable only as an honest block."""
    issues: list[TaskManifestIssue] = []
    task_id = _text(task.get("id"))
    contract = task.get("honest_block_contract")
    required_contract_fields = (
        "reason",
        "summary",
        "packet_type",
        "missing_capabilities",
        "suggested_action",
    )
    valid_contract = isinstance(contract, Mapping) and all(
        (
            _nonempty_string_sequence(contract.get(field))
            if field == "missing_capabilities"
            else bool(_text(contract.get(field)))
        )
        for field in required_contract_fields
    )
    if (
        not _text(task.get("disposition_reason"))
        or _text(task.get("expected_outcome")) != "honest_block"
        or not _nonempty_string_sequence(task.get("expected_blocker_ids"))
        or not valid_contract
    ):
        issues.append(
            _issue(
                manifest_name,
                "legacy.invalid_honest_block",
                "expected_honest_block requires an authored reason, blockers, and complete repair contract",
                task_id=task_id,
                path=path,
            )
        )

    if task_id == "T09":
        blocker_ids = set(
            str(item).strip()
            for item in (task.get("expected_blocker_ids") or ())
            if isinstance(item, str) and item.strip()
        )
        capabilities = set(
            str(item).strip()
            for item in (
                contract.get("missing_capabilities", ())
                if isinstance(contract, Mapping)
                else ()
            )
            if isinstance(item, str) and item.strip()
        )
        if (
            "semantic_product_contract_gap:variable_coupon_schedule" not in blocker_ids
            or "variable_coupon_schedule" not in capabilities
            or not isinstance(contract, Mapping)
            or _text(contract.get("reason"))
            != "callable_bond_variable_coupon_schedule_missing"
            or _text(contract.get("packet_type")) != "semantic_product_contract_gap"
            or _text(contract.get("follow_on_issue")) != "QUA-1251"
        ):
            issues.append(
                _issue(
                    manifest_name,
                    "legacy.t09_invalid_honest_block",
                    "T09 must name the variable-coupon schedule gap and QUA-1251 follow-on",
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


def _iso_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _iso_date_sequence(value: Any) -> tuple[date, ...] | None:
    if not _nonempty_sequence(value):
        return None
    parsed = tuple(_iso_date(item) for item in value)
    if any(item is None for item in parsed):
        return None
    return tuple(item for item in parsed if item is not None)


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
