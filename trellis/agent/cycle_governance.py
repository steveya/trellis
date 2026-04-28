"""Promotion/adoption governance derived from stable agent cycle reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_BLOCKING_BUCKETS = (
    "deterministic_blockers",
    "conceptual_blockers",
    "calibration_blockers",
)


@dataclass(frozen=True)
class CyclePromotionGovernance:
    """Serializable promotion/adoption gate result for one cycle report."""

    eligible: bool
    decision: str
    prerequisites: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    cycle_report: Mapping[str, Any]
    adapter_lifecycle: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-safe governance artifact."""
        return {
            "eligible": self.eligible,
            "decision": self.decision,
            "prerequisites": list(self.prerequisites),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "cycle_report": dict(self.cycle_report),
            "adapter_lifecycle": dict(self.adapter_lifecycle),
        }


def evaluate_cycle_promotion_governance(
    cycle_report: Mapping[str, Any] | None,
    *,
    adapter_lifecycle: Mapping[str, Any] | None = None,
    require_cycle_report: bool = True,
) -> CyclePromotionGovernance:
    """Evaluate whether a build cycle is eligible for promotion/adoption.

    The function is deliberately deterministic and low-cardinality. It consumes
    the stable ``cycle_report`` projection produced by platform traces rather
    than parsing critic/model-validator prose.
    """
    report = dict(cycle_report or {}) if isinstance(cycle_report, Mapping) else {}
    lifecycle_summary = _adapter_lifecycle_summary(adapter_lifecycle)
    blockers: list[str] = []
    warnings: list[str] = []
    prerequisites = [
        "cycle_report_present",
        "cycle_success_true",
        "no_failed_cycle_stages",
        "no_blocking_cycle_buckets",
    ]

    if not report:
        if require_cycle_report:
            blockers.append("cycle_report_missing")
        else:
            warnings.append("cycle_report_missing")
        return CyclePromotionGovernance(
            eligible=not blockers,
            decision="blocked" if blockers else "advisory",
            prerequisites=tuple(prerequisites),
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            cycle_report={},
            adapter_lifecycle=lifecycle_summary,
        )

    if report.get("success") is not True:
        blockers.append("cycle_success_not_true")

    stage_statuses = report.get("stage_statuses") or {}
    if isinstance(stage_statuses, Mapping):
        for stage, status in sorted(stage_statuses.items()):
            if str(status or "").strip().lower() == "failed":
                blockers.append(f"cycle_stage_failed:{stage}")
    else:
        warnings.append("cycle_stage_statuses_missing")

    failure_count = _int_value(report.get("failure_count"))
    if failure_count > 0:
        blockers.append("cycle_failure_count_nonzero")

    for bucket in _BLOCKING_BUCKETS:
        count = _item_count(report.get(bucket))
        if count:
            blockers.append(f"{bucket}_present")

    residual_risks = _string_list(report.get("residual_risks"))
    if residual_risks:
        warnings.append("residual_risks_present")
    if _item_count(report.get("residual_limitations")):
        warnings.append("residual_limitations_present")

    lifecycle_counts = lifecycle_summary.get("status_counts") or {}
    if isinstance(lifecycle_counts, Mapping):
        for status in ("stale", "deprecated", "archived"):
            count = _int_value(lifecycle_counts.get(status))
            if count:
                warnings.append(f"adapter_lifecycle_{status}:{count}")

    blockers = _unique(blockers)
    warnings = _unique(warnings)
    return CyclePromotionGovernance(
        eligible=not blockers,
        decision="eligible" if not blockers else "blocked",
        prerequisites=tuple(prerequisites),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        cycle_report=_cycle_report_summary(report),
        adapter_lifecycle=lifecycle_summary,
    )


def _cycle_report_summary(report: Mapping[str, Any]) -> dict[str, object]:
    """Return the compact cycle fields promotion/adoption artifacts need."""
    stage_statuses = report.get("stage_statuses") or {}
    if not isinstance(stage_statuses, Mapping):
        stage_statuses = {}
    return {
        "request_id": str(report.get("request_id") or ""),
        "status": str(report.get("status") or ""),
        "outcome": str(report.get("outcome") or ""),
        "success": report.get("success"),
        "pricing_method": str(report.get("pricing_method") or ""),
        "validation_contract_id": str(report.get("validation_contract_id") or ""),
        "stage_statuses": {str(key): str(value) for key, value in stage_statuses.items()},
        "failure_count": _int_value(report.get("failure_count")),
        "blocker_counts": {
            bucket: _item_count(report.get(bucket))
            for bucket in _BLOCKING_BUCKETS
        },
        "residual_limitations_count": _item_count(report.get("residual_limitations")),
        "residual_risks": _string_list(report.get("residual_risks")),
    }


def _adapter_lifecycle_summary(lifecycle: Mapping[str, Any] | None) -> dict[str, object]:
    """Extract low-cardinality adapter lifecycle state from review/adoption payloads."""
    if not isinstance(lifecycle, Mapping):
        return {
            "status_counts": {},
            "stale_adapter_ids": [],
            "deprecated_adapter_ids": [],
            "archived_adapter_ids": [],
        }

    summary = lifecycle.get("summary")
    if not isinstance(summary, Mapping):
        resolved = lifecycle.get("resolved")
        if isinstance(resolved, Mapping):
            summary = resolved.get("summary")
    if not isinstance(summary, Mapping):
        summary = lifecycle

    status_counts = summary.get("status_counts") if isinstance(summary, Mapping) else {}
    if not isinstance(status_counts, Mapping):
        status_counts = {}
    return {
        "status_counts": {
            str(key): _int_value(value)
            for key, value in status_counts.items()
        },
        "stale_adapter_ids": _string_list(summary.get("stale_adapter_ids")),
        "deprecated_adapter_ids": _string_list(summary.get("deprecated_adapter_ids")),
        "archived_adapter_ids": _string_list(summary.get("archived_adapter_ids")),
    }


def _item_count(value: object) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    return 0


def _int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
    else:
        values = [value]
    return _unique(str(item).strip() for item in values if str(item).strip())


def _unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "CyclePromotionGovernance",
    "evaluate_cycle_promotion_governance",
]
