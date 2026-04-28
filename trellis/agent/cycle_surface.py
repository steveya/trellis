"""Product-facing projections for governed agent-review cycle evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


SCHEMA_VERSION = "agent_cycle_result.v1"
SCORECARD_SCHEMA_VERSION = "agent_cycle_scorecard.v1"

_STAGE_ORDER = (
    "quant",
    "validation_bundle",
    "reference_oracle",
    "critic",
    "arbiter",
    "model_validator",
)

_BLOCKING_BUCKETS = (
    "deterministic_blockers",
    "conceptual_blockers",
    "calibration_blockers",
)

_EVIDENCE_BUCKETS = (
    *_BLOCKING_BUCKETS,
    "residual_limitations",
    "residual_risks",
)

_CERTIFIES = (
    "governed agent-review cycle completed",
    "deterministic validation evidence surfaced",
    "critic/arbiter/model-validator outcomes reported",
)

_DOES_NOT_CERTIFY = (
    "external model approval",
    "regulatory certification",
    "xVA or counterparty-risk coverage",
    "FpML product coverage",
    "correctness beyond the recorded validation scope",
)


def build_cycle_result_surface(
    cycle_report: Mapping[str, Any] | None,
    *,
    promotion_governance: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Project raw cycle evidence into a stable UI/API result surface.

    The surface is intentionally conservative. It explains what the current
    governed cycle can evidence while spelling out what it cannot certify.
    """
    report = _mapping_or_empty(cycle_report)
    governance = _mapping_or_empty(promotion_governance)
    if not report:
        return _missing_surface(governance)

    stage_statuses = _stage_statuses(report)
    evidence_counts = {
        bucket: _item_count(report.get(bucket))
        for bucket in _EVIDENCE_BUCKETS
    }
    blockers = _blockers(report, stage_statuses, evidence_counts, governance)
    warnings = _warnings(report, governance)
    status = _surface_status(report, blockers)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "status": status,
        "headline": _headline(status),
        "summary": _summary(report, status, evidence_counts),
        "request_id": str(report.get("request_id") or "").strip(),
        "outcome": str(report.get("outcome") or "").strip(),
        "success": report.get("success"),
        "pricing_method": str(report.get("pricing_method") or "").strip(),
        "validation_contract_id": str(report.get("validation_contract_id") or "").strip(),
        "stage_statuses": stage_statuses,
        "stage_summary": _stage_summary(report, stage_statuses),
        "evidence_counts": evidence_counts,
        "blockers": blockers,
        "warnings": warnings,
        "promotion": _promotion_surface(governance),
        "claim": _claim_surface(),
        "operator_actions": _operator_actions(status, blockers, warnings),
    }


def summarize_cycle_behavior(records: Iterable[Mapping[str, Any]]) -> dict[str, object]:
    """Aggregate cycle availability, outcomes, and stage trigger rates."""
    records_list = [dict(record) for record in records]
    stage_counts = {
        stage: {
            "triggered_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "trigger_rate": 0.0,
        }
        for stage in _STAGE_ORDER
    }
    blocker_counts = {bucket: 0 for bucket in _BLOCKING_BUCKETS}
    residual_limitations_count = 0
    residual_risk_count = 0
    available_count = 0
    passed_count = 0
    failed_count = 0
    incomplete_count = 0

    for record in records_list:
        report = _cycle_report_from_record(record)
        surface = build_cycle_result_surface(report)
        status = str(surface.get("status") or "")
        if not surface.get("available"):
            continue
        available_count += 1
        if status == "passed":
            passed_count += 1
        elif status == "failed":
            failed_count += 1
        elif status == "incomplete":
            incomplete_count += 1

        evidence_counts = _mapping_or_empty(surface.get("evidence_counts"))
        for bucket in _BLOCKING_BUCKETS:
            blocker_counts[bucket] += _int_value(evidence_counts.get(bucket))
        residual_limitations_count += _int_value(evidence_counts.get("residual_limitations"))
        residual_risk_count += _int_value(evidence_counts.get("residual_risks"))

        for stage, stage_status in _stage_statuses(report).items():
            if stage not in stage_counts:
                stage_counts[stage] = {
                    "triggered_count": 0,
                    "passed_count": 0,
                    "failed_count": 0,
                    "skipped_count": 0,
                    "trigger_rate": 0.0,
                }
            stage_counts[stage]["triggered_count"] += 1
            if stage_status == "passed":
                stage_counts[stage]["passed_count"] += 1
            elif stage_status == "failed":
                stage_counts[stage]["failed_count"] += 1
            elif stage_status == "skipped":
                stage_counts[stage]["skipped_count"] += 1

    for counts in stage_counts.values():
        counts["trigger_rate"] = (
            round(counts["triggered_count"] / available_count, 6)
            if available_count
            else 0.0
        )

    run_count = len(records_list)
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run_count": run_count,
        "available_count": available_count,
        "not_available_count": run_count - available_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "incomplete_count": incomplete_count,
        "stage_trigger_rates": stage_counts,
        "blocker_counts": blocker_counts,
        "residual_limitations_count": residual_limitations_count,
        "residual_risk_count": residual_risk_count,
    }


def _missing_surface(governance: Mapping[str, Any]) -> dict[str, object]:
    blockers = []
    if governance:
        blockers = _string_list(governance.get("blockers"))
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "status": "not_available",
        "headline": "No governed agent-review cycle report is available.",
        "summary": "No stable quant/critic/arbiter/model-validator cycle evidence was attached to this result.",
        "request_id": "",
        "outcome": "",
        "success": None,
        "pricing_method": "",
        "validation_contract_id": "",
        "stage_statuses": {},
        "stage_summary": [],
        "evidence_counts": {bucket: 0 for bucket in _EVIDENCE_BUCKETS},
        "blockers": blockers,
        "warnings": _string_list(governance.get("warnings")) if governance else [],
        "promotion": _promotion_surface(governance),
        "claim": _claim_surface(extra_non_claims=("agent-cycle evidence",)),
        "operator_actions": [
            "Inspect platform trace artifacts or rerun through the governed agent-review cycle before promotion claims."
        ],
    }


def _surface_status(report: Mapping[str, Any], blockers: list[str]) -> str:
    if blockers:
        return "failed"
    if report.get("success") is True:
        return "passed"
    return "incomplete"


def _headline(status: str) -> str:
    if status == "passed":
        return "Governed agent-review cycle passed."
    if status == "failed":
        return "Governed agent-review cycle found blocking issues."
    return "Governed agent-review cycle is incomplete."


def _summary(
    report: Mapping[str, Any],
    status: str,
    evidence_counts: Mapping[str, int],
) -> str:
    method = str(report.get("pricing_method") or "unknown").strip() or "unknown"
    contract_id = str(report.get("validation_contract_id") or "no validation contract").strip()
    blocking_count = sum(_int_value(evidence_counts.get(bucket)) for bucket in _BLOCKING_BUCKETS)
    residual_count = _int_value(evidence_counts.get("residual_limitations"))
    return (
        f"Cycle {status} for method {method} under {contract_id}; "
        f"{blocking_count} blocking evidence bucket item(s), "
        f"{residual_count} residual limitation item(s)."
    )


def _stage_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    statuses = report.get("stage_statuses")
    if not isinstance(statuses, Mapping):
        return {}
    normalized = {
        str(stage).strip(): str(status).strip().lower()
        for stage, status in statuses.items()
        if str(stage).strip()
    }
    return {
        stage: normalized[stage]
        for stage in sorted(normalized, key=_stage_sort_key)
    }


def _stage_summary(
    report: Mapping[str, Any],
    stage_statuses: Mapping[str, str],
) -> list[dict[str, object]]:
    raw_stages = report.get("stages")
    if isinstance(raw_stages, list):
        summaries = []
        for item in raw_stages:
            if not isinstance(item, Mapping):
                continue
            stage = str(item.get("stage") or "").strip()
            if not stage:
                continue
            summaries.append(
                {
                    "stage": stage,
                    "status": str(item.get("status") or stage_statuses.get(stage) or "").strip().lower(),
                    "summary": str(item.get("summary") or "").strip(),
                }
            )
        if summaries:
            return sorted(summaries, key=lambda item: _stage_sort_key(str(item["stage"])))
    return [
        {"stage": stage, "status": status, "summary": ""}
        for stage, status in stage_statuses.items()
    ]


def _blockers(
    report: Mapping[str, Any],
    stage_statuses: Mapping[str, str],
    evidence_counts: Mapping[str, int],
    governance: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if report.get("success") is False:
        blockers.append("cycle_success_not_true")
    for stage, status in stage_statuses.items():
        if status == "failed":
            blockers.append(f"cycle_stage_failed:{stage}")
    if _int_value(report.get("failure_count")) > 0:
        blockers.append("cycle_failure_count_nonzero")
    for bucket in _BLOCKING_BUCKETS:
        if _int_value(evidence_counts.get(bucket)) > 0:
            blockers.append(f"{bucket}_present")
    blockers.extend(_string_list(governance.get("blockers")) if governance else [])
    return _unique(blockers)


def _warnings(report: Mapping[str, Any], governance: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _item_count(report.get("residual_limitations")):
        warnings.append("residual_limitations_present")
    if _item_count(report.get("residual_risks")):
        warnings.append("residual_risks_present")
    warnings.extend(_string_list(governance.get("warnings")) if governance else [])
    return _unique(warnings)


def _promotion_surface(governance: Mapping[str, Any]) -> dict[str, object]:
    if not governance:
        return {
            "eligible": None,
            "decision": "not_evaluated",
            "blockers": [],
            "warnings": [],
        }
    return {
        "eligible": governance.get("eligible"),
        "decision": str(governance.get("decision") or "").strip(),
        "blockers": _string_list(governance.get("blockers")),
        "warnings": _string_list(governance.get("warnings")),
    }


def _claim_surface(*, extra_non_claims: Iterable[str] = ()) -> dict[str, object]:
    return {
        "certifies": list(_CERTIFIES),
        "does_not_certify": _unique((*extra_non_claims, *_DOES_NOT_CERTIFY)),
    }


def _operator_actions(status: str, blockers: list[str], warnings: list[str]) -> list[str]:
    if status == "failed":
        return [
            "Resolve blocking cycle evidence before promotion or desk-safe reuse.",
            "Inspect deterministic, conceptual, and calibration blocker buckets in the audit bundle.",
        ]
    if status == "incomplete":
        return ["Complete or rerun the governed agent-review cycle before making cycle-based claims."]
    if warnings:
        return ["Review residual limitations and warnings before production use."]
    return ["Use audit references to inspect full cycle evidence when needed."]


def _cycle_report_from_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(record.get("stage_statuses"), Mapping):
        return record
    for key in ("cold_agent_cycle_report", "cycle_report"):
        value = record.get(key)
        if isinstance(value, Mapping) and value:
            return value
    observability = record.get("build_observability")
    if isinstance(observability, Mapping):
        value = observability.get("cycle_report")
        if isinstance(value, Mapping) and value:
            return value
    return None


def _stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        index = _STAGE_ORDER.index(stage)
    except ValueError:
        index = 999
    return index, stage


def _mapping_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


__all__ = [
    "build_cycle_result_surface",
    "summarize_cycle_behavior",
]
