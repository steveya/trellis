"""Model Validation Agent — independent assessment of model quality.

Mirrors the Model Risk Management (MRM) function in banks:
- Conceptual soundness review
- Calibration quality checks
- Sensitivity analysis
- Benchmarking against alternative implementations
- Limitation analysis

Issues formal findings (Critical/High/Medium/Low) that the builder must remediate.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, Mapping

from trellis.agent.review_policy import determine_review_policy
from trellis.agent.validation_report import ValidationFinding, ValidationReport
from trellis.agent.validation_tests import (
    check_benchmark,
    check_calibration,
    check_sensitivity_signs,
)


def validate_model(
    payoff_factory,
    market_state_factory,
    code: str,
    instrument_type: str,
    method: str,
    knowledge_context: str = "",
    model: str | None = None,
    product_ir=None,
    generation_plan=None,
    validation_contract=None,
    deterministic_evidence_packet: Mapping[str, object] | None = None,
    review_reason: str | None = None,
    run_llm_review: bool | None = None,
) -> ValidationReport:
    """Run full model validation — automated tests + LLM review.

    Parameters
    ----------
    payoff_factory : callable() -> Payoff
    market_state_factory : callable(rate, vol) -> MarketState
    code : str
        The generated source code.
    instrument_type : str
        e.g. "callable_bond", "barrier_option"
    method : str
        e.g. "rate_tree", "monte_carlo"
    model : str or None
        LLM model for the validation agent.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport(instrument=instrument_type, method=method)

    # 1. Automated sensitivity tests
    sensitivity_findings = check_sensitivity_signs(
        payoff_factory, market_state_factory, instrument_type,
    )
    report.findings.extend(sensitivity_findings)

    # 2. Automated benchmark tests
    benchmark_findings = check_benchmark(
        payoff_factory, market_state_factory, instrument_type,
    )
    report.findings.extend(benchmark_findings)

    # 3. LLM-based conceptual review
    if run_llm_review is None:
        policy = determine_review_policy(
            validation="thorough",
            method=method,
            instrument_type=instrument_type,
            product_ir=product_ir,
            validation_contract=validation_contract,
        )
        run_llm_review = policy.run_model_validator_llm
    if run_llm_review:
        llm_findings = _llm_conceptual_review(
            code,
            instrument_type,
            method,
            knowledge_context=knowledge_context,
            generation_plan=generation_plan,
            residual_risks=_model_validator_residual_risks(validation_contract),
            quant_challenger_packet=_model_validator_quant_challenger_packet(
                validation_contract
            ),
            deterministic_evidence_packet=deterministic_evidence_packet,
            review_reason=review_reason,
            model=model,
        )
        report.findings.extend(llm_findings)

    # Set approval status
    report.approved = not report.has_blockers

    return report


def validate_model_for_request(
    compiled_request,
    payoff_factory,
    market_state_factory,
    code: str,
    knowledge_context: str = "",
    model: str | None = None,
) -> ValidationReport:
    """Validate a model using the canonical compiled-request context."""
    instrument = (
        getattr(compiled_request.product_ir, "instrument", None)
        or compiled_request.request.instrument_type
        or "unknown"
    )
    method = (
        compiled_request.execution_plan.route_method
        or getattr(compiled_request.pricing_plan, "method", None)
        or "unknown"
    )
    return validate_model(
        payoff_factory=payoff_factory,
        market_state_factory=market_state_factory,
        code=code,
        instrument_type=instrument,
        method=method,
        knowledge_context=knowledge_context,
        model=model,
        product_ir=compiled_request.product_ir,
        generation_plan=getattr(compiled_request, "generation_plan", None),
        validation_contract=getattr(compiled_request, "validation_contract", None),
        deterministic_evidence_packet=getattr(
            compiled_request,
            "deterministic_evidence_packet",
            None,
        ),
        review_reason=None,
    )


def _llm_conceptual_review(
    code: str,
    instrument_type: str,
    method: str,
    knowledge_context: str = "",
    generation_plan=None,
    residual_risks: tuple[str, ...] = (),
    quant_challenger_packet: dict[str, object] | None = None,
    deterministic_evidence_packet: Mapping[str, object] | None = None,
    review_reason: str | None = None,
    model: str | None = None,
) -> list[ValidationFinding]:
    """LLM-based model validation — conceptual soundness review."""
    from trellis.agent.config import llm_generate_json, get_default_model

    model = model or get_default_model()

    knowledge_section = ""
    if knowledge_context.strip():
        knowledge_section = f"\n## Shared Knowledge\n{knowledge_context}\n"
    residual_risk_section = ""
    if residual_risks:
        residual_risk_lines = "\n".join(f"- `{risk}`" for risk in residual_risks)
        residual_risk_section = (
            "\n## Residual Conceptual Risks\n"
            "Residual conceptual review only. Focus on the unresolved items below.\n"
            f"{residual_risk_lines}\n"
        )
    quant_packet_section = ""
    if quant_challenger_packet:
        quant_packet_section = (
            "\n## Quant Challenger Packet\n"
            "Use this as the method-selection contract. Do not reconstruct "
            "quant reasoning from prose when these fields are present.\n"
            f"```json\n{json.dumps(quant_challenger_packet, indent=2, sort_keys=True)}\n```\n"
        )
    deterministic_evidence_section = ""
    if deterministic_evidence_packet:
        deterministic_evidence_section = (
            "\n## Executed Deterministic Evidence\n"
            "These checks already own deterministic validation claims. "
            "Do not convert passed deterministic evidence into prose findings. "
            "Only review residual conceptual and calibration risks left after this evidence.\n"
            f"```json\n{json.dumps(dict(deterministic_evidence_packet), indent=2, sort_keys=True)}\n```\n"
        )
    review_reason_section = ""
    if review_reason:
        review_reason_section = f"\n## Review Trigger\n- `{review_reason}`\n"
    route_contract_section = ""
    if generation_plan is not None:
        from trellis.agent.codegen_guardrails import render_review_contract_card

        route_contract_section = "\n" + render_review_contract_card(generation_plan) + "\n"

    prompt = f"""You are a model validation analyst at a quantitative finance firm.
Your role is to independently assess whether a pricing model is conceptually
sound, correctly calibrated, and produces reasonable risk measures.

You are NOT reviewing code quality — the code reviewer already did that.
You are reviewing MODEL quality.
Do not repeat deterministic checks such as non-negativity, basic sensitivity,
reference bounds, or other validations already covered by the validation contract.

## Instrument type: {instrument_type}
## Pricing method: {method}
{knowledge_section}
{route_contract_section}
{residual_risk_section}
{quant_packet_section}
{deterministic_evidence_section}
{review_reason_section}

## Code to validate
```python
{code}
```

## What to check

1. CONCEPTUAL SOUNDNESS:
   - Is {method} the right approach for {instrument_type}?
   - Are the dynamics assumptions appropriate? (lognormal vs normal, mean-reverting vs not)
   - Are all relevant risk factors captured?

2. CALIBRATION:
   - Does the model calibrate to market instruments?
   - For rate trees: does it solve for theta(t) to reprice the yield curve?
   - For MC: is the process calibrated to match the vol surface?

3. NUMERICAL QUALITY:
   - Are there enough time steps / paths for convergence?
   - Are cashflows embedded at actual dates (not spread uniformly)?
   - Are boundary conditions correct?

4. LIMITATIONS:
   - What scenarios would this model handle poorly?
   - What approximations were made?

## Output
Return a JSON array of findings. Each finding:
{{
    "severity": "critical" | "high" | "medium" | "low",
    "category": "conceptual" | "calibration" | "implementation" | "limitation",
    "description": "what's wrong or could be improved",
    "evidence": "specific code reference or reasoning",
    "remediation": "what to change"
}}

Be specific. Reference line numbers or variable names.
Only issue critical/high for genuine pricing errors.
Return ONLY the JSON array."""

    try:
        data = llm_generate_json(prompt, model=model)
    except Exception:
        from trellis.agent.config import llm_generate
        try:
            text = llm_generate(prompt, model=model)
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                return []
        except Exception:
            return []

    findings = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            findings.append(ValidationFinding(
                id=f"MV-L{i+1:03d}",
                severity=item.get("severity", "medium"),
                category=item.get("category", "conceptual"),
                description=item.get("description", ""),
                evidence=item.get("evidence", ""),
                remediation=item.get("remediation", ""),
            ))
    return findings


def _model_validator_residual_risks(validation_contract) -> tuple[str, ...]:
    """Return the residual conceptual risk ids that still justify LLM review."""
    if validation_contract is None:
        return ()
    return tuple(getattr(validation_contract, "residual_risks", ()) or ())


def _model_validator_quant_challenger_packet(validation_contract) -> dict[str, object]:
    """Return the structured quant challenger packet attached to validation."""
    if validation_contract is None:
        return {}
    return dict(getattr(validation_contract, "quant_challenger_packet", {}) or {})


def build_model_validation_evidence_packet(
    *,
    validation_contract=None,
    validation_bundle_execution=None,
    reference_oracle=None,
    arbiter_verdicts=(),
) -> dict[str, object]:
    """Build the deterministic evidence packet handed to model validation."""
    contract_summary = _validation_contract_evidence_summary(validation_contract)
    bundle_summary = _validation_bundle_execution_summary(validation_bundle_execution)
    oracle_summary = _summary_dict(reference_oracle)
    arbiter_summary = _arbiter_evidence_summary(arbiter_verdicts)
    packet: dict[str, object] = {}
    if contract_summary:
        packet["validation_contract"] = contract_summary
    if bundle_summary:
        packet["validation_bundle"] = bundle_summary
    if oracle_summary:
        packet["reference_oracle"] = oracle_summary
    if arbiter_summary:
        packet["arbiter"] = arbiter_summary
    deterministic_blockers = _deterministic_blockers_from_packet(packet)
    if deterministic_blockers:
        packet["deterministic_blockers"] = deterministic_blockers
    return packet


def classify_model_validation_findings(
    findings,
) -> dict[str, list[dict[str, object]]]:
    """Classify model-validator findings into residual report buckets."""
    classification = {
        "conceptual_blockers": [],
        "calibration_blockers": [],
        "residual_limitations": [],
    }
    for finding in findings or ():
        summary = _finding_summary(finding)
        severity = str(summary.get("severity", "") or "").strip().lower()
        category = str(summary.get("category", "") or "").strip().lower()
        is_blocker = severity in {"critical", "high"}
        if category == "calibration" and is_blocker:
            classification["calibration_blockers"].append(summary)
        elif category == "conceptual" and is_blocker:
            classification["conceptual_blockers"].append(summary)
        elif category in {"limitation", "residual_limitation"} or not is_blocker:
            classification["residual_limitations"].append(summary)
        elif is_blocker:
            classification["conceptual_blockers"].append(summary)
    return classification


def _validation_contract_evidence_summary(validation_contract) -> dict[str, object]:
    """Project the validation contract fields relevant to residual review."""
    if validation_contract is None:
        return {}
    return {
        "contract_id": getattr(validation_contract, "contract_id", ""),
        "deterministic_checks": [
            {
                "check_id": getattr(check, "check_id", ""),
                "category": getattr(check, "category", ""),
                "relation": getattr(check, "relation", None),
            }
            for check in getattr(validation_contract, "deterministic_checks", ()) or ()
        ],
        "comparison_relations": [
            {
                "target_id": getattr(relation, "target_id", ""),
                "relation": getattr(relation, "relation", ""),
                "source": getattr(relation, "source", ""),
            }
            for relation in getattr(validation_contract, "comparison_relations", ()) or ()
        ],
        "lowering_errors": list(getattr(validation_contract, "lowering_errors", ()) or ()),
        "admissibility_failures": list(
            getattr(validation_contract, "admissibility_failures", ()) or ()
        ),
        "residual_risks": list(getattr(validation_contract, "residual_risks", ()) or ()),
    }


def _validation_bundle_execution_summary(execution) -> dict[str, object]:
    """Project validation-bundle execution into prompt-safe primitives."""
    if execution is None:
        return {}
    data = _summary_dict(execution)
    failures = list(data.get("failures") or ())
    failure_details = [
        _summary_dict(item) for item in data.get("failure_details") or ()
    ]
    failure_count = len(failures)
    if not failures and isinstance(data.get("failure_count"), (int, float)):
        failure_count = max(int(data.get("failure_count") or 0), 0)
    return {
        "executed_checks": list(data.get("executed_checks") or ()),
        "skipped_checks": list(data.get("skipped_checks") or ()),
        "failure_count": failure_count,
        "failures": failures,
        "failure_details": failure_details,
    }


def _arbiter_evidence_summary(arbiter_verdicts) -> dict[str, object]:
    """Project arbiter verdicts into prompt-safe primitives."""
    verdicts = [_summary_dict(verdict) for verdict in arbiter_verdicts or ()]
    if not verdicts:
        return {}
    return {
        "verdicts": verdicts,
        "failure_count": sum(
            1 for verdict in verdicts if verdict.get("status") == "failed"
        ),
        "executed_count": sum(1 for verdict in verdicts if verdict.get("executed")),
    }


def _deterministic_blockers_from_packet(
    packet: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return deterministic blockers already owned outside model validation."""
    blockers: list[dict[str, object]] = []
    contract = dict(packet.get("validation_contract") or {})
    for key in ("lowering_errors", "admissibility_failures"):
        for item in contract.get(key) or ():
            blockers.append({"source": "validation_contract", "kind": key, "message": item})

    bundle = dict(packet.get("validation_bundle") or {})
    for detail in bundle.get("failure_details") or ():
        data = dict(detail or {})
        blockers.append({
            "source": "validation_bundle",
            "check_id": data.get("check") or data.get("check_id"),
            "message": data.get("message") or data.get("exception_message") or "",
        })
    if bundle.get("failure_count") and not bundle.get("failure_details"):
        blockers.append({
            "source": "validation_bundle",
            "failure_count": bundle.get("failure_count"),
        })

    oracle = dict(packet.get("reference_oracle") or {})
    if oracle and not bool(oracle.get("passed", True)):
        blockers.append({
            "source": "reference_oracle",
            "oracle_id": oracle.get("oracle_id"),
            "message": oracle.get("failure_message") or "",
        })

    arbiter = dict(packet.get("arbiter") or {})
    for verdict in arbiter.get("verdicts") or ():
        data = dict(verdict or {})
        if data.get("status") == "failed":
            blockers.append({
                "source": "arbiter",
                "check_id": data.get("check_id"),
                "reason": data.get("reason"),
                "message": data.get("message") or data.get("detail") or "",
            })
    return blockers


def _finding_summary(finding) -> dict[str, object]:
    """Project one ValidationFinding or finding-like mapping into primitives."""
    if isinstance(finding, Mapping):
        data = dict(finding)
    elif is_dataclass(finding):
        data = asdict(finding)
    else:
        data = {
            "id": getattr(finding, "id", ""),
            "severity": getattr(finding, "severity", ""),
            "category": getattr(finding, "category", ""),
            "description": getattr(finding, "description", ""),
            "evidence": getattr(finding, "evidence", ""),
            "remediation": getattr(finding, "remediation", ""),
        }
    return {
        "source": "model_validator",
        "id": data.get("id", ""),
        "severity": data.get("severity", ""),
        "category": data.get("category", ""),
        "description": data.get("description", ""),
        "evidence": data.get("evidence", ""),
        "remediation": data.get("remediation", ""),
    }


def _summary_dict(value) -> dict[str, Any]:
    """Return a shallow primitive dictionary for dataclasses and mappings."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    data: dict[str, Any] = {}
    for key in (
        "contract_id",
        "executed_checks",
        "skipped_checks",
        "failures",
        "failure_details",
        "failure_count",
        "oracle_id",
        "instrument_type",
        "method",
        "source",
        "relation",
        "tolerance",
        "passed",
        "sampled_prices",
        "max_abs_deviation",
        "max_rel_deviation",
        "failure_message",
        "check_id",
        "status",
        "reason",
        "executed",
        "severity",
        "description",
        "message",
        "detail",
    ):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    return data
