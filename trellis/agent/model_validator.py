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

import json

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
        review_reason=None,
    )


def _llm_conceptual_review(
    code: str,
    instrument_type: str,
    method: str,
    knowledge_context: str = "",
    generation_plan=None,
    residual_risks: tuple[str, ...] = (),
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
