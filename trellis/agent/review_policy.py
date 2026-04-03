"""Deterministic review-escalation policy for build validation."""

from __future__ import annotations

from dataclasses import dataclass


_HIGH_RISK_EXERCISE = {"issuer_call", "holder_put", "bermudan", "american"}
_HIGH_RISK_STATE = {"path_dependent", "schedule_dependent"}
_HIGH_RISK_MODEL_FAMILIES = {
    "interest_rate",
    "stochastic_volatility",
    "jump_diffusion",
    "credit_copula",
    "cashflow_structured",
}
_HIGH_RISK_TRAITS = {
    "callable",
    "puttable",
    "bermudan",
    "american",
    "asian",
    "barrier",
    "lookback",
    "path_dependent",
    "range_condition",
    "stochastic_vol",
    "jump_diffusion",
    "prepayment",
}


@dataclass(frozen=True)
class ReviewPolicy:
    """Deterministic policy for critic and model-validator escalation."""

    risk_level: str
    run_critic: bool
    run_model_validator_llm: bool
    critic_reason: str
    model_validator_reason: str
    critic_mode: str = "skip"
    critic_json_max_retries: int | None = None
    critic_allow_text_fallback: bool = False
    critic_text_max_retries: int | None = None


def determine_review_policy(
    *,
    validation: str,
    method: str,
    instrument_type: str | None = None,
    product_ir=None,
    validation_contract=None,
) -> ReviewPolicy:
    """Classify one build into a deterministic review policy.

    Low-risk routes skip LLM reviewer stages by default. Higher-risk routes keep
    the current critic / conceptual model-review path.
    """
    low_risk_reason = _low_risk_reason(
        method=method,
        instrument_type=instrument_type,
        product_ir=product_ir,
    )
    risk_level = "low" if low_risk_reason else "high"
    blocking_reason = _validation_contract_blocking_reason(validation_contract)
    contract_review_reason = _validation_contract_review_reason(validation_contract)

    if validation not in {"standard", "thorough"}:
        return ReviewPolicy(
            risk_level=risk_level,
            run_critic=False,
            run_model_validator_llm=False,
            critic_reason="validation_mode_skipped",
            model_validator_reason="validation_mode_skipped",
            critic_mode="skip",
        )

    if blocking_reason:
        return ReviewPolicy(
            risk_level="blocked",
            run_critic=False,
            run_model_validator_llm=False,
            critic_reason=blocking_reason,
            model_validator_reason=blocking_reason,
            critic_mode="skip",
        )

    if low_risk_reason and contract_review_reason is None:
        return ReviewPolicy(
            risk_level="low",
            run_critic=False,
            run_model_validator_llm=False,
            critic_reason=low_risk_reason,
            model_validator_reason=low_risk_reason,
            critic_mode="skip",
        )

    review_reason = contract_review_reason or "high_risk_route_requires_llm_review"
    return ReviewPolicy(
        risk_level="high",
        run_critic=True,
        run_model_validator_llm=(validation == "thorough"),
        critic_reason=review_reason,
        model_validator_reason=(
            review_reason
            if validation == "thorough"
            else "validation_mode_skipped"
        ),
        critic_mode="required" if validation == "thorough" else "advisory",
        critic_json_max_retries=None if validation == "thorough" else 0,
        critic_allow_text_fallback=(validation == "thorough"),
        critic_text_max_retries=None if validation == "thorough" else 0,
    )


def _low_risk_reason(
    *,
    method: str,
    instrument_type: str | None,
    product_ir=None,
) -> str | None:
    """Return the deterministic low-risk reason, or ``None`` when escalation is needed."""
    if method != "analytical":
        return None

    if product_ir is not None:
        if not getattr(product_ir, "supported", True):
            return None
        if getattr(product_ir, "unresolved_primitives", ()):
            return None
        if getattr(product_ir, "exercise_style", None) not in {"european", "none"}:
            return None
        if getattr(product_ir, "state_dependence", None) not in {"terminal_markov"}:
            return None
        if bool(getattr(product_ir, "schedule_dependence", False)):
            return None
        if getattr(product_ir, "model_family", None) not in {"equity_diffusion", "generic"}:
            return None
        if set(getattr(product_ir, "payoff_traits", ())) & _HIGH_RISK_TRAITS:
            return None
        if getattr(product_ir, "instrument", instrument_type) != "european_option":
            return None
        return "low_risk_supported_vanilla_analytical"

    normalized = (instrument_type or "").strip().lower()
    if normalized == "european_option":
        return "low_risk_supported_vanilla_analytical"
    return None


def _validation_contract_blocking_reason(validation_contract) -> str | None:
    """Return a deterministic blocking reason from the compiled validation contract."""
    if validation_contract is None:
        return None
    if getattr(validation_contract, "lowering_errors", ()) or ():
        return "validation_contract_lowering_errors_present"
    if getattr(validation_contract, "admissibility_failures", ()) or ():
        return "validation_contract_admissibility_failures_present"
    return None


def _validation_contract_review_reason(validation_contract) -> str | None:
    """Return the contract-driven review escalation reason, if any."""
    if validation_contract is None:
        return None
    if getattr(validation_contract, "residual_risks", ()) or ():
        return "validation_contract_residual_risks_present"

    relations = tuple(getattr(validation_contract, "comparison_relations", ()) or ())
    if any(getattr(item, "relation", None) in {"<=", ">="} for item in relations):
        return "validation_contract_directional_relations_present"
    if any(
        (getattr(item, "relation", None) or "") not in {"", "within_tolerance", "<=", ">="}
        for item in relations
    ):
        return "validation_contract_complex_relations_present"
    return None
