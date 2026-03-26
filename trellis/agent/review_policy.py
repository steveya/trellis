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


def determine_review_policy(
    *,
    validation: str,
    method: str,
    instrument_type: str | None = None,
    product_ir=None,
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

    if validation not in {"standard", "thorough"}:
        return ReviewPolicy(
            risk_level=risk_level,
            run_critic=False,
            run_model_validator_llm=False,
            critic_reason="validation_mode_skipped",
            model_validator_reason="validation_mode_skipped",
        )

    if low_risk_reason:
        return ReviewPolicy(
            risk_level="low",
            run_critic=False,
            run_model_validator_llm=False,
            critic_reason=low_risk_reason,
            model_validator_reason=low_risk_reason,
        )

    return ReviewPolicy(
        risk_level="high",
        run_critic=True,
        run_model_validator_llm=(validation == "thorough"),
        critic_reason="high_risk_route_requires_llm_review",
        model_validator_reason=(
            "high_risk_route_requires_llm_review"
            if validation == "thorough"
            else "validation_mode_skipped"
        ),
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
