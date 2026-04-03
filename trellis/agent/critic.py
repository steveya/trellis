"""Critic agent: reads generated code and selects structured review findings.

The critic sees ONLY the generated code (not the builder's reasoning).
Its job is to select from a bounded menu of deterministic checks that the
arbiter can execute cheaply. Legacy ``test_code`` payloads are available only
through an explicit compatibility flag and are disabled by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from trellis.agent.config import llm_generate, llm_generate_json


@dataclass(frozen=True)
class CriticConcern:
    """A potential issue identified by the critic."""
    check_id: str
    description: str
    severity: str = "error"  # "error" or "warning"
    evidence: str = ""
    remediation: str = ""
    status: str = "suspect"
    test_code: str = ""


@dataclass(frozen=True)
class CriticCheck:
    """A deterministic arbiter check that the critic may select."""

    check_id: str
    title: str
    when_to_use: str
    deterministic_contract: str


_CHECK_LIBRARY: dict[str, CriticCheck] = {
    "price_non_negative": CriticCheck(
        check_id="price_non_negative",
        title="Price should remain non-negative",
        when_to_use=(
            "the route prices a long-only vanilla or option-like payoff that should not"
            " produce a negative PV under the default review market state"
        ),
        deterministic_contract="Fail when price_payoff(payoff, ms) < -1e-6.",
    ),
    "volatility_input_usage": CriticCheck(
        check_id="volatility_input_usage",
        title="Volatility should move the price materially",
        when_to_use=(
            "the instrument or method requires volatility but the code looks"
            " volatility-insensitive"
        ),
        deterministic_contract=(
            "Fail when changing flat vol from 5% to 40% moves price by less than 0.1%."
        ),
    ),
    "rate_sensitivity_present": CriticCheck(
        check_id="rate_sensitivity_present",
        title="Discount-rate changes should move the price",
        when_to_use=(
            "the route discounts future cashflows and the implementation looks"
            " insensitive to the discount curve"
        ),
        deterministic_contract=(
            "Fail when changing flat discount rate from 3% to 7% leaves price nearly unchanged."
        ),
    ),
    "callable_bound_vs_straight_bond": CriticCheck(
        check_id="callable_bound_vs_straight_bond",
        title="Callable bond should not exceed the equivalent straight bond",
        when_to_use=(
            "the instrument is callable and the exercise logic may overvalue the issuer option"
        ),
        deterministic_contract=(
            "Fail when price_payoff(payoff, ms) exceeds straight_bond_pv by more than 1e-6."
        ),
    ),
    "puttable_bound_vs_straight_bond": CriticCheck(
        check_id="puttable_bound_vs_straight_bond",
        title="Puttable bond should not be below the equivalent straight bond",
        when_to_use=(
            "the instrument is puttable and the holder option may be ignored or undervalued"
        ),
        deterministic_contract=(
            "Fail when price_payoff(payoff, ms) is below straight_bond_pv by more than 1e-6."
        ),
    ),
}

_OPTION_LIKE_INSTRUMENTS = {
    "european_option",
    "barrier_option",
    "digital_option",
    "fx_option",
    "swaption",
    "cap",
    "floor",
}

_VOL_SENSITIVE_INSTRUMENTS = _OPTION_LIKE_INSTRUMENTS | {
    "callable_bond",
    "puttable_bond",
}

_RATE_SENSITIVE_INSTRUMENTS = {
    "bond",
    "callable_bond",
    "puttable_bond",
    "swaption",
    "cap",
    "floor",
}


def available_critic_checks(
    *,
    instrument_type: str | None = None,
    method: str | None = None,
    product_ir=None,
    validation_contract=None,
) -> list[CriticCheck]:
    """Return the bounded deterministic check menu for one route."""

    instrument = (
        str(getattr(validation_contract, "instrument_type", "") or "").strip().lower()
        or (instrument_type or "").strip().lower()
        or str(getattr(product_ir, "instrument", "") or "").strip().lower()
    )
    contract_checks = _checks_from_validation_contract(
        instrument=instrument,
        validation_contract=validation_contract,
    )
    if contract_checks:
        return contract_checks

    checks: list[CriticCheck] = []

    if instrument in _OPTION_LIKE_INSTRUMENTS:
        checks.append(_CHECK_LIBRARY["price_non_negative"])
    if instrument in _VOL_SENSITIVE_INSTRUMENTS:
        checks.append(_CHECK_LIBRARY["volatility_input_usage"])
    if instrument in _RATE_SENSITIVE_INSTRUMENTS:
        checks.append(_CHECK_LIBRARY["rate_sensitivity_present"])
    if instrument == "callable_bond":
        checks.append(_CHECK_LIBRARY["callable_bound_vs_straight_bond"])
    if instrument == "puttable_bond":
        checks.append(_CHECK_LIBRARY["puttable_bound_vs_straight_bond"])

    seen: set[str] = set()
    deduped: list[CriticCheck] = []
    for check in checks:
        if check.check_id in seen:
            continue
        seen.add(check.check_id)
        deduped.append(check)
    return deduped


def _checks_from_validation_contract(
    *,
    instrument: str,
    validation_contract=None,
) -> list[CriticCheck]:
    """Map compiled deterministic validation checks onto critic-visible check families."""
    if validation_contract is None:
        return []

    deterministic_checks = tuple(getattr(validation_contract, "deterministic_checks", ()) or ())
    deterministic_ids = {
        str(getattr(item, "check_id", "") or "").strip()
        for item in deterministic_checks
        if str(getattr(item, "check_id", "") or "").strip()
    }
    bound_relation = None
    for item in deterministic_checks:
        if str(getattr(item, "check_id", "") or "").strip() == "check_bounded_by_reference":
            relation = str(getattr(item, "relation", "") or "").strip()
            if relation:
                bound_relation = relation
                break

    checks: list[CriticCheck] = []
    if "check_non_negativity" in deterministic_ids:
        checks.append(_CHECK_LIBRARY["price_non_negative"])
    if {"check_vol_sensitivity", "check_vol_monotonicity"} & deterministic_ids:
        checks.append(_CHECK_LIBRARY["volatility_input_usage"])
    if "check_rate_monotonicity" in deterministic_ids:
        checks.append(_CHECK_LIBRARY["rate_sensitivity_present"])
    if "check_bounded_by_reference" in deterministic_ids:
        if instrument == "puttable_bond" or bound_relation == ">=":
            checks.append(_CHECK_LIBRARY["puttable_bound_vs_straight_bond"])
        elif instrument == "callable_bond" or bound_relation == "<=":
            checks.append(_CHECK_LIBRARY["callable_bound_vs_straight_bond"])

    seen: set[str] = set()
    deduped: list[CriticCheck] = []
    for check in checks:
        if check.check_id in seen:
            continue
        seen.add(check.check_id)
        deduped.append(check)
    return deduped


def _format_available_checks(checks: Sequence[CriticCheck]) -> str:
    """Render the allowed deterministic check menu for the critic prompt."""

    if not checks:
        return "- No route-specific deterministic critic checks are available for this route.\n"
    lines = []
    for check in checks:
        lines.append(
            f"- `{check.check_id}`: {check.title}. "
            f"Use when {check.when_to_use}. "
            f"Deterministic contract: {check.deterministic_contract}"
        )
    return "\n".join(lines) + "\n"


CRITIC_PROMPT_TEMPLATE = """\
You are a quantitative model validator reviewing agent-generated pricing code.
Your job is to find deterministic review concerns, not to praise. Be adversarial.

## Code to review
```python
{code}
```

## Instrument description
{description}
{knowledge_section}
{route_contract_section}

## Available deterministic checks
You may ONLY select from this bounded menu:
{available_checks}

## Review policy
- Do not write Python code.
- Do not invent new checks, formulas, or test procedures.
- Only emit a finding when the code strongly suggests the selected check is likely to fail.
- If none of the available checks are justified, return [].

## Output
Return a JSON array of concerns. Each concern:
{{
    "check_id": "one of the available check ids",
    "description": "short concern summary",
    "severity": "error" or "warning",
    "evidence": "specific code reference or reasoning",
    "remediation": "what to change",
    "status": "suspect"
}}

Focus on pricing errors that the deterministic checks can confirm.
Return at most 3 concerns, ordered by severity.
Return ONLY the JSON array."""


def critique(
    code: str,
    description: str,
    knowledge_context: str = "",
    model: str | None = None,
    *,
    generation_plan=None,
    available_checks: Sequence[CriticCheck] | None = None,
    json_max_retries: int | None = None,
    allow_text_fallback: bool = True,
    text_max_retries: int | None = None,
    allow_legacy_test_code: bool = False,
) -> list[CriticConcern]:
    """Run the critic agent on generated code.

    Parameters
    ----------
    code : str
        The full Python module source.
    description : str
        What the instrument is (e.g. "Callable bond with call schedule").
    model : str or None
        LLM model to use.

    Returns
    -------
    list[CriticConcern]
    """
    knowledge_section = ""
    if knowledge_context.strip():
        knowledge_section = f"\n## Shared Knowledge\n{knowledge_context}\n"
    route_contract_section = ""
    if generation_plan is not None:
        from trellis.agent.codegen_guardrails import render_review_contract_card

        route_contract_section = "\n" + render_review_contract_card(generation_plan) + "\n"

    prompt = CRITIC_PROMPT_TEMPLATE.format(
        code=code,
        description=description,
        knowledge_section=knowledge_section,
        route_contract_section=route_contract_section,
        available_checks=_format_available_checks(available_checks or ()),
    )

    try:
        data = llm_generate_json(prompt, model=model, max_retries=json_max_retries)
    except Exception:
        if not allow_text_fallback:
            raise
        text = llm_generate(prompt, model=model, max_retries=text_max_retries)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            return []
        data = text[start:end]

    if isinstance(data, str):
        data = json.loads(data)

    concerns = []
    allowed_check_ids = {
        check.check_id
        for check in (available_checks or ())
    }
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            check_id = str(item.get("check_id") or ("legacy_test_code" if item.get("test_code") else "")).strip()
            if check_id == "legacy_test_code" and not allow_legacy_test_code:
                continue
            if allowed_check_ids and check_id not in allowed_check_ids and check_id != "legacy_test_code":
                continue
            concerns.append(
                CriticConcern(
                    check_id=check_id,
                    description=str(item.get("description", "") or "").strip(),
                    severity=str(item.get("severity", "warning") or "warning").strip(),
                    evidence=str(item.get("evidence", "") or "").strip(),
                    remediation=str(item.get("remediation", "") or "").strip(),
                    status=str(item.get("status", "suspect") or "suspect").strip(),
                    test_code=str(item.get("test_code", "") or "").strip(),
                )
            )
    return concerns
