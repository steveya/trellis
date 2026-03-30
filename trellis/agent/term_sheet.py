"""Parse a natural-language instrument description into a structured TermSheet.

The TermSheet has a handful of universal fields (instrument type, notional,
currency) plus a free-form parameters dict for everything instrument-specific,
since the fields of a cap differ completely from those of an autocallable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class TermSheet:
    """Parsed instrument description with common fields and a free-form parameters dict."""

    instrument_type: str              # "cap", "swaption", "bond", "swap", etc.
    notional: float = 100.0
    currency: str = "USD"
    parameters: dict = field(default_factory=dict)
    raw_description: str = ""


def parse_term_sheet(
    description: str,
    settlement: date | None = None,
    model: str | None = None,
) -> TermSheet:
    """Parse a natural-language instrument description into a TermSheet.

    Uses the LLM to extract structured fields.
    """
    from trellis.agent.config import llm_generate_json, get_default_model

    settlement = settlement or date.today()
    model = model or get_default_model()

    prompt = _build_parse_prompt(description, settlement)

    try:
        data = llm_generate_json(prompt, model=model)
    except Exception:
        # Fallback: try text mode
        from trellis.agent.config import llm_generate
        import json
        text = llm_generate(prompt, model=model)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            raise ValueError(f"Could not parse term sheet from: {description}")

    return TermSheet(
        instrument_type=data.get("instrument_type", "unknown"),
        notional=float(data.get("notional", 100.0)),
        currency=data.get("currency", "USD"),
        parameters=data.get("parameters", {}),
        raw_description=description,
    )


def _build_parse_prompt(description: str, settlement: date) -> str:
    """Build the extraction prompt used by the term-sheet parser LLM call."""
    return f"""You are parsing a financial instrument description into structured data.

## Description
"{description}"

## Settlement date
{settlement.isoformat()} (use this to resolve relative dates like "5Y" → concrete dates)

## Output format
Return a JSON object:
{{
    "instrument_type": "cap" | "floor" | "swaption" | "swap" | "bond" | "callable_bond" | "cds" | "fx_forward" | "basket_option" | other,
    "notional": 1000000,
    "currency": "USD",
    "parameters": {{
        // All instrument-specific fields as key-value pairs.
        // Dates must be ISO format strings: "2029-11-15"
        // Rates as decimals: 0.04 not 4%
        // Use these standard field names where applicable:
        //   strike, coupon, start_date, end_date, expiry_date,
        //   swap_start, swap_end, frequency ("quarterly", "semi-annual", "annual", "monthly"),
        //   rate_index ("SOFR_3M", "EURIBOR_6M", "SONIA", etc.),
        //   day_count ("ACT_360", "ACT_365", "THIRTY_360"),
        //   is_payer (true/false),
        //   call_schedule (list of {{"date": "YYYY-MM-DD", "price": 100.0}}),
        //   maturity (integer years, if applicable),
        //   constituents (basket underlier names),
        //   observation_dates (ordered list of ISO dates),
        //   selection_operator ("best_of_remaining"),
        //   selection_scope ("remaining_constituents"),
        //   selection_count (integer, typically 1),
        //   lock_rule ("remove_selected"),
        //   aggregation_rule ("average_locked_returns"),
        //   correlation_matrix (nested list of pairwise correlations),
        //   correlation_matrix_key (lookup key for model_parameters)
    }}
}}

## Rules
- Convert percentage rates to decimals: "4%" → 0.04
- Resolve relative dates: "5Y" → "{(date(settlement.year + 5, settlement.month, settlement.day)).isoformat()}"
- Default notional is 100 (for bonds) or 1000000 (for derivatives)
- Default currency is USD
- If a rate index is mentioned (SOFR, LIBOR, EURIBOR, SONIA), include it
- For swaptions: extract both expiry and underlying swap dates
- For callable bonds: extract the call schedule
- For ranked-observation basket payoffs: extract the ordered observation schedule, basket constituents, and correlation inputs; keep the selection semantics explicit and family-name-free

Return ONLY the JSON object, no other text."""
