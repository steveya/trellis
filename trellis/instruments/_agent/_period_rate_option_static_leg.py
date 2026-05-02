"""Shared static-leg execution helpers for cap/floor compatibility wrappers."""

from __future__ import annotations

from trellis.agent.static_leg_contract import (
    NotionalSchedule,
    NotionalStep,
    OvernightRateIndex,
    PeriodRateOptionPeriod,
    PeriodRateOptionStripLeg,
    SettlementRule,
    SignedLeg,
    StaticLegContractIR,
    TermRateIndex,
)
from trellis.core.payoff import ExecutionBackedPayoff
from trellis.execution import compile_static_leg_execution_ir


def build_period_rate_option_execution_payoff(
    spec,
    timeline,
    *,
    option_side: str,
    label: str,
    method: str = "analytical",
) -> ExecutionBackedPayoff:
    instrument_class = "cap" if option_side == "call" else "floor"
    currency = _currency_from_rate_index(getattr(spec, "rate_index", None))
    execution_terms = {
        key: value
        for key in (
            "calendar_name",
            "business_day_adjustment",
            "model",
            "shift",
            "sabr",
        )
        if (value := getattr(spec, key, None)) is not None
    }
    execution_terms["reconstruct_from_spec_schedule"] = True
    contract = StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="receive",
                leg=PeriodRateOptionStripLeg(
                    currency=currency,
                    notional_schedule=NotionalSchedule(
                        (
                            NotionalStep(
                                start_date=spec.start_date,
                                end_date=spec.end_date,
                                amount=spec.notional,
                            ),
                        )
                    ),
                    option_periods=tuple(
                        PeriodRateOptionPeriod(
                            accrual_start=period.start_date,
                            accrual_end=period.end_date,
                            fixing_date=period.start_date,
                            payment_date=period.payment_date,
                        )
                        for period in timeline
                    ),
                    rate_index=_parse_rate_index(getattr(spec, "rate_index", None)),
                    strike=spec.strike,
                    option_side=option_side,
                    day_count=_enum_value(spec.day_count, fallback="ACT/360"),
                    payment_frequency=_enum_name(spec.frequency, fallback="quarterly"),
                    label=label,
                    metadata={
                        "family": "period_rate_option_strip",
                        "instrument_class": instrument_class,
                        "semantic_family": "period_rate_option_strip",
                    },
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency=currency),
        metadata={
            "family": "period_rate_option_strip",
            "instrument_class": instrument_class,
            "semantic_family": "period_rate_option_strip",
        },
    )
    return ExecutionBackedPayoff(
        compile_static_leg_execution_ir(contract, requested_method=method),
        method=method,
        execution_terms=execution_terms,
    )


def _parse_rate_index(rate_index: str | None):
    text = str(rate_index or "").strip().upper()
    if not text:
        return OvernightRateIndex("SOFR")
    pieces = text.split("-")
    if len(pieces) >= 2 and _looks_like_tenor(pieces[-1]):
        return TermRateIndex("-".join(pieces[:-1]), pieces[-1])
    return OvernightRateIndex(text)


def _currency_from_rate_index(rate_index: str | None) -> str:
    text = str(rate_index or "").strip().upper()
    if not text:
        return "USD"
    head = text.split("-", 1)[0]
    if len(head) == 3 and head.isalpha():
        return head
    return "USD"


def _looks_like_tenor(value: str) -> bool:
    token = str(value or "").strip().upper()
    return len(token) >= 2 and token[:-1].isdigit() and token[-1] in {"D", "W", "M", "Y"}


def _enum_name(value: object, *, fallback: str) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name).strip().lower()
    text = str(value or "").strip()
    return text or fallback


def _enum_value(value: object, *, fallback: str) -> str:
    raw = getattr(value, "value", None)
    if raw:
        return str(raw)
    text = str(value or "").strip()
    return text or fallback
