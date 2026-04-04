"""High-level ask() pipeline: parse → match → price.

This is the user-facing entry point for natural-language pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.agent.term_sheet import TermSheet, parse_term_sheet


@dataclass
class AskResult:
    """Result of an ask() call."""

    price: float
    term_sheet: TermSheet
    payoff_class: str
    matched_existing: bool          # True if we used an existing payoff
    details: dict | None = None     # instrument-specific details
    analytics: object | None = None  # AnalyticsResult if measures were requested


# ---------------------------------------------------------------------------
# Matcher: TermSheet → existing payoff class + spec kwargs
# ---------------------------------------------------------------------------

def _resolve_date(val) -> date | None:
    """Convert a string or date to a date object."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        return date.fromisoformat(val)
    return None


def _resolve_frequency(val: str | None) -> object | None:
    """Convert frequency string to Frequency enum."""
    if val is None:
        return None
    from trellis.core.types import Frequency
    mapping = {
        "annual": Frequency.ANNUAL,
        "semi-annual": Frequency.SEMI_ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    return mapping.get(val.lower())


def _resolve_day_count(val: str | None) -> object | None:
    """Convert day count string to DayCountConvention enum."""
    if val is None:
        return None
    from trellis.conventions.day_count import DayCountConvention
    mapping = {
        "act_360": DayCountConvention.ACT_360,
        "act/360": DayCountConvention.ACT_360,
        "act_365": DayCountConvention.ACT_365,
        "act/365": DayCountConvention.ACT_365,
        "thirty_360": DayCountConvention.THIRTY_360,
        "30/360": DayCountConvention.THIRTY_360,
    }
    return mapping.get(val.lower().replace(" ", "_"))


def match_payoff(term_sheet: TermSheet, settlement: date) -> tuple | None:
    """Match a TermSheet to an existing payoff class + spec kwargs.

    Returns ``(payoff_instance, requirements)`` or ``None`` if no match.
    """
    p = term_sheet.parameters
    itype = term_sheet.instrument_type.lower().replace(" ", "_").replace("-", "_")

    if itype == "bond":
        return _match_bond(term_sheet, p, settlement)
    elif itype == "cap":
        return _match_cap(term_sheet, p, settlement, is_floor=False)
    elif itype == "floor":
        return _match_cap(term_sheet, p, settlement, is_floor=True)
    elif itype == "swap":
        return _match_swap(term_sheet, p, settlement)
    elif itype == "swaption":
        return _match_swaption(term_sheet, p, settlement)
    else:
        return None  # unknown → trigger build


def _match_bond(ts: TermSheet, p: dict, settlement: date):
    """Map a parsed bond term sheet onto Trellis' deterministic cashflow bond payoff."""
    from trellis.instruments.bond import Bond
    from trellis.core.payoff import DeterministicCashflowPayoff

    maturity_date = _resolve_date(p.get("end_date") or p.get("maturity_date"))
    maturity = p.get("maturity")
    if maturity_date is None and maturity:
        maturity_date = date(settlement.year + int(maturity), settlement.month, settlement.day)

    bond = Bond(
        face=ts.notional,
        coupon=p.get("coupon", 0.0),
        maturity_date=maturity_date,
        maturity=int(maturity) if maturity else None,
        frequency=(_resolve_frequency(p.get("frequency")) or
                   __import__("trellis.core.types", fromlist=["Frequency"]).Frequency.SEMI_ANNUAL).value,
    )
    payoff = DeterministicCashflowPayoff(bond)
    return payoff, {"discount_curve"}


def _match_cap(ts: TermSheet, p: dict, settlement: date, is_floor: bool):
    """Map a parsed cap/floor term sheet onto the built-in Black-76 cap/floor payoff."""
    from trellis.instruments.cap import CapFloorSpec, CapPayoff, FloorPayoff
    from trellis.core.types import Frequency

    start = _resolve_date(p.get("start_date")) or settlement
    end = _resolve_date(p.get("end_date"))
    if end is None:
        maturity = p.get("maturity", 5)
        end = date(settlement.year + int(maturity), settlement.month, settlement.day)

    spec = CapFloorSpec(
        notional=ts.notional,
        strike=p.get("strike", 0.05),
        start_date=start,
        end_date=end,
        frequency=_resolve_frequency(p.get("frequency")) or Frequency.QUARTERLY,
        rate_index=p.get("rate_index"),
    )
    dc = _resolve_day_count(p.get("day_count"))
    if dc:
        spec = CapFloorSpec(
            notional=spec.notional, strike=spec.strike,
            start_date=spec.start_date, end_date=spec.end_date,
            frequency=spec.frequency, day_count=dc,
            rate_index=spec.rate_index,
        )

    cls = FloorPayoff if is_floor else CapPayoff
    return cls(spec), {"discount_curve", "forward_curve", "black_vol_surface"}


def _match_swap(ts: TermSheet, p: dict, settlement: date):
    """Map a parsed swap term sheet onto the built-in fixed-vs-floating swap payoff."""
    from trellis.instruments.swap import SwapPayoff, SwapSpec
    from trellis.core.types import Frequency

    start = _resolve_date(p.get("start_date")) or settlement
    end = _resolve_date(p.get("end_date"))
    if end is None:
        maturity = p.get("maturity", 5)
        end = date(settlement.year + int(maturity), settlement.month, settlement.day)

    spec = SwapSpec(
        notional=ts.notional,
        fixed_rate=p.get("strike") or p.get("fixed_rate", 0.05),
        start_date=start,
        end_date=end,
        is_payer=p.get("is_payer", True),
        rate_index=p.get("rate_index"),
    )
    return SwapPayoff(spec), {"discount_curve", "forward_curve"}


def _match_swaption(ts: TermSheet, p: dict, settlement: date):
    """Try to match to an agent-built swaption, or return None to trigger build."""
    try:
        import importlib
        mod = importlib.import_module("trellis.instruments._agent.swaption")
        SwaptionPayoff = mod.SwaptionPayoff
        SwaptionSpec = mod.SwaptionSpec
    except (ImportError, AttributeError):
        return None  # no swaption pricer yet → trigger build

    expiry = _resolve_date(p.get("expiry_date") or p.get("expiry"))
    swap_start = _resolve_date(p.get("swap_start") or p.get("start_date")) or expiry
    swap_end = _resolve_date(p.get("swap_end") or p.get("end_date"))
    if swap_end is None:
        maturity = p.get("swap_maturity") or p.get("maturity", 5)
        swap_end = date(swap_start.year + int(maturity), swap_start.month, swap_start.day)

    spec = SwaptionSpec(
        notional=ts.notional,
        strike=p.get("strike", 0.05),
        expiry_date=expiry,
        swap_start=swap_start,
        swap_end=swap_end,
        is_payer=p.get("is_payer", True),
        rate_index=p.get("rate_index"),
    )
    return SwaptionPayoff(spec), {"discount_curve", "forward_curve", "black_vol_surface"}


# ---------------------------------------------------------------------------
# Main ask() function
# ---------------------------------------------------------------------------

def ask_session(
    description: str,
    session,
    measures: list | None = None,
    model: str | None = None,
) -> AskResult:
    """Parse a natural-language description and price it.

    Parameters
    ----------
    description : str
        e.g. "Price a 5Y cap at 4% on $10M SOFR"
    session : Session
        The market snapshot to price against.
    model : str or None
        LLM model override.

    Returns
    -------
    AskResult
    """
    # Step 1: Parse
    term_sheet = parse_term_sheet(description, session.settlement, model=model)
    compiled_request = None
    result = None

    try:
        from trellis.agent.platform_requests import (
            compile_platform_request,
            make_term_sheet_request,
        )
        from trellis.agent.platform_traces import append_platform_trace_event
        from trellis.platform.executor import execute_compiled_request
        from trellis.platform.results import (
            execution_result_exception,
            execution_result_trace_details,
            project_execution_result_value,
        )

        compiled_request = compile_platform_request(
            make_term_sheet_request(
                description=description,
                term_sheet=term_sheet,
                session=session,
                measures=measures,
                model=model,
            )
        )
        append_platform_trace_event(
            compiled_request,
            "request_compiled",
            status="ok",
            details={
                "action": compiled_request.execution_plan.action,
                "route_method": compiled_request.execution_plan.route_method,
                "requires_build": compiled_request.execution_plan.requires_build,
            },
        )
        result = execute_compiled_request(
            compiled_request,
            session.to_execution_context(),
        )
        payload = project_execution_result_value(
            result,
            key="",
            default_message="Governed ask execution failed.",
        )
        details = dict(execution_result_trace_details(result))
        details["payoff_class"] = payload.get("payoff_class", "")
        outcome = (
            "ask_priced_existing"
            if payload.get("matched_existing", False)
            else "ask_built_and_priced"
        )
        from trellis.agent.platform_traces import record_platform_trace

        record_platform_trace(
            compiled_request,
            success=True,
            outcome=outcome,
            details=details,
        )
        return AskResult(
            price=payload["price"],
            term_sheet=payload.get("term_sheet", term_sheet),
            payoff_class=payload["payoff_class"],
            matched_existing=payload.get("matched_existing", False),
            details=payload.get("details"),
            analytics=payload.get("analytics"),
        )
    except Exception as exc:
        if compiled_request is not None:
            _handle_ask_failure(compiled_request, result, exc)
        raise


def _handle_ask_failure(compiled_request, result, exc: Exception) -> None:
    """Record one governed ask failure and re-raise the appropriate exception."""
    from trellis.agent.platform_traces import record_platform_trace
    from trellis.platform.results import (
        execution_result_exception,
        execution_result_trace_details,
    )

    if result is not None and result.status == "blocked":
        chunks, blocker_codes = _blocked_request_sections(compiled_request)
        details = execution_result_trace_details(result)
        details["blocker_codes"] = blocker_codes
        try:
            record_platform_trace(
                compiled_request,
                success=False,
                outcome="request_blocked",
                details=details,
            )
        except Exception:
            pass
        if chunks:
            raise RuntimeError(
                "Request is blocked by missing foundational machinery:\n\n"
                + "\n\n".join(chunks)
            ) from exc
        raise execution_result_exception(
            result,
            default_message="Governed ask execution was blocked.",
        ) from exc

    details = {
        **({} if result is None else execution_result_trace_details(result)),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    try:
        record_platform_trace(
            compiled_request,
            success=False,
            outcome="ask_failed",
            details=details,
        )
    except Exception:
        pass


def _blocked_request_sections(compiled_request) -> tuple[list[str], list[str]]:
    """Render the structured blocker sections for one blocked ask request."""
    chunks: list[str] = []
    blocker_codes: list[str] = []
    if compiled_request.blocker_report is not None:
        from trellis.agent.blocker_planning import render_blocker_report

        blocker_codes = [
            blocker.id for blocker in compiled_request.blocker_report.blockers
        ]
        chunks.append(render_blocker_report(compiled_request.blocker_report))
    if compiled_request.new_primitive_workflow is not None:
        from trellis.agent.new_primitive_workflow import render_new_primitive_workflow

        chunks.append(
            render_new_primitive_workflow(compiled_request.new_primitive_workflow)
        )
    return chunks, blocker_codes


def _infer_requirements(ts: TermSheet) -> set[str]:
    """Infer MarketState requirements from instrument type."""
    itype = ts.instrument_type.lower()
    if itype in ("bond", "zero_coupon_bond"):
        return {"discount_curve"}
    elif itype in ("swap",):
        return {"discount_curve", "forward_curve"}
    elif itype in ("cap", "floor", "swaption", "callable_bond"):
        return {"discount_curve", "forward_curve", "black_vol_surface"}
    else:
        # Conservative: assume everything
        return {"discount_curve", "forward_curve", "black_vol_surface"}
