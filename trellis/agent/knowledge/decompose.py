"""Product → feature decomposition.

For known instruments, returns a static decomposition from
canonical/decompositions.yaml.  For novel/composite products,
uses LLM to decompose into known features from the taxonomy.
"""

from __future__ import annotations

from dataclasses import replace
from calendar import monthrange
from datetime import date, timedelta
import re
from typing import Any
from typing import TYPE_CHECKING

from trellis.agent.contract_ir import (
    Annuity,
    ArithmeticMean,
    CompositeUnderlying,
    Constant,
    ContractIR,
    ContinuousInterval,
    CurveQuote,
    EquitySpot,
    Exercise,
    FiniteSchedule,
    ForwardRate,
    ForwardRateInterval,
    Gt,
    Indicator,
    LinearBasket,
    Lt,
    Max,
    Mul,
    Observation,
    ParRateTenor,
    QuoteCurve,
    QuoteSurface,
    Scaled,
    Singleton,
    Spot,
    Strike,
    Sub,
    SurfaceQuote,
    SwapRate,
    Underlying,
    VarianceObservable,
    VolDeltaPoint,
    VolPoint,
    ZeroRateTenor,
    canonicalize,
)
from trellis.agent.dynamic_contract_ir import (
    ActionSpec,
    AutomaticTerminationEvent,
    ControlProgram,
    CouponEvent,
    DecisionEvent,
    DynamicContractIR,
    EventProgram,
    EventTimeBucket,
    ObservationEvent,
    PaymentEvent,
    StateFieldSpec,
    StateSchema,
    StateUpdateSpec,
    StateResetEvent,
    TerminationRule,
)
from trellis.agent.insurance_overlay_contract import (
    InsuranceOverlayContractIR,
    OverlayCompositionRule,
    OverlayFeeEvent,
    OverlayParameterSet,
    OverlayParameterSpec,
    OverlayTransitionEvent,
    PolicyStateSchema,
)
from trellis.agent.knowledge.methods import normalize_method
from trellis.agent.knowledge.schema import ProductDecomposition, ProductIR, RetrievalSpec
from trellis.agent.static_leg_contract import (
    CouponLeg,
    CouponPeriod,
    FixedCouponFormula,
    FloatingCouponFormula,
    KnownCashflow,
    KnownCashflowLeg,
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
from trellis.agent.semantic_tokens import (
    EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY,
)
from trellis.core.capabilities import normalize_market_data_requirements
from trellis.core.date_utils import generate_schedule
from trellis.core.types import Frequency

if TYPE_CHECKING:
    from trellis.agent.knowledge.store import KnowledgeStore


_DECOMPOSITION_CACHE: dict[tuple[Any, ...], ProductDecomposition] = {}
_DECOMPOSITION_CACHE_HITS = 0
_DECOMPOSITION_CACHE_MISSES = 0


def decompose(
    description: str,
    instrument_type: str | None = None,
    model: str | None = None,
    store: KnowledgeStore | None = None,
) -> ProductDecomposition:
    """Decompose a product into features.

    1. Normalise instrument_type and check static decompositions.
    2. Try fuzzy matching against known decomposition keys.
    3. Fall back to LLM decomposition using the feature taxonomy.

    Parameters
    ----------
    description
        Natural-language product description (e.g., "callable range accrual").
    instrument_type
        Optional explicit type key (e.g., "callable_bond").
    model
        LLM model to use for fallback decomposition.
    store
        KnowledgeStore instance (default: global singleton).
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    cache_key = _decomposition_cache_key(
        description=description,
        instrument_type=instrument_type,
        model=model,
        store=store,
    )
    cached = _DECOMPOSITION_CACHE.get(cache_key)
    if cached is not None:
        global _DECOMPOSITION_CACHE_HITS
        _DECOMPOSITION_CACHE_HITS += 1
        return cached
    global _DECOMPOSITION_CACHE_MISSES
    _DECOMPOSITION_CACHE_MISSES += 1

    # Step 1: exact match on normalised key
    key = _normalise(instrument_type or description)
    matched = _match_static_decomposition(key, store)
    if matched is not None:
        _DECOMPOSITION_CACHE[cache_key] = matched
        return matched

    # Step 4: LLM decomposition
    result = _decompose_via_llm(description, key, store, model)
    _DECOMPOSITION_CACHE[cache_key] = result
    return result


def decomposition_cache_stats() -> dict[str, int]:
    """Return lightweight runtime decomposition-cache statistics."""
    return {
        "hits": _DECOMPOSITION_CACHE_HITS,
        "misses": _DECOMPOSITION_CACHE_MISSES,
        "size": len(_DECOMPOSITION_CACHE),
    }


def clear_decomposition_cache() -> None:
    """Clear the warm/runtime decomposition cache."""
    global _DECOMPOSITION_CACHE_HITS, _DECOMPOSITION_CACHE_MISSES
    _DECOMPOSITION_CACHE.clear()
    _DECOMPOSITION_CACHE_HITS = 0
    _DECOMPOSITION_CACHE_MISSES = 0


def _decomposition_cache_key(
    *,
    description: str,
    instrument_type: str | None,
    model: str | None,
    store: KnowledgeStore,
) -> tuple[Any, ...]:
    """Build a stable cache key for runtime decomposition reuse."""
    return (
        id(store),
        description.strip(),
        _normalise(instrument_type) if instrument_type else None,
        model,
    )


def decompose_to_ir(
    description: str,
    instrument_type: str | None = None,
    *,
    store: KnowledgeStore | None = None,
) -> ProductIR:
    """Decompose a product description into a structured ``ProductIR`` without calling an LLM.

    For known instruments (e.g. "callable bond"), returns the canonical
    static decomposition from YAML.  For novel or composite products that
    have no static entry, falls back to keyword-based trait extraction
    that avoids guessing -- it only assigns traits it can identify with
    certainty from the text, leaving unknowns as unresolved primitives.
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    inferred_instrument = _infer_instrument(description, instrument_type)
    matched = None
    if inferred_instrument and not _looks_composite(description):
        matched = store._decompositions.get(inferred_instrument)
    if matched is None:
        matched = _match_static_decomposition(_normalise(description), store)
        if matched is not None and _looks_composite(description):
            matched = None

    if matched is not None:
        instrument = inferred_instrument or matched.instrument
        if instrument != matched.instrument and instrument in store._decompositions:
            matched = store._decompositions[instrument]
        return _augment_ir_with_contract_ir_support(
            _product_ir_from_decomposition(
                instrument=instrument,
                decomposition=matched,
                description=description,
                store=store,
            ),
            description,
        )

    return _augment_ir_with_contract_ir_support(
        _infer_composite_ir(description, inferred_instrument, store),
        description,
    )


def decompose_to_contract_ir(
    description: str,
    instrument_type: str | None = None,
    *,
    product_ir: ProductIR | None = None,
    store: KnowledgeStore | None = None,
) -> ContractIR | None:
    """Build a bounded Contract IR for the four Phase 2 payoff families.

    This parser is deliberately fixture-driven and route-free. It handles:

    - European terminal vanilla/basket/swaption ramps
    - variance-settled contracts
    - cash-or-nothing / asset-or-nothing digitals
    - arithmetic Asians with bounded monthly/weekly schedule phrases

    Everything outside that surface returns ``None``.
    """
    product_ir = product_ir or decompose_to_ir(
        description,
        instrument_type=instrument_type,
        store=store,
    )
    instrument = _normalise(instrument_type or getattr(product_ir, "instrument", ""))
    lower = str(description or "").lower()

    if instrument in {
        "american_option",
        "american_put",
        "bermudan_swaption",
        "barrier_option",
        "lookback_option",
        "chooser_option",
        "callable_bond",
        "puttable_bond",
        "cds",
        "credit_default_swap",
        "cap",
        "floor",
        "range_accrual",
        "compound_option",
        "cliquet_option",
    }:
        return None
    if any(marker in lower for marker in ("american ", "bermudan ", "barrier ", "lookback", "chooser", "callable bond", " cds ", " caplet", " floorlet")):
        return None

    builders = (
        _build_curve_quote_contract_ir,
        _build_surface_quote_contract_ir,
        _build_swaption_contract_ir,
        _build_basket_contract_ir,
        _build_variance_contract_ir,
        _build_digital_contract_ir,
        _build_asian_contract_ir,
        _build_vanilla_contract_ir,
    )
    for builder in builders:
        contract_ir = builder(
            description,
            instrument=instrument,
            product_ir=product_ir,
        )
        if contract_ir is not None:
            return contract_ir
    return None


def decompose_to_static_leg_contract_ir(
    description: str,
    instrument_type: str | None = None,
    *,
    product_ir: ProductIR | None = None,
    store: KnowledgeStore | None = None,
) -> StaticLegContractIR | None:
    """Build a bounded static leg contract IR for the first post-Phase-4 leg slice."""

    product_ir = product_ir or decompose_to_ir(
        description,
        instrument_type=instrument_type,
        store=store,
    )
    instrument = _normalise(instrument_type or getattr(product_ir, "instrument", ""))
    lower = str(description or "").lower()

    if any(
        marker in lower
        for marker in (
            "caplet",
            "floorlet",
            "callable ",
            "puttable ",
            "autocall",
            "phoenix",
            "snowball",
            "tarn",
            "tarf",
            "range accrual",
            "swing option",
            "gmwb",
            "gmxb",
            "curve-spread payoff",
            "curve spread payoff",
            "vol-skew payoff",
            "vol skew payoff",
        )
    ):
        return None

    builders = (
        _build_static_period_rate_option_strip_contract_ir,
        _build_static_fixed_float_swap_contract_ir,
        _build_static_basis_swap_contract_ir,
        _build_static_fixed_coupon_bond_contract_ir,
    )
    for builder in builders:
        contract = builder(
            description,
            instrument=instrument,
            product_ir=product_ir,
        )
        if contract is not None:
            return contract
    return None


def decompose_to_dynamic_contract_ir(
    description: str,
    instrument_type: str | None = None,
    *,
    product_ir: ProductIR | None = None,
    store: KnowledgeStore | None = None,
) -> DynamicContractIR | None:
    """Build a bounded dynamic wrapper IR for admitted post-Phase-4 lane fixtures."""

    product_ir = product_ir or decompose_to_ir(
        description,
        instrument_type=instrument_type,
        store=store,
    )
    instrument = _normalise(instrument_type or getattr(product_ir, "instrument", ""))

    builders = (
        _build_dynamic_autocallable_contract_ir,
        _build_dynamic_tarn_contract_ir,
        _build_dynamic_callable_bond_contract_ir,
        _build_dynamic_swing_option_contract_ir,
        _build_dynamic_gmwb_contract_ir,
    )
    for builder in builders:
        contract = builder(
            description,
            instrument=instrument,
            product_ir=product_ir,
        )
        if contract is not None:
            return contract
    return None


def decompose_to_insurance_overlay_contract_ir(
    description: str,
    instrument_type: str | None = None,
    *,
    product_ir: ProductIR | None = None,
    store: KnowledgeStore | None = None,
) -> InsuranceOverlayContractIR | None:
    """Build a bounded insurance-overlay wrapper above the financial-control core."""

    product_ir = product_ir or decompose_to_ir(
        description,
        instrument_type=instrument_type,
        store=store,
    )
    instrument = _normalise(instrument_type or getattr(product_ir, "instrument", ""))

    builders = (_build_gmwb_insurance_overlay_contract_ir,)
    for builder in builders:
        contract = builder(
            description,
            instrument=instrument,
            product_ir=product_ir,
        )
        if contract is not None:
            return contract
    return None


def _augment_ir_with_contract_ir_support(
    ir: ProductIR,
    description: str,
) -> ProductIR:
    """Upgrade sparse free-form basket IRs from the route-free ContractIR view.

    The incumbent route authority for analytical basket helpers still relies on
    ``ProductIR`` carrying the structural facts that the Phase 2 ContractIR
    parser can already recover: European exercise, equity-diffusion underliers,
    and the exact two-asset terminal-basket shape. Reusing that structure keeps
    the build path route-free while allowing the legacy authority packet to bind
    correctly for the same contracts the structural compiler already admits.
    """
    if getattr(ir, "instrument", "") != "basket_option":
        return ir

    contract_ir = decompose_to_contract_ir(
        description,
        instrument_type="basket_option",
        product_ir=ir,
    )
    if not _is_two_asset_terminal_equity_basket_contract_ir(contract_ir):
        return ir

    payoff_traits = list(getattr(ir, "payoff_traits", ()) or ())
    for trait in ("multi_asset", "two_asset_terminal_basket"):
        if trait not in payoff_traits:
            payoff_traits.append(trait)

    enriched = replace(
        ir,
        payoff_traits=tuple(payoff_traits),
        exercise_style="european",
        state_dependence="terminal_markov",
        schedule_dependence=False,
        model_family="equity_diffusion",
        required_market_data=frozenset(
            normalize_market_data_requirements(
                {
                    *(getattr(ir, "required_market_data", ()) or ()),
                    "discount_curve",
                    "black_vol_surface",
                    "spot",
                    "model_parameters",
                }
            )
        ),
    )
    return _augment_ir_with_promoted_route_support(enriched)


def _build_static_fixed_float_swap_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> StaticLegContractIR | None:
    lower = description.lower()
    if instrument not in {"", "swap", "interest_rate_swap"} and getattr(product_ir, "instrument", "") not in {"swap"}:
        return None
    if "basis swap" in lower:
        return None
    if "irs" not in lower and "interest rate swap" not in lower:
        return None

    direction = _extract_fixed_leg_direction(description)
    start = _extract_named_date(description, labels=("effective", "start"))
    end = _extract_named_date(description, labels=("maturity", "end"))
    currency = _extract_currency_code(description) or "USD"
    notional = _extract_numeric_after(description, labels=("notional",))
    fixed_rate = _extract_numeric_after(description, labels=("fixed rate",))
    fixed_frequency = _extract_frequency_after_label(description, label="fixed") or "semiannual"
    float_frequency = _extract_frequency_after_label(description, label="float") or "quarterly"
    fixed_day_count = _extract_day_count(description, labels=("fixed day count",)) or "30/360"
    float_day_count = _extract_day_count(description, labels=("float day count",)) or "ACT/360"
    index_name = _extract_rate_index_after_label(description, label="index") or "SOFR"
    if None in {direction, start, end, notional, fixed_rate} or start >= end:
        return None

    fixed_leg = CouponLeg(
        currency=currency,
        notional_schedule=_constant_notional_schedule(start, end, notional),
        coupon_periods=_coupon_periods(start, end, fixed_frequency, fixing_at_start=False),
        coupon_formula=FixedCouponFormula(fixed_rate),
        day_count=fixed_day_count,
        payment_frequency=fixed_frequency,
        label="fixed_leg",
    )
    floating_leg = CouponLeg(
        currency=currency,
        notional_schedule=_constant_notional_schedule(start, end, notional),
        coupon_periods=_coupon_periods(start, end, float_frequency, fixing_at_start=True),
        coupon_formula=FloatingCouponFormula(_parse_rate_index(index_name)),
        day_count=float_day_count,
        payment_frequency=float_frequency,
        label="floating_leg",
    )
    floating_direction = "receive" if direction == "pay" else "pay"
    return StaticLegContractIR(
        legs=(
            SignedLeg(direction=direction, leg=fixed_leg),
            SignedLeg(direction=floating_direction, leg=floating_leg),
        ),
        settlement=SettlementRule(payout_currency=currency),
        metadata={"family": "fixed_float_swap"},
    )


def _build_static_basis_swap_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> StaticLegContractIR | None:
    lower = description.lower()
    if instrument not in {"", "swap", "basis_swap"} and getattr(product_ir, "instrument", "") not in {"swap"}:
        return None
    if "basis swap" not in lower:
        return None

    start = _extract_named_date(description, labels=("effective", "start"))
    end = _extract_named_date(description, labels=("maturity", "end"))
    notional = _extract_numeric_after(description, labels=("notional",))
    pay_leg = _extract_floating_leg_terms(description, direction="pay")
    receive_leg = _extract_floating_leg_terms(description, direction="receive")
    currency = _extract_currency_code(description) or _currency_from_rate_indices(
        pay_leg[0] if pay_leg is not None else "",
        receive_leg[0] if receive_leg is not None else "",
    ) or "USD"
    if start is None or end is None or notional is None or pay_leg is None or receive_leg is None or start >= end:
        return None

    pay_index, pay_frequency, pay_spread = pay_leg
    receive_index, receive_frequency, receive_spread = receive_leg
    return StaticLegContractIR(
        legs=(
            SignedLeg(
                direction="pay",
                leg=CouponLeg(
                    currency=currency,
                    notional_schedule=_constant_notional_schedule(start, end, notional),
                    coupon_periods=_coupon_periods(start, end, pay_frequency, fixing_at_start=True),
                    coupon_formula=FloatingCouponFormula(
                        _parse_rate_index(pay_index),
                        spread=pay_spread,
                    ),
                    day_count="ACT/360",
                    payment_frequency=pay_frequency,
                    label="pay_leg",
                ),
            ),
            SignedLeg(
                direction="receive",
                leg=CouponLeg(
                    currency=currency,
                    notional_schedule=_constant_notional_schedule(start, end, notional),
                    coupon_periods=_coupon_periods(start, end, receive_frequency, fixing_at_start=True),
                    coupon_formula=FloatingCouponFormula(
                        _parse_rate_index(receive_index),
                        spread=receive_spread,
                    ),
                    day_count="ACT/360",
                    payment_frequency=receive_frequency,
                    label="receive_leg",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency=currency),
        metadata={"family": "basis_swap"},
    )


def _build_static_fixed_coupon_bond_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> StaticLegContractIR | None:
    return _build_fixed_coupon_bond_static_leg_contract(
        description,
        instrument=instrument,
        product_ir=product_ir,
        allow_dynamic=False,
    )


def _build_fixed_coupon_bond_static_leg_contract(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
    allow_dynamic: bool,
) -> StaticLegContractIR | None:
    lower = description.lower()
    if instrument not in {"", "bond", "callable_bond"} and getattr(product_ir, "instrument", "") not in {"bond", "callable_bond"}:
        return None
    if "bond" not in lower:
        return None
    if not allow_dynamic and any(marker in lower for marker in ("callable", "puttable")):
        return None

    currency = _extract_currency_code(description) or "USD"
    face = _extract_numeric_after(description, labels=("face", "notional"))
    coupon = _extract_numeric_after(description, labels=("coupon",))
    issue = _extract_named_date(description, labels=("issue", "effective"))
    maturity = _extract_named_date(description, labels=("maturity", "end"))
    frequency = _extract_frequency_token(description) or "semiannual"
    day_count = _extract_day_count(description, labels=("day count",)) or "ACT/ACT"
    if None in {face, coupon, issue, maturity} or issue >= maturity:
        return None

    coupon_leg = CouponLeg(
        currency=currency,
        notional_schedule=_constant_notional_schedule(issue, maturity, face),
        coupon_periods=_coupon_periods(issue, maturity, frequency, fixing_at_start=False),
        coupon_formula=FixedCouponFormula(coupon),
        day_count=day_count,
        payment_frequency=frequency,
        label="coupon_leg",
    )
    principal_leg = KnownCashflowLeg(
        currency=currency,
        cashflows=(
            KnownCashflow(
                payment_date=maturity,
                amount=face,
                currency=currency,
                label="principal_redemption",
            ),
        ),
        label="principal_leg",
    )
    return StaticLegContractIR(
        legs=(
            SignedLeg(direction="receive", leg=coupon_leg),
            SignedLeg(direction="receive", leg=principal_leg),
        ),
        settlement=SettlementRule(payout_currency=currency),
        metadata={"family": "fixed_coupon_bond"},
    )


def _build_static_period_rate_option_strip_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> StaticLegContractIR | None:
    lower = description.lower()
    product_instrument = getattr(product_ir, "instrument", "")
    admitted_instruments = {
        "",
        "cap",
        "floor",
        "rate_cap_floor_strip",
        "period_rate_option_strip",
    }
    if instrument not in admitted_instruments and product_instrument not in admitted_instruments:
        return None
    if "caplet" in lower or "floorlet" in lower:
        return None

    option_side = _extract_rate_option_strip_side(
        description,
        instrument=instrument,
        product_instrument=product_instrument,
    )
    if option_side is None:
        return None

    start = _extract_named_date(description, labels=("start date", "start", "effective"))
    end = _extract_named_date(description, labels=("end date", "end", "maturity"))
    notional = _extract_numeric_after(description, labels=("notional",))
    strike = _extract_numeric_after(description, labels=("strike",))
    frequency = (
        _extract_frequency_after_label(description, label="payment frequency")
        or _extract_frequency_after_label(description, label="frequency")
        or _extract_frequency_token(description)
        or "quarterly"
    )
    day_count = _extract_day_count(description, labels=("day count",)) or "ACT/360"
    rate_index = (
        _extract_rate_index_after_label(description, label="rate index")
        or _extract_rate_cap_floor_index(description)
    )
    currency = _extract_currency_code(description) or _currency_from_rate_indices(rate_index or "") or "USD"
    if None in {start, end, notional, strike, rate_index} or start >= end:
        return None

    option_leg = PeriodRateOptionStripLeg(
        currency=currency,
        notional_schedule=_constant_notional_schedule(start, end, notional),
        option_periods=_period_rate_option_periods(start, end, frequency),
        rate_index=_parse_rate_index(rate_index),
        strike=strike,
        option_side=option_side,
        day_count=day_count,
        payment_frequency=frequency,
        label="period_rate_option_strip_leg",
        metadata={
            "family": "period_rate_option_strip",
            "instrument_class": _instrument_class_from_period_rate_option_side(option_side),
            "semantic_family": "period_rate_option_strip",
        },
    )
    return StaticLegContractIR(
        legs=(SignedLeg(direction="receive", leg=option_leg),),
        settlement=SettlementRule(payout_currency=currency),
        metadata={
            "family": "period_rate_option_strip",
            "instrument_class": _instrument_class_from_period_rate_option_side(option_side),
        },
    )


def _build_dynamic_autocallable_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> DynamicContractIR | None:
    lower = description.lower()
    product_instrument = getattr(product_ir, "instrument", "")
    if instrument not in {"", "autocallable"} and product_instrument not in {"autocallable"}:
        return None
    if not any(marker in lower for marker in ("autocall", "phoenix", "snowball")):
        return None

    observation_dates = _extract_date_list_after_labels(
        description,
        labels=("observation dates", "observation schedule", "fixing dates"),
    )
    maturity = _extract_named_date(description, labels=("maturity", "expiry", "end"))
    if maturity is None and observation_dates:
        maturity = observation_dates[-1]
    if maturity is None or not observation_dates:
        return None

    barrier = _extract_numeric_after(description, labels=("autocall barrier", "barrier")) or 1.0
    coupon_rate = _extract_numeric_after(description, labels=("coupon",)) or 0.0
    currency = _extract_currency_code(description) or "USD"
    state_schema = StateSchema(
        fields=(
            StateFieldSpec(
                name="coupon_memory",
                domain="float",
                initial_value=0.0,
                tags=("event_state", "coupon_memory"),
            ),
        ),
    )

    buckets: list[EventTimeBucket] = []
    termination_rules: list[TerminationRule] = []
    for event_date in observation_dates:
        events = [
            ObservationEvent(
                label=f"observe_{event_date.isoformat()}",
                schedule_role="observation_dates",
                observed_terms=("underlier_spot",),
            ),
            CouponEvent(
                label=f"coupon_{event_date.isoformat()}",
                schedule_role="observation_dates",
                coupon_formula="conditional_coupon",
                state_updates=(
                    StateUpdateSpec(
                        "coupon_memory",
                        f"0.0 if underlier_spot >= {barrier} else coupon_memory + {coupon_rate}",
                    ),
                ),
            ),
        ]
        phase_sequence: tuple[str, ...]
        if event_date < maturity:
            termination_label = f"autocall_{event_date.isoformat()}"
            events.append(
                AutomaticTerminationEvent(
                    label=termination_label,
                    trigger=f"underlier_spot >= {barrier}",
                    settlement_expression="par_plus_coupon",
                )
            )
            termination_rules.append(
                TerminationRule(
                    label=f"terminate_{event_date.isoformat()}",
                    trigger=f"underlier_spot >= {barrier}",
                    settlement_expression="par_plus_coupon",
                    event_label=termination_label,
                )
            )
            phase_sequence = ("observation", "coupon", "termination")
        else:
            events.append(
                PaymentEvent(
                    label="maturity_redemption",
                    schedule_role="payment_dates",
                    cashflow_formula="terminal_redemption",
                )
            )
            phase_sequence = ("observation", "coupon", "payment")
        buckets.append(
            EventTimeBucket(
                event_date=event_date,
                phase_sequence=phase_sequence,
                events=tuple(events),
            )
        )

    return DynamicContractIR(
        base_contract=None,
        semantic_family="autocallable",
        base_track="payoff_expression",
        state_schema=state_schema,
        event_program=EventProgram(
            buckets=tuple(buckets),
            termination_rules=tuple(termination_rules),
        ),
        settlement=SettlementRule(payout_currency=currency),
    )


def _build_dynamic_tarn_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> DynamicContractIR | None:
    lower = description.lower()
    product_instrument = getattr(product_ir, "instrument", "")
    if instrument not in {"", "tarn", "tarf"} and product_instrument not in {"tarn", "tarf"}:
        return None
    if not any(marker in lower for marker in ("tarn", "tarf", "target redemption")):
        return None

    fixing_dates = _extract_date_list_after_labels(
        description,
        labels=("fixing dates", "observation dates"),
    )
    maturity = _extract_named_date(description, labels=("maturity", "expiry", "end"))
    if maturity is None and fixing_dates:
        maturity = fixing_dates[-1]
    if maturity is None or not fixing_dates:
        return None

    target_level = _extract_numeric_after(description, labels=("target", "target level")) or 0.0
    coupon_rate = _extract_numeric_after(description, labels=("coupon", "coupon rate")) or 0.0
    currency = _extract_currency_code(description) or "USD"
    family = "tarf" if "tarf" in lower else "tarn"
    state_field = "cumulative_gain" if family == "tarf" else "accrued_coupon"
    state_schema = StateSchema(
        fields=(
            StateFieldSpec(
                name=state_field,
                domain="float",
                initial_value=0.0,
                tags=("event_state", "running_total"),
            ),
        ),
    )

    buckets: list[EventTimeBucket] = []
    termination_rules: list[TerminationRule] = []
    for event_date in fixing_dates:
        termination_label = f"target_hit_{event_date.isoformat()}"
        buckets.append(
            EventTimeBucket(
                event_date=event_date,
                phase_sequence=("observation", "coupon", "termination"),
                events=(
                    ObservationEvent(
                        label=f"observe_{event_date.isoformat()}",
                        schedule_role="fixing_dates",
                        observed_terms=("forward_fixing",),
                    ),
                    CouponEvent(
                        label=f"accrue_{event_date.isoformat()}",
                        schedule_role="fixing_dates",
                        coupon_formula="running_target_coupon",
                        state_updates=(
                            StateUpdateSpec(
                                state_field,
                                f"{state_field} + {coupon_rate}",
                            ),
                        ),
                    ),
                    AutomaticTerminationEvent(
                        label=termination_label,
                        trigger=f"{state_field} >= {target_level}",
                        settlement_expression="target_redemption_settlement",
                    ),
                ),
            )
        )
        termination_rules.append(
            TerminationRule(
                label=f"terminate_{event_date.isoformat()}",
                trigger=f"{state_field} >= {target_level}",
                settlement_expression="target_redemption_settlement",
                event_label=termination_label,
            )
        )

    if fixing_dates[-1] != maturity:
        buckets.append(
            EventTimeBucket(
                event_date=maturity,
                phase_sequence=("payment",),
                events=(
                    PaymentEvent(
                        label="maturity_settlement",
                        schedule_role="payment_dates",
                        cashflow_formula="final_target_redemption",
                    ),
                ),
            )
        )

    return DynamicContractIR(
        base_contract=None,
        semantic_family=family,
        base_track="payoff_expression",
        state_schema=state_schema,
        event_program=EventProgram(
            buckets=tuple(buckets),
            termination_rules=tuple(termination_rules),
        ),
        settlement=SettlementRule(payout_currency=currency),
    )


def _build_dynamic_callable_bond_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> DynamicContractIR | None:
    lower = description.lower()
    if instrument not in {"", "callable_bond"} and getattr(product_ir, "instrument", "") not in {"callable_bond"}:
        return None
    if "callable" not in lower or "bond" not in lower:
        return None

    base_contract = _build_fixed_coupon_bond_static_leg_contract(
        description,
        instrument="callable_bond",
        product_ir=product_ir,
        allow_dynamic=True,
    )
    call_dates = _extract_date_list_after_label(description, label="call dates")
    if base_contract is None or not call_dates:
        return None

    maturity = max(
        cashflow.payment_date
        for signed_leg in base_contract.legs
        if isinstance(signed_leg.leg, KnownCashflowLeg)
        for cashflow in signed_leg.leg.cashflows
    )
    call_dates = tuple(call_date for call_date in call_dates if call_date < maturity)
    if not call_dates:
        return None

    redeem = ActionSpec("redeem", "terminate", "redeem at par")
    continue_ = ActionSpec("continue", "continue", "continue outstanding")
    buckets = tuple(
        EventTimeBucket(
            event_date=call_date,
            phase_sequence=("decision", "termination"),
            events=(
                DecisionEvent(
                    label=f"call_{call_date.isoformat()}",
                    schedule_role="call_date",
                    action_set=(redeem, continue_),
                    controller_role="issuer",
                ),
            ),
        )
        for call_date in call_dates
    )
    termination_rules = tuple(
        TerminationRule(
            label=f"terminate_{call_date.isoformat()}",
            trigger="action == redeem",
            settlement_expression="par_redemption",
            event_label=f"call_{call_date.isoformat()}",
        )
        for call_date in call_dates
    )
    return DynamicContractIR(
        base_contract=base_contract,
        semantic_family="callable_bond",
        base_track="static_leg",
        event_program=EventProgram(
            buckets=buckets,
            termination_rules=termination_rules,
        ),
        control_program=ControlProgram(
            controller_role="issuer",
            decision_style="bermudan",
            decision_event_labels=tuple(f"call_{call_date.isoformat()}" for call_date in call_dates),
            admissible_actions=(redeem, continue_),
        ),
        settlement=base_contract.settlement,
    )


def _build_dynamic_swing_option_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> DynamicContractIR | None:
    lower = description.lower()
    product_instrument = getattr(product_ir, "instrument", "")
    if instrument not in {"", "swing_option"} and product_instrument not in {"swing_option"}:
        return None
    if "swing option" not in lower:
        return None

    exercise_dates = _extract_date_list_after_labels(
        description,
        labels=("exercise dates", "decision dates", "observation dates"),
    )
    if not exercise_dates:
        return None
    rights = _extract_integer_after(description, labels=("rights", "max exercises")) or 1
    currency = _extract_currency_code(description) or "USD"
    exercise = ActionSpec(
        "exercise",
        "exercise",
        state_updates=(StateUpdateSpec("remaining_rights", "remaining_rights - 1"),),
    )
    continue_ = ActionSpec("continue", "continue")
    buckets = tuple(
        EventTimeBucket(
            event_date=event_date,
            phase_sequence=("decision", "payment"),
            events=(
                DecisionEvent(
                    label=f"exercise_{event_date.isoformat()}",
                    schedule_role="exercise_dates",
                    action_set=(exercise, continue_),
                    controller_role="holder",
                ),
                PaymentEvent(
                    label=f"exercise_cashflow_{event_date.isoformat()}",
                    schedule_role="settlement_dates",
                    cashflow_formula="swing_exercise_payoff_if_exercised",
                ),
            ),
        )
        for event_date in exercise_dates
    )
    return DynamicContractIR(
        base_contract=None,
        semantic_family="swing_option",
        base_track="payoff_expression",
        state_schema=StateSchema(
            fields=(
                StateFieldSpec(
                    name="remaining_rights",
                    domain="int",
                    initial_value=rights,
                    tags=("inventory_state", "discrete_control"),
                ),
            ),
        ),
        event_program=EventProgram(buckets=buckets),
        control_program=ControlProgram(
            controller_role="holder",
            decision_style="swing",
            decision_event_labels=tuple(f"exercise_{event_date.isoformat()}" for event_date in exercise_dates),
            admissible_actions=(exercise, continue_),
            inventory_fields=("remaining_rights",),
        ),
        settlement=SettlementRule(payout_currency=currency),
    )


def _build_dynamic_gmwb_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> DynamicContractIR | None:
    return _build_gmwb_financial_control_core_contract_ir(
        description,
        instrument=instrument,
        product_ir=product_ir,
        allow_overlay_terms=False,
    )


def _build_gmwb_financial_control_core_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
    allow_overlay_terms: bool,
) -> DynamicContractIR | None:
    lower = description.lower()
    product_instrument = getattr(product_ir, "instrument", "")
    if instrument not in {"", "gmwb"} and product_instrument not in {"gmwb"}:
        return None
    if "gmwb" not in lower and "guaranteed minimum withdrawal benefit" not in lower:
        return None
    if not allow_overlay_terms and re.search(
        r"\b(mortality|lapse|fee|fees|death benefit|policy status|alive|dead|lapsed)\b",
        lower,
    ):
        return None

    withdrawal_dates = _extract_date_list_after_labels(
        description,
        labels=("withdrawal dates", "decision dates", "exercise dates"),
    )
    if not withdrawal_dates:
        return None
    account_value = _extract_numeric_after(description, labels=("account value", "premium")) or 0.0
    guarantee_base = _extract_numeric_after(description, labels=("guarantee base", "benefit base")) or account_value
    currency = _extract_currency_code(description) or "USD"
    withdraw = ActionSpec(
        "withdraw",
        "withdraw",
        action_domain="continuous",
        quantity_source="withdrawal_amount",
        bounds_expression="0 <= withdrawal_amount <= guarantee_base",
        state_updates=(
            StateUpdateSpec("account_value", "account_value - withdrawal_amount"),
            StateUpdateSpec("guarantee_base", "guarantee_base - withdrawal_amount"),
        ),
    )
    buckets = tuple(
        EventTimeBucket(
            event_date=event_date,
            phase_sequence=("decision", "payment"),
            events=(
                DecisionEvent(
                    label=f"withdraw_{event_date.isoformat()}",
                    schedule_role="withdrawal_dates",
                    action_set=(withdraw,),
                    controller_role="holder",
                ),
                PaymentEvent(
                    label=f"withdrawal_cashflow_{event_date.isoformat()}",
                    schedule_role="payment_dates",
                    cashflow_formula="withdrawal_amount",
                ),
            ),
        )
        for event_date in withdrawal_dates
    )
    return DynamicContractIR(
        base_contract=None,
        semantic_family="gmwb",
        base_track="payoff_expression",
        state_schema=StateSchema(
            fields=(
                StateFieldSpec(
                    name="account_value",
                    domain="float",
                    initial_value=account_value,
                    tags=("financial_state", "continuous_control"),
                ),
                StateFieldSpec(
                    name="guarantee_base",
                    domain="float",
                    initial_value=guarantee_base,
                    tags=("financial_state", "continuous_control"),
                ),
            ),
        ),
        event_program=EventProgram(buckets=buckets),
        control_program=ControlProgram(
            controller_role="holder",
            decision_style="continuous_withdrawal",
            decision_event_labels=tuple(
                f"withdraw_{event_date.isoformat()}" for event_date in withdrawal_dates
            ),
            admissible_actions=(withdraw,),
            inventory_fields=("guarantee_base",),
        ),
        settlement=SettlementRule(payout_currency=currency),
    )


def _build_gmwb_insurance_overlay_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> InsuranceOverlayContractIR | None:
    lower = description.lower()
    if not re.search(
        r"\b(mortality|lapse|fee|fees|death benefit|policy status|alive|dead|lapsed)\b",
        lower,
    ):
        return None

    core_contract = _build_gmwb_financial_control_core_contract_ir(
        description,
        instrument=instrument,
        product_ir=product_ir,
        allow_overlay_terms=True,
    )
    if core_contract is None:
        return None

    overlay_events: list[OverlayTransitionEvent | OverlayFeeEvent] = []
    overlay_parameters: list[OverlayParameterSpec] = []

    if re.search(r"\b(mortality|death benefit|dead)\b", lower):
        overlay_events.append(
            OverlayTransitionEvent(
                label="mortality_transition",
                schedule_role="overlay_monitoring",
                trigger_expression="mortality_event_occurs",
                state_updates=(StateUpdateSpec("policy_status", "dead"),),
                cashflow_adjustment=(
                    "death_benefit_adjustment"
                    if "death benefit" in lower
                    else ""
                ),
            )
        )
        overlay_parameters.append(
            OverlayParameterSpec(
                "mortality_hazard",
                "hazard_rate",
                "deferred",
                notes=("future lane must bind an explicit mortality model",),
            )
        )

    if re.search(r"\b(lapse|lapsed)\b", lower):
        overlay_events.append(
            OverlayTransitionEvent(
                label="lapse_transition",
                schedule_role="overlay_monitoring",
                trigger_expression="lapse_event_occurs",
                state_updates=(StateUpdateSpec("policy_status", "lapsed"),),
            )
        )
        overlay_parameters.append(
            OverlayParameterSpec(
                "lapse_hazard",
                "hazard_rate",
                "deferred",
                notes=("future lane must bind an explicit lapse model",),
            )
        )

    fee_rate = _extract_numeric_after(description, labels=("rider fee", "fee", "fees"))
    if fee_rate is not None or re.search(r"\b(fee|fees)\b", lower):
        overlay_events.append(
            OverlayFeeEvent(
                label="rider_fee",
                schedule_role="overlay_fee_dates",
                fee_formula=(
                    "rider_fee_rate * account_value"
                    if fee_rate is not None
                    else "rider_fee_rate * account_value"
                ),
            )
        )
        overlay_parameters.append(
            OverlayParameterSpec(
                "rider_fee_rate",
                "fee_rate",
                fee_rate if fee_rate is not None else "deferred",
                notes=("expressed as a proportion of account_value in the bounded scaffold",),
            )
        )

    if not overlay_events:
        return None

    return InsuranceOverlayContractIR(
        core_contract=core_contract,
        semantic_family="gmwb",
        policy_state_schema=PolicyStateSchema(
            fields=(
                StateFieldSpec(
                    "policy_status",
                    "enum",
                    "alive",
                    tags=("policy_state", "insurance_overlay"),
                ),
            ),
        ),
        overlay_events=tuple(overlay_events),
        overlay_parameters=OverlayParameterSet(parameters=tuple(overlay_parameters)),
        composition_rule=OverlayCompositionRule(
            composition_style="policy_state_gates_financial_control",
            policy_state_field="policy_status",
            notes=(
                "overlay wrapper is representational only and does not widen the executable continuous-control lane",
            ),
        ),
    )


def _is_two_asset_terminal_equity_basket_contract_ir(
    contract_ir: ContractIR | None,
) -> bool:
    """Return whether a ContractIR is a two-asset terminal equity basket option."""
    if contract_ir is None:
        return False
    if (
        str(getattr(getattr(contract_ir, "exercise", None), "style", "") or "").strip()
        != "european"
    ):
        return False
    if (
        str(getattr(getattr(contract_ir, "observation", None), "kind", "") or "").strip()
        != "terminal"
    ):
        return False

    spec = getattr(getattr(contract_ir, "underlying", None), "spec", None)
    if not isinstance(spec, CompositeUnderlying) or len(spec.parts) != 2:
        return False
    if not all(isinstance(part, EquitySpot) for part in spec.parts):
        return False

    payoff = getattr(contract_ir, "payoff", None)
    if not isinstance(payoff, Max) or len(payoff.args) != 2:
        return False
    if not isinstance(payoff.args[1], Constant) or payoff.args[1].value != 0.0:
        return False
    body = payoff.args[0]
    if not isinstance(body, Sub):
        return False
    return isinstance(body.lhs, LinearBasket) or isinstance(body.rhs, LinearBasket)


def _extract_named_date(
    description: str,
    *,
    labels: tuple[str, ...],
) -> date | None:
    for label in labels:
        match = re.search(
            rf"\b{re.escape(label)}\b[^0-9]*(\d{{4}}-\d{{2}}-\d{{2}})",
            description,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return date.fromisoformat(match.group(1))
    return None


def _extract_fixed_leg_direction(description: str) -> str | None:
    lower = description.lower()
    if "pay fixed" in lower:
        return "pay"
    if "receive fixed" in lower:
        return "receive"
    return None


def _extract_currency_code(description: str) -> str | None:
    match = re.search(
        r"\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD)\b",
        description,
    )
    return match.group(1) if match is not None else None


def _extract_frequency_after_label(description: str, *, label: str) -> str | None:
    match = re.search(
        rf"\b{re.escape(label)}\b\s+(annual|semiannual|quarterly|monthly)\b",
        description,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match is not None else None


def _extract_frequency_token(description: str) -> str | None:
    match = re.search(
        r"\b(annual|semiannual|quarterly|monthly)\b",
        description,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match is not None else None


def _extract_day_count(
    description: str,
    *,
    labels: tuple[str, ...],
) -> str | None:
    for label in labels:
        match = re.search(
            rf"\b{re.escape(label)}\b[^A-Z0-9]*(ACT/360|ACT/365|ACT/ACT|30/360)\b",
            description,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return match.group(1).upper()
    return None


def _extract_rate_index_after_label(description: str, *, label: str) -> str | None:
    match = re.search(
        rf"\b{re.escape(label)}\b\s+([A-Z][A-Z0-9_-]*(?:\s+\d+[DWMY])?)",
        description,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match is not None else None


def _extract_floating_leg_terms(
    description: str,
    *,
    direction: str,
) -> tuple[str, str, float] | None:
    match = re.search(
        rf"\b{direction}\b\s+([A-Z][A-Z0-9_-]*)\s+(annual|semiannual|quarterly|monthly)(?:\s+plus\s+(-?\d+(?:\.\d+)?%?))?",
        description,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    spread_token = match.group(3)
    spread = 0.0
    if spread_token:
        spread = float(spread_token[:-1]) / 100.0 if spread_token.endswith("%") else float(spread_token)
    return match.group(1).upper(), match.group(2).lower(), spread


def _currency_from_rate_indices(*indices: str) -> str | None:
    normalized = {index.upper() for index in indices if index}
    if normalized and normalized.issubset({"SOFR", "FF", "FEDFUNDS"}):
        return "USD"
    return None


def _constant_notional_schedule(
    start: date,
    end: date,
    amount: float,
) -> NotionalSchedule:
    return NotionalSchedule(
        (
            NotionalStep(start_date=start, end_date=end, amount=amount),
        )
    )


def _frequency_enum_from_text(token: str) -> Frequency:
    mapping = {
        "annual": Frequency.ANNUAL,
        "semiannual": Frequency.SEMI_ANNUAL,
        "quarterly": Frequency.QUARTERLY,
        "monthly": Frequency.MONTHLY,
    }
    return mapping[token]


def _coupon_periods(
    start: date,
    end: date,
    frequency: str,
    *,
    fixing_at_start: bool,
) -> tuple[CouponPeriod, ...]:
    schedule = generate_schedule(start, end, _frequency_enum_from_text(frequency))
    periods: list[CouponPeriod] = []
    previous = start
    for payment_date in schedule:
        payment = payment_date if isinstance(payment_date, date) else date.fromisoformat(str(payment_date))
        periods.append(
            CouponPeriod(
                accrual_start=previous,
                accrual_end=payment,
                payment_date=payment,
                fixing_date=previous if fixing_at_start else None,
            )
        )
        previous = payment
    return tuple(periods)


def _parse_rate_index(token: str) -> OvernightRateIndex | TermRateIndex:
    normalized = str(token or "").strip().upper().replace(" ", "")
    if normalized in {"SOFR", "FF", "FEDFUNDS"}:
        return OvernightRateIndex("FEDFUNDS" if normalized == "FEDFUNDS" else normalized)
    match = re.fullmatch(r"([A-Z][A-Z0-9_-]*?)[-_]?(\d+[DWMY])", normalized)
    if match is not None:
        return TermRateIndex(match.group(1), match.group(2))
    return OvernightRateIndex(normalized)


def _period_rate_option_periods(
    start: date,
    end: date,
    frequency: str,
) -> tuple[PeriodRateOptionPeriod, ...]:
    schedule = generate_schedule(start, end, _frequency_enum_from_text(frequency))
    periods: list[PeriodRateOptionPeriod] = []
    previous = start
    for payment_date in schedule:
        payment = payment_date if isinstance(payment_date, date) else date.fromisoformat(str(payment_date))
        periods.append(
            PeriodRateOptionPeriod(
                accrual_start=previous,
                accrual_end=payment,
                fixing_date=previous,
                payment_date=payment,
            )
        )
        previous = payment
    return tuple(periods)


def _extract_rate_cap_floor_index(description: str) -> str | None:
    labeled = _extract_rate_index_after_label(description, label="index")
    if labeled is not None:
        return labeled
    match = re.search(
        r"\bon\s+([A-Z][A-Z0-9_-]*(?:-\d+[DWMY])?)\b",
        description,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).strip()
    tokens = re.findall(r"\b[A-Z]{3}-[A-Z0-9_-]+(?:-\d+[DWMY])?\b", description)
    if tokens:
        return tokens[0]
    fallback = re.search(
        r"\b(SOFR|FF|FEDFUNDS|EURIBOR(?:\d+[DWMY])?|LIBOR(?:\d+[DWMY])?)\b",
        description,
        flags=re.IGNORECASE,
    )
    return fallback.group(1) if fallback is not None else None


def _extract_rate_option_strip_side(
    description: str,
    *,
    instrument: str,
    product_instrument: str,
) -> str | None:
    for candidate in (instrument, product_instrument):
        normalized = _normalise(candidate)
        if normalized == "cap":
            return "call"
        if normalized == "floor":
            return "put"

    for label in ("instrument class", "instrument"):
        match = re.search(
            rf"\b{re.escape(label)}\b[^A-Z0-9]*(cap|floor)\b",
            description,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return "call" if match.group(1).lower() == "cap" else "put"

    lower = description.lower()
    saw_cap = bool(re.search(r"\bcap\b", lower))
    saw_floor = bool(re.search(r"\bfloor\b", lower))
    if saw_cap and not saw_floor:
        return "call"
    if saw_floor and not saw_cap:
        return "put"
    return None


def _instrument_class_from_period_rate_option_side(option_side: str) -> str:
    normalized = str(option_side or "").strip().lower()
    if normalized == "call":
        return "cap"
    if normalized == "put":
        return "floor"
    raise ValueError(f"Unsupported period rate option side {option_side!r}")


def _extract_date_list_after_label(
    description: str,
    *,
    label: str,
) -> tuple[date, ...]:
    match = re.search(
        rf"\b{re.escape(label)}\b(.+)$",
        description,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ()
    observed: list[date] = []
    for token in re.findall(r"\d{4}-\d{2}-\d{2}", match.group(1)):
        parsed = date.fromisoformat(token)
        if parsed not in observed:
            observed.append(parsed)
    return tuple(observed)


def _extract_date_list_after_labels(
    description: str,
    *,
    labels: tuple[str, ...],
) -> tuple[date, ...]:
    for label in labels:
        observed = _extract_date_list_after_label(description, label=label)
        if observed:
            return observed
    return ()


def build_product_ir(
    *,
    description: str,
    instrument: str | None = None,
    payoff_family: str | None = None,
    payoff_traits: tuple[str, ...] | list[str] = (),
    exercise_style: str | None = None,
    state_dependence: str | None = None,
    schedule_dependence: bool | None = None,
    model_family: str | None = None,
    candidate_engine_families: tuple[str, ...] | list[str] | None = None,
    required_market_data: frozenset[str] | set[str] | tuple[str, ...] = frozenset(),
    reusable_primitives: tuple[str, ...] | list[str] = (),
    unresolved_primitives: tuple[str, ...] | list[str] | None = None,
    supported: bool | None = None,
    preferred_method: str | None = None,
    store: KnowledgeStore | None = None,
    event_machine: object | None = None,
) -> ProductIR:
    """Build a ``ProductIR`` from explicit structured fields.

    This is the deterministic bridge for user-defined product specifications.
    It reuses the same normalization and inference helpers as the text-based IR
    path, but starts from explicit semantic fields instead of a free-form
    product description.
    """
    if store is None:
        from trellis.agent.knowledge import get_store
        store = get_store()

    normalized_instrument = _normalise(instrument or description)
    normalized_traits = tuple(sorted(set(payoff_traits)))
    resolved_schedule_dependence = (
        _schedule_dependence_for(normalized_instrument, normalized_traits)
        if schedule_dependence is None
        else schedule_dependence
    )
    resolved_state_dependence = (
        _state_dependence_for(normalized_instrument, normalized_traits, resolved_schedule_dependence)
        if state_dependence is None
        else state_dependence
    )
    resolved_exercise_style = (
        _exercise_style_for(normalized_instrument, normalized_traits, description)
        if exercise_style is None
        else exercise_style
    )
    resolved_model_family = (
        _model_family_for(normalized_instrument, normalized_traits, preferred_method or "", description)
        if model_family is None
        else model_family
    )
    resolved_payoff_family = (
        _payoff_family_for(normalized_instrument, normalized_traits, description)
        if not payoff_family
        else payoff_family
    )
    resolved_route_families = _route_families_for(
        normalized_instrument,
        resolved_payoff_family,
        resolved_exercise_style,
        resolved_model_family,
    )
    resolved_engine_families = tuple(candidate_engine_families or _candidate_engine_families_for(
        preferred_method or "",
        resolved_exercise_style,
        normalized_traits,
        resolved_model_family,
    ))
    resolved_unresolved_primitives = tuple(
        unresolved_primitives
        if unresolved_primitives is not None
        else _unresolved_primitives_for(
            normalized_traits,
            resolved_exercise_style,
            resolved_model_family,
        )
    )
    resolved_required_market_data = frozenset(
        normalize_market_data_requirements(
            required_market_data or _market_data_for_traits(normalized_traits, store)
        )
    )
    resolved_reusable_primitives = tuple(
        reusable_primitives or _reusable_primitives_for(
            normalized_traits,
            resolved_model_family,
        )
    )

    return _augment_ir_with_promoted_route_support(_augment_ir_with_contextual_support(ProductIR(
        instrument=normalized_instrument,
        payoff_family=payoff_family or _payoff_family_for(
            normalized_instrument,
            normalized_traits,
            description,
        ),
        payoff_traits=normalized_traits,
        exercise_style=resolved_exercise_style,
        state_dependence=resolved_state_dependence,
        schedule_dependence=resolved_schedule_dependence,
        model_family=resolved_model_family,
        candidate_engine_families=resolved_engine_families,
        route_families=resolved_route_families,
        required_market_data=resolved_required_market_data,
        reusable_primitives=resolved_reusable_primitives,
        unresolved_primitives=resolved_unresolved_primitives,
        supported=len(resolved_unresolved_primitives) == 0 if supported is None else supported,
        event_machine=event_machine,
    ), description))


def _build_vanilla_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    lower = description.lower()
    if instrument not in {"", "european_option", "vanilla_option"} and getattr(product_ir, "payoff_family", "") != "vanilla_option":
        return None
    if any(marker in lower for marker in ("basket", "digital", "asian", "variance", "swaption")):
        return None
    option_side = _extract_option_side(description)
    underlier = _extract_single_underlier(description)
    strike = _extract_numeric_after(description, labels=("strike",))
    expiry = _extract_expiry_date(description)
    if option_side not in {"call", "put"} or underlier is None or strike is None or expiry is None:
        return None
    core = Sub(Spot(underlier), Strike(strike)) if option_side == "call" else Sub(Strike(strike), Spot(underlier))
    return ContractIR(
        payoff=canonicalize(Max((core, Constant(0.0)))),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(spec=EquitySpot(underlier, "gbm")),
    )


def _build_swaption_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    if instrument not in {"", "swaption"} and getattr(product_ir, "payoff_family", "") != "swaption":
        return None
    lower = description.lower()
    if "swaption" not in lower or "bermudan" in lower:
        return None
    expiry = _extract_expiry_date(description)
    strike = _extract_numeric_after(description, labels=("strike",))
    underlier_id, tenor_years = _extract_swaption_underlier(description)
    direction = _extract_swaption_direction(description)
    if expiry is None or strike is None or underlier_id is None or tenor_years is None or direction is None:
        return None
    schedule = _annual_schedule_from_expiry(expiry, tenor_years)
    rate_expr = SwapRate(underlier_id, schedule)
    ramp = Sub(rate_expr, Strike(strike)) if direction == "payer" else Sub(Strike(strike), rate_expr)
    return ContractIR(
        payoff=canonicalize(
            Scaled(
                Annuity(underlier_id, schedule),
                Max((ramp, Constant(0.0))),
            )
        ),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(spec=ForwardRate(underlier_id, "lognormal_forward")),
    )


def _build_basket_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    if instrument not in {"", "basket_option"} and getattr(product_ir, "payoff_family", "") != "basket_option":
        return None
    lower = description.lower()
    if "basket" not in lower:
        return None
    option_side = _extract_option_side(description)
    strike = _extract_numeric_after(description, labels=("strike",))
    expiry = _extract_expiry_date(description)
    basket_terms = _extract_basket_terms(description)
    if option_side not in {"call", "put"} or strike is None or expiry is None or basket_terms is None:
        return None
    basket_expr = LinearBasket(tuple((weight, Spot(name)) for name, weight in basket_terms))
    core = Sub(basket_expr, Strike(strike)) if option_side == "call" else Sub(Strike(strike), basket_expr)
    return ContractIR(
        payoff=canonicalize(Max((core, Constant(0.0)))),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(
            spec=CompositeUnderlying(tuple(EquitySpot(name, "gbm") for name, _ in basket_terms))
        ),
    )


def _build_variance_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    if instrument not in {"", "variance_swap"} and getattr(product_ir, "instrument", "") != "variance_swap":
        return None
    lower = description.lower()
    if "variance swap" not in lower:
        return None
    underlier = _extract_single_underlier(description)
    strike = _extract_numeric_after(description, labels=("variance strike", "strike variance"))
    notional = _extract_numeric_after(description, labels=("notional",))
    expiry = _extract_expiry_date(description)
    if underlier is None or strike is None or notional is None or expiry is None:
        return None
    interval = ContinuousInterval(date(expiry.year, 1, 1), expiry)
    return ContractIR(
        payoff=canonicalize(
            Scaled(
                Constant(notional),
                Sub(VarianceObservable(underlier, interval), Strike(strike)),
            )
        ),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(spec=EquitySpot(underlier, "gbm")),
    )


def _build_digital_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    if instrument not in {"", "digital_option"} and getattr(product_ir, "instrument", "") != "digital_option":
        return None
    lower = description.lower()
    if "digital" not in lower:
        return None
    underlier = _extract_single_underlier(description)
    strike = _extract_numeric_after(description, labels=("strike",))
    if strike is None:
        strike = _extract_digital_threshold(description)
    expiry = _extract_expiry_date(description)
    option_side = _extract_option_side(description)
    if underlier is None or strike is None or expiry is None or option_side not in {"call", "put"}:
        return None
    predicate = Gt(Spot(underlier), Strike(strike)) if option_side == "call" else Lt(Spot(underlier), Strike(strike))
    if "asset-or-nothing" in lower or "asset or nothing" in lower:
        payout_expr = Spot(underlier)
    else:
        payout = _extract_numeric_after(description, labels=("paying",)) or 1.0
        payout_expr = Constant(payout)
    payoff = canonicalize(Mul((payout_expr, Indicator(predicate))))
    return ContractIR(
        payoff=payoff,
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(spec=EquitySpot(underlier, "gbm")),
    )


def _build_asian_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    if instrument not in {"", "asian_option"} and getattr(product_ir, "payoff_family", "") != "asian_option":
        return None
    lower = description.lower()
    if "asian" not in lower or "arithmetic" not in lower:
        return None
    underlier = _extract_single_underlier(description)
    strike = _extract_numeric_after(description, labels=("strike",))
    option_side = _extract_option_side(description)
    averaging_schedule = _extract_asian_schedule(description)
    if underlier is None or strike is None or option_side not in {"call", "put"} or averaging_schedule is None:
        return None
    mean_expr = ArithmeticMean(Spot(underlier), averaging_schedule)
    core = Sub(mean_expr, Strike(strike)) if option_side == "call" else Sub(Strike(strike), mean_expr)
    expiry = averaging_schedule.dates[-1]
    return ContractIR(
        payoff=canonicalize(Max((core, Constant(0.0)))),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="schedule", schedule=averaging_schedule),
        underlying=Underlying(spec=EquitySpot(underlier, "gbm")),
    )


def _build_curve_quote_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    lower = description.lower()
    if instrument not in {"", "quoted_observable", "curve_spread_payoff"}:
        if getattr(product_ir, "instrument", "") not in {"quoted_observable", "curve_spread_payoff"}:
            return None
    if "curve-spread payoff" not in lower and "curve spread payoff" not in lower:
        return None
    if any(marker in lower for marker in (" option", " call ", " put ")):
        return None
    curve_id = _extract_single_underlier(description)
    convention = _extract_curve_quote_convention(description)
    coordinates = _extract_curve_quote_coordinates(description, convention=convention)
    notional = _extract_numeric_after(description, labels=("notional",))
    expiry = _extract_expiry_date(description)
    if curve_id is None or convention is None or coordinates is None or notional is None or expiry is None:
        return None
    lhs, rhs = coordinates
    return ContractIR(
        payoff=canonicalize(
            Scaled(
                Constant(notional),
                Sub(
                    CurveQuote(curve_id, lhs, convention),
                    CurveQuote(curve_id, rhs, convention),
                ),
            )
        ),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(spec=QuoteCurve(curve_id)),
    )


def _build_surface_quote_contract_ir(
    description: str,
    *,
    instrument: str,
    product_ir: ProductIR,
) -> ContractIR | None:
    lower = description.lower()
    if instrument not in {"", "quoted_observable", "vol_skew_payoff"}:
        if getattr(product_ir, "instrument", "") not in {"quoted_observable", "vol_skew_payoff"}:
            return None
    if "vol-skew payoff" not in lower and "vol skew payoff" not in lower:
        return None
    if any(marker in lower for marker in (" option", " call ", " put ")):
        return None
    surface_id = _extract_single_underlier(description)
    convention = _extract_surface_quote_convention(description)
    coordinates = _extract_surface_quote_coordinates(description)
    notional = _extract_numeric_after(description, labels=("notional",))
    expiry = _extract_expiry_date(description)
    if surface_id is None or convention is None or coordinates is None or notional is None or expiry is None:
        return None
    lhs, rhs = coordinates
    return ContractIR(
        payoff=canonicalize(
            Scaled(
                Constant(notional),
                Sub(
                    SurfaceQuote(surface_id, lhs, convention),
                    SurfaceQuote(surface_id, rhs, convention),
                ),
            )
        ),
        exercise=Exercise(style="european", schedule=Singleton(expiry)),
        observation=Observation(kind="terminal", schedule=Singleton(expiry)),
        underlying=Underlying(spec=QuoteSurface(surface_id)),
    )


def _extract_expiry_date(description: str) -> date | None:
    match = re.search(
        r"\b(?:expiring|expiry|at expiry|maturity)\b[^0-9]*(\d{4}-\d{2}-\d{2})",
        description,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return date.fromisoformat(match.group(1))
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", description)
    return date.fromisoformat(iso_dates[-1]) if iso_dates else None


def _extract_numeric_after(description: str, *, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        pattern = rf"\b{re.escape(label)}\b[^0-9-]*\$?(-?\d+(?:\.\d+)?%?)"
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match is None:
            continue
        token = match.group(1).strip()
        if token.endswith("%"):
            return float(token[:-1]) / 100.0
        return float(token)
    return None


def _extract_integer_after(description: str, *, labels: tuple[str, ...]) -> int | None:
    observed = _extract_numeric_after(description, labels=labels)
    return int(observed) if observed is not None else None


def _extract_single_underlier(description: str) -> str | None:
    match = re.search(r"\bon\s+([A-Z][A-Z0-9_.-]*)\b", description)
    if match is not None:
        return match.group(1)
    tokens = re.findall(r"\b[A-Z][A-Z0-9_.-]{1,}\b", description)
    return tokens[0] if tokens else None


def _extract_curve_quote_convention(description: str) -> str | None:
    lower = description.lower()
    if "par rate" in lower:
        return "par_rate"
    if "zero rate" in lower:
        return "zero_rate"
    if "forward rate" in lower:
        return "forward_rate"
    return None


def _extract_curve_quote_coordinates(
    description: str,
    *,
    convention: str | None,
) -> tuple[ParRateTenor | ZeroRateTenor | ForwardRateInterval, ParRateTenor | ZeroRateTenor | ForwardRateInterval] | None:
    if convention is None:
        return None
    if convention == "forward_rate":
        intervals = re.findall(
            r"\b(\d+[DWMY])\s+to\s+(\d+[DWMY])\b",
            description,
            flags=re.IGNORECASE,
        )
        if len(intervals) != 2:
            return None
        left = ForwardRateInterval(intervals[0][0].upper(), intervals[0][1].upper())
        right = ForwardRateInterval(intervals[1][0].upper(), intervals[1][1].upper())
        return left, right

    match = re.search(
        r"\b(\d+[DWMY])\s*(?:-|minus)\s*(\d+[DWMY])\b",
        description,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    left_tenor = match.group(1).upper()
    right_tenor = match.group(2).upper()
    if convention == "par_rate":
        return ParRateTenor(left_tenor), ParRateTenor(right_tenor)
    if convention == "zero_rate":
        return ZeroRateTenor(left_tenor), ZeroRateTenor(right_tenor)
    return None


def _extract_surface_quote_convention(description: str) -> str | None:
    lower = description.lower()
    if "black vol" in lower:
        return "black_vol"
    if "normal vol" in lower:
        return "normal_vol"
    return None


def _extract_surface_quote_coordinates(
    description: str,
) -> tuple[VolPoint | VolDeltaPoint, VolPoint | VolDeltaPoint] | None:
    matches = re.findall(
        r"\b(\d+[DWMY])\s+(-?\d+(?:\.\d+)?)%\s+(moneyness|spot delta|forward delta|delta)\b",
        description,
        flags=re.IGNORECASE,
    )
    if len(matches) != 2:
        return None
    coordinates: list[VolPoint | VolDeltaPoint] = []
    for option_tenor, raw_level, raw_style in matches:
        level = float(raw_level) / 100.0
        style = raw_style.strip().lower().replace(" ", "_")
        if "delta" in style:
            coordinates.append(VolDeltaPoint(option_tenor.upper(), level, style))
        else:
            coordinates.append(VolPoint(option_tenor.upper(), level, style))
    return coordinates[0], coordinates[1]


def _extract_digital_threshold(description: str) -> float | None:
    match = re.search(r"\bspot\s*[<>]\s*(-?\d+(?:\.\d+)?)", description, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def _extract_option_side(description: str) -> str | None:
    lower = description.lower()
    if re.search(r"\bcall\b", lower):
        return "call"
    if re.search(r"\bput\b", lower):
        return "put"
    return None


def _extract_swaption_direction(description: str) -> str | None:
    lower = description.lower()
    if "payer swaption" in lower:
        return "payer"
    if "receiver swaption" in lower:
        return "receiver"
    return None


def _extract_swaption_underlier(description: str) -> tuple[str | None, int | None]:
    explicit = re.search(r"\b([A-Z]{3}-IRS-\d+Y)\b", description)
    if explicit is not None:
        tenor_match = re.search(r"-(\d+)Y$", explicit.group(1))
        return explicit.group(1), int(tenor_match.group(1)) if tenor_match is not None else None
    match = re.search(r"\b(\d+)Y\s+([A-Z]{3})\s+IRS\b", description, flags=re.IGNORECASE)
    if match is None:
        return None, None
    tenor = int(match.group(1))
    currency = match.group(2).upper()
    return f"{currency}-IRS-{tenor}Y", tenor


def _annual_schedule_from_expiry(expiry: date, tenor_years: int) -> FiniteSchedule:
    return FiniteSchedule(
        tuple(
            _clamp_to_valid_day(expiry.year + offset, expiry.month, expiry.day)
            for offset in range(1, tenor_years + 1)
        )
    )


def _extract_basket_terms(description: str) -> tuple[tuple[str, float], ...] | None:
    basket_match = re.search(r"\{([^}]+)\}", description)
    if basket_match is None:
        return None
    inside = basket_match.group(1)
    weighted = re.findall(r"([A-Z][A-Z0-9_.-]*)\s+(-?\d+(?:\.\d+)?)\s*%", inside)
    if weighted:
        return tuple((name, float(weight) / 100.0) for name, weight in weighted)
    names = re.findall(r"[A-Z][A-Z0-9_.-]*", inside)
    if len(names) < 2:
        return None
    equal_weight = 1.0 / len(names)
    return tuple((name, equal_weight) for name in names)


def _extract_asian_schedule(description: str) -> FiniteSchedule | None:
    yearly = re.search(r"\bmonthly average over (\d{4})\b", description, flags=re.IGNORECASE)
    if yearly is not None:
        return _month_end_schedule(int(yearly.group(1)))
    weekly = re.search(
        r"\bweekly average from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\b",
        description,
        flags=re.IGNORECASE,
    )
    if weekly is not None:
        return _weekly_schedule(
            date.fromisoformat(weekly.group(1)),
            date.fromisoformat(weekly.group(2)),
        )
    return None


def _month_end_schedule(year: int) -> FiniteSchedule:
    month_ends = tuple(
        date(year, month, monthrange(year, month)[1])
        for month in range(1, 13)
    )
    return FiniteSchedule(month_ends)


def _weekly_schedule(start: date, end: date) -> FiniteSchedule | None:
    if start > end:
        return None
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current = current + timedelta(days=7)
    if dates[-1] != end:
        dates.append(end)
    return FiniteSchedule(tuple(dates))


def _clamp_to_valid_day(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, monthrange(year, month)[1]))


def retrieval_spec_from_ir(
    ir: ProductIR,
    *,
    preferred_method: str | None = None,
) -> RetrievalSpec:
    """Build a retrieval spec from ProductIR.

    This is the Phase 3 bridge that lets retrieval and prompt guidance use the
    same typed product representation that semantic validation already consumes.
    """
    features = set(ir.payoff_traits)
    features.update(_retrieval_features_from_exercise(ir.exercise_style))
    features.update(_retrieval_features_from_state(ir.state_dependence))
    features.update(_retrieval_features_from_model(ir.model_family))
    features.update(_retrieval_features_from_market_data(ir.required_market_data))
    if normalize_method(preferred_method or "") == "pde_solver":
        features.add("pde_grid")

    return RetrievalSpec(
        method=normalize_method(preferred_method) if preferred_method else None,
        features=sorted(features),
        instrument=ir.instrument,
        exercise_style=ir.exercise_style,
        state_dependence=ir.state_dependence,
        schedule_dependence=ir.schedule_dependence,
        model_family=ir.model_family,
        candidate_engine_families=tuple(ir.candidate_engine_families),
        semantic_text_markers=_semantic_text_markers_from_ir(ir),
        reusable_primitives=tuple(ir.reusable_primitives),
        unresolved_primitives=tuple(ir.unresolved_primitives),
    )


def _normalise(text: str) -> str:
    """Normalise to a decomposition key: lowercase, underscores."""
    return text.lower().strip().replace(" ", "_").replace("-", "_")


def _match_static_decomposition(
    key: str,
    store: KnowledgeStore,
) -> ProductDecomposition | None:
    """Match a static decomposition by exact or fuzzy key."""
    if key in store._decompositions:
        return store._decompositions[key]

    candidates: list[tuple[int, str, ProductDecomposition]] = []
    for known_key, decomp in store._decompositions.items():
        if known_key in key or key in known_key:
            candidates.append((len(known_key), known_key, decomp))
    for known_key, decomp in store._decompositions.items():
        if all(word in key for word in known_key.split("_")):
            candidates.append((len(known_key), known_key, decomp))
    if not candidates:
        return None

    candidates.sort(key=lambda item: -item[0])
    return candidates[0][2]


def _infer_instrument(description: str, instrument_type: str | None) -> str | None:
    """Infer the most specific supported instrument key from text."""
    desc = _normalise(description)
    if instrument_type:
        normalized = _normalise(instrument_type)
        if normalized in {"credit_default_swap"}:
            normalized = "cds"
        if normalized in {"basket_option", "basket_path_payoff"}:
            if any(
                cue in desc
                for cue in (
                    "nth_to_default",
                    "nth to default",
                    "nth-default",
                    "first_to_default",
                    "first to default",
                    "default correlation",
                    "basket cds",
                )
            ):
                return "nth_to_default"
            if any(
                cue in desc
                for cue in (
                    "cdo tranche",
                    "collateralized debt obligation",
                    "attachment",
                    "detachment",
                )
            ):
                return "cdo"
        return normalized

    patterns = [
        ("bermudan_swaption", ("bermudan_swaption", "bermudan swaption")),
        ("callable_bond", ("callable_bond", "callable bond")),
        ("puttable_bond", ("puttable_bond", "puttable bond")),
        ("zcb_option", ("zcb_option", "zcb option", "zero_coupon_bond_option", "zero-coupon bond option")),
        ("american_put", ("american_put", "american put")),
        ("american_option", ("american_option", "american option")),
        ("barrier_option", ("barrier_option", "barrier option")),
        ("asian_option", ("asian_option", "asian option")),
        ("heston_option", ("heston_option", "heston option", "heston")),
        ("variance_swap", ("variance_swap", "variance swap")),
        (
            "credit_loss_distribution",
            (
                "credit_loss_distribution",
                "portfolio_loss_distribution",
                "portfolio loss distribution",
                "multi-name portfolio loss distribution",
                "recursive loss distribution",
            ),
        ),
        ("cds", ("cds", "credit default swap", "credit_default_swap")),
        ("nth_to_default", ("nth_to_default", "nth-to-default", "nth to default")),
        ("swaption", ("swaption",)),
        ("cap", ("cap",)),
        ("floor", ("floor",)),
        ("swap", ("swap",)),
        ("bond", ("bond",)),
    ]
    for instrument, aliases in patterns:
        if any(alias.replace(" ", "_") in desc for alias in aliases):
            return instrument
    if "european_call" in desc or "european_put" in desc or "european_option" in desc:
        return "european_option"
    return None


def _looks_composite(description: str) -> bool:
    """Return whether the description combines multiple primary product traits."""
    desc = _normalise(description)
    composite_markers = [
        "asian",
        "barrier",
        "lookback",
        "american",
        "bermudan",
        "heston",
        "jump",
        "callable",
    ]
    hits = sum(1 for marker in composite_markers if marker in desc)
    return hits >= 3


def _product_ir_from_decomposition(
    *,
    instrument: str,
    decomposition: ProductDecomposition,
    description: str,
    store: KnowledgeStore,
) -> ProductIR:
    """Convert a canonical product decomposition into ``ProductIR``."""
    payoff_traits = tuple(sorted(set(decomposition.features)))
    exercise_style = _exercise_style_for(instrument, payoff_traits, description)
    schedule_dependence = _schedule_dependence_for(instrument, payoff_traits)
    state_dependence = _state_dependence_for(instrument, payoff_traits, schedule_dependence)
    model_family = _model_family_for(instrument, payoff_traits, decomposition.method, description)
    route_families = _route_families_for(
        instrument,
        _payoff_family_for(instrument, payoff_traits, description),
        exercise_style,
        model_family,
    )
    candidate_engine_families = _candidate_engine_families_for(
        decomposition.method,
        exercise_style,
        payoff_traits,
        model_family,
    )
    normalized_desc = _normalise(description)
    if instrument == "bermudan_swaption" and (
        "analytical_lower_bound" in normalized_desc
        or ("analytical" in normalized_desc and "lower_bound" in normalized_desc)
    ):
        route_families = tuple(dict.fromkeys((*route_families, "analytical")))
        candidate_engine_families = tuple(dict.fromkeys((*candidate_engine_families, "analytical")))
    return _augment_ir_with_promoted_route_support(_augment_ir_with_contextual_support(ProductIR(
        instrument=instrument,
        payoff_family=_payoff_family_for(instrument, payoff_traits, description),
        payoff_traits=payoff_traits,
        exercise_style=exercise_style,
        state_dependence=state_dependence,
        schedule_dependence=schedule_dependence,
        model_family=model_family,
        candidate_engine_families=candidate_engine_families,
        route_families=route_families,
        required_market_data=frozenset(
            normalize_market_data_requirements(decomposition.required_market_data)
        ),
        reusable_primitives=decomposition.method_modules,
        unresolved_primitives=(),
        supported=True,
    ), description))


def _infer_composite_ir(
    description: str,
    instrument: str | None,
    store: KnowledgeStore,
) -> ProductIR:
    """Rule-based IR for unsupported or composite products."""
    desc = _normalise(description)
    payoff_traits = _traits_from_text(desc)
    schedule_dependence = _schedule_dependence_for(instrument or "", payoff_traits)
    state_dependence = _state_dependence_for(instrument or "", payoff_traits, schedule_dependence)
    model_family = _model_family_for(instrument or "", payoff_traits, "", description)
    exercise_style = _exercise_style_for(instrument or "", payoff_traits, description)
    route_families = _route_families_for(
        instrument or "",
        _payoff_family_for(instrument or "", payoff_traits, description),
        exercise_style,
        model_family,
    )
    candidate_engine_families = _candidate_engine_families_for(
        "",
        exercise_style,
        payoff_traits,
        model_family,
    )
    required_market_data = frozenset(
        normalize_market_data_requirements(_market_data_for_traits(payoff_traits, store))
    )
    unresolved_primitives = _unresolved_primitives_for(
        payoff_traits,
        exercise_style,
        model_family,
    )
    return _augment_ir_with_promoted_route_support(_augment_ir_with_contextual_support(ProductIR(
        instrument=instrument or _normalise(description),
        payoff_family=_payoff_family_for(instrument or "", payoff_traits, description),
        payoff_traits=payoff_traits,
        exercise_style=exercise_style,
        state_dependence=state_dependence,
        schedule_dependence=schedule_dependence,
        model_family=model_family,
        candidate_engine_families=candidate_engine_families,
        route_families=route_families,
        required_market_data=required_market_data,
        reusable_primitives=_reusable_primitives_for(payoff_traits, model_family),
        unresolved_primitives=unresolved_primitives,
        supported=len(unresolved_primitives) == 0,
    ), description))


def _traits_from_text(desc: str) -> tuple[str, ...]:
    """Infer feature-like payoff traits from free text."""
    trait_aliases = {
        "asian": ("asian",),
        "barrier": ("barrier",),
        "lookback": ("lookback",),
        "callable": ("callable",),
        "puttable": ("puttable",),
        "bermudan": ("bermudan",),
        "american": ("american", "early_exercise"),
        "early_exercise": ("early exercise",),
        "stochastic_vol": ("heston", "stochastic vol", "stochastic_vol"),
        "jump_diffusion": ("jump", "jump_diffusion", "merton"),
        "mean_reversion": ("mean_reversion", "mean reversion", "short rate"),
    }
    traits: set[str] = set()
    for trait, aliases in trait_aliases.items():
        if any(alias.replace(" ", "_") in desc for alias in aliases):
            traits.add(trait)
    if any(
        marker in desc
        for marker in (
            "best_of_two",
            "best_of",
            "rainbow_option",
            "spread_option",
            "kirk_approximation",
            "kirk_spread",
        )
    ):
        traits.add("two_asset_terminal_basket")
    if "option" in desc and "asian" not in traits and "barrier" not in traits:
        traits.add("vanilla_option")
    return tuple(sorted(traits))


def _retrieval_features_from_exercise(exercise_style: str) -> set[str]:
    """Map exercise-style labels onto retrieval features used by the knowledge store."""
    if exercise_style in {"american", "bermudan", "issuer_call", "holder_put"}:
        features = {"early_exercise"}
        if exercise_style == "issuer_call":
            features.add("callable")
        elif exercise_style == "holder_put":
            features.add("puttable")
        return features
    return set()


def _retrieval_features_from_state(state_dependence: str) -> set[str]:
    """Map state-dependence labels onto retrieval features."""
    if state_dependence == "path_dependent":
        return {"path_dependent"}
    if state_dependence == "schedule_dependent":
        return {"backward_induction"}
    return set()


def _retrieval_features_from_model(model_family: str) -> set[str]:
    """Map model-family labels onto retrieval features."""
    if model_family == "interest_rate":
        return {"mean_reversion"}
    if model_family == "stochastic_volatility":
        return {"stochastic_vol"}
    return set()


def _retrieval_features_from_market_data(
    required_market_data: frozenset[str] | set[str] | tuple[str, ...],
) -> set[str]:
    """Map required market-data capabilities onto retrieval features.

    ProductIR carries precise market-data requirements, but the retrieval bridge
    previously dropped them and kept only payoff traits. That made knowledge
    ranking too generic for routes that depend on specific foreign-carry /
    forward-rate contracts, such as the FX vanilla analytical lane.
    """
    mapping = {
        "forward_curve": {"forward_rate"},
        "forecast_curve": {"forward_rate"},
        "fx_rates": {"fx"},
    }
    features: set[str] = set()
    for capability in required_market_data:
        features.update(mapping.get(str(capability).strip(), set()))
    return features


def _semantic_text_markers_from_ir(ir: ProductIR) -> tuple[str, ...]:
    """Build generic high-signal text markers for lesson reranking.

    Keep these markers focused on helper and primitive identity. Broader
    product-family labels are already represented in the indexed retrieval
    features and instrument fields; repeating them here perturbs unrelated
    canary prompt surfaces without adding much disambiguation value.
    """
    raw_markers: list[str] = []
    raw_markers.extend(ir.reusable_primitives)
    raw_markers.extend(ir.unresolved_primitives)
    if ir.model_family == "fx":
        raw_markers.extend(
            value
            for value in (
                ir.instrument,
                ir.payoff_family,
                ir.model_family,
            )
            if value and value != "generic"
        )
        raw_markers.extend(ir.payoff_traits)

    markers: list[str] = []
    seen: set[str] = set()
    for raw in raw_markers:
        text = str(raw).strip()
        if not text:
            continue
        variants = (
            text.lower(),
            text.replace("_", " ").lower(),
            re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip().lower(),
        )
        for variant in variants:
            normalized = " ".join(variant.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            markers.append(normalized)
    return tuple(markers)


def _payoff_family_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    description: str,
) -> str:
    """Map an instrument/trait set onto a stable payoff-family label."""
    product_traits = {"asian", "barrier", "lookback", "callable", "puttable"}
    if instrument == "ranked_observation_basket":
        return "basket_path_payoff"
    if instrument == "basket_option":
        if "ranked_observation" in payoff_traits:
            return "basket_path_payoff"
        return "basket_option"
    if len(product_traits.intersection(payoff_traits)) >= 2:
        return "composite_option"
    if instrument in {"swaption", "bermudan_swaption"}:
        return "swaption"
    if instrument == "cds":
        return EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY
    if instrument == "zcb_option":
        return "zcb_option"
    if instrument == "callable_bond":
        return "callable_fixed_income"
    if instrument == "puttable_bond":
        return "puttable_fixed_income"
    if instrument in {"asian_option"}:
        return "asian_option"
    if instrument in {"barrier_option"}:
        return "barrier_option"
    if instrument in {"american_put", "american_option", "european_option", "heston_option"}:
        return "vanilla_option"
    if instrument == "credit_loss_distribution":
        return "credit_loss_distribution"
    if instrument in {"cdo", "nth_to_default"}:
        return "credit_basket"
    if instrument == "mbs":
        return "mortgage_pool"
    if "option" in _normalise(description):
        return "composite_option"
    return instrument or "generic_product"


def _exercise_style_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    description: str,
) -> str:
    """Infer exercise style from instrument and traits."""
    desc = _normalise(description)
    if instrument == "callable_bond" or "callable" in payoff_traits:
        return "issuer_call"
    if instrument == "puttable_bond":
        return "holder_put"
    if instrument == "bermudan_swaption" or "bermudan" in payoff_traits:
        return "bermudan"
    if instrument in {"american_put", "american_option"} or "american" in payoff_traits or "american" in desc:
        return "american"
    if "option" in desc or instrument in {
        "swaption",
        "barrier_option",
        "asian_option",
        "european_option",
        "heston_option",
    }:
        return "european"
    return "none"


def _schedule_dependence_for(instrument: str, payoff_traits: tuple[str, ...]) -> bool:
    """Return whether the product is schedule dependent."""
    if instrument in {"american_put", "american_option", "european_option", "heston_option", "asian_option", "barrier_option"}:
        return False
    if any(
        trait in payoff_traits
        for trait in ("callable", "puttable", "bermudan", "fixed_coupons", "floating_coupons", "amortization", "prepayment")
    ):
        return True
    return instrument in {
        "bond",
        "swap",
        "cap",
        "floor",
        "swaption",
        "bermudan_swaption",
        "callable_bond",
        "puttable_bond",
        "mbs",
    }


def _state_dependence_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    schedule_dependence: bool,
) -> str:
    """Infer state dependence from traits."""
    if instrument == "barrier_option" and not any(
        trait in payoff_traits for trait in ("asian", "lookback", "path_dependent")
    ):
        return "terminal_markov"
    if any(trait in payoff_traits for trait in ("barrier", "asian", "lookback", "path_dependent", "prepayment", "range_condition")):
        return "path_dependent"
    if schedule_dependence:
        return "schedule_dependent"
    return "terminal_markov"


def _model_family_for(
    instrument: str,
    payoff_traits: tuple[str, ...],
    method: str,
    description: str,
) -> str:
    """Map traits and method to a broad model family."""
    desc = _normalise(description)
    if "stochastic_vol" in payoff_traits or "heston" in desc or instrument == "heston_option":
        return "stochastic_volatility"
    if "jump_diffusion" in payoff_traits:
        return "jump_diffusion"
    if instrument in {"swap", "swaption", "bermudan_swaption", "callable_bond", "puttable_bond", "bond", "cap", "floor", "zcb_option"}:
        return "interest_rate"
    if method == "copula" or instrument in {"cdo", "nth_to_default", "credit_loss_distribution"}:
        return "credit_copula"
    if method == "waterfall" or instrument == "mbs":
        return "cashflow_structured"
    if "option" in desc or instrument in {"barrier_option", "asian_option", "american_put", "american_option", "european_option"}:
        return "equity_diffusion"
    return "generic"


def _candidate_engine_families_for(
    method: str,
    exercise_style: str,
    payoff_traits: tuple[str, ...],
    model_family: str,
) -> tuple[str, ...]:
    """Map canonical methods and traits to conceptual engine families."""
    method_map = {
        "analytical": ("analytical",),
        "rate_tree": ("lattice",),
        "monte_carlo": ("monte_carlo",),
        "qmc": ("qmc",),
        "pde_solver": ("pde",),
        "fft_pricing": ("transforms",),
        "copula": ("copula",),
        "waterfall": ("cashflow",),
    }
    families = list(method_map.get(method, ()))
    if exercise_style not in {"none", "european"} and "exercise" not in families:
        families.append("exercise")
    if any(trait in payoff_traits for trait in ("asian", "barrier", "lookback", "path_dependent")) and "monte_carlo" not in families:
        families.append("monte_carlo")
    if "barrier" in payoff_traits and model_family == "equity_diffusion" and "pde" not in families:
        families.append("pde")
    if model_family == "stochastic_volatility" and "monte_carlo" not in families:
        families.append("monte_carlo")
    return tuple(families)


def _augment_ir_with_contextual_support(ir: ProductIR, description: str) -> ProductIR:
    """Augment ProductIR with high-signal request context missing from static decompositions."""
    if ir.instrument == "quanto_option":
        candidate_engine_families = list(ir.candidate_engine_families)
        for family in ("analytical", "monte_carlo"):
            if family not in candidate_engine_families:
                candidate_engine_families.append(family)

        required_market_data = set(ir.required_market_data)
        required_market_data.update(
            {"discount_curve", "forward_curve", "spot", "black_vol_surface", "fx_rates", "model_parameters"}
        )

        payoff_traits = list(ir.payoff_traits)
        for trait in ("discounting", "fx_translation", "vol_surface_dependence"):
            if trait not in payoff_traits:
                payoff_traits.append(trait)

        return replace(
            ir,
            payoff_family="vanilla_option",
            payoff_traits=tuple(payoff_traits),
            model_family="fx_cross_currency",
            candidate_engine_families=tuple(candidate_engine_families),
            required_market_data=frozenset(
                normalize_market_data_requirements(required_market_data)
            ),
        )

    if not _looks_like_fx_option_context(description, instrument_type=ir.instrument):
        return ir

    candidate_engine_families = list(ir.candidate_engine_families)
    for family in ("analytical", "monte_carlo"):
        if family not in candidate_engine_families:
            candidate_engine_families.append(family)

    required_market_data = set(ir.required_market_data)
    required_market_data.update({"fx_rates", "forward_curve", "spot"})

    return replace(
        ir,
        model_family="fx",
        candidate_engine_families=tuple(candidate_engine_families),
        required_market_data=frozenset(
            normalize_market_data_requirements(required_market_data)
        ),
    )


def _augment_ir_with_promoted_route_support(ir: ProductIR) -> ProductIR:
    """Augment ProductIR with compatible promoted route families from the live registry."""
    from trellis.agent.route_registry import load_route_registry

    route_families = list(ir.route_families)
    candidate_engine_families = list(ir.candidate_engine_families)
    exercise_style = str(getattr(ir, "exercise_style", "") or "").strip().lower()

    for route in load_route_registry().routes:
        if route.status != "promoted":
            continue
        if not _route_matches_product_ir(route, ir):
            continue
        # QUA-909: a route whose scorer declares ``non_european_penalty`` is
        # signaling that it is a lower-bound / fallback approximation for
        # non-European exercise styles and must not be advertised as a
        # first-class candidate engine family against rate-tree / PDE /
        # Monte-Carlo routes that are the true method for those products
        # (e.g. Bermudan swaption selects rate_tree, not the Black76
        # lower-bound helper). Skip the augmentation contribution for such
        # routes; their direct ``match_candidate_routes`` dispatch via the
        # scorer still works.
        if exercise_style and exercise_style != "european":
            score_hints = getattr(route, "score_hints", None) or {}
            non_european_penalty = float(
                score_hints.get("non_european_penalty", 0.0) or 0.0
            )
            if non_european_penalty < 0:
                continue

        route_family = str(getattr(route, "route_family", "") or "").strip()
        if (
            route_family
            and getattr(route, "match_instruments", None) is not None
            and ir.instrument in route.match_instruments
            and route_family not in route_families
        ):
            route_families.append(route_family)

        for engine_family in _candidate_engine_families_from_route(route.engine_family):
            if engine_family and engine_family not in candidate_engine_families:
                candidate_engine_families.append(engine_family)

    if tuple(route_families) == tuple(ir.route_families) and tuple(candidate_engine_families) == tuple(ir.candidate_engine_families):
        return ir
    return replace(
        ir,
        route_families=tuple(route_families),
        candidate_engine_families=tuple(candidate_engine_families),
    )


def _route_matches_product_ir(route, ir: ProductIR) -> bool:
    """Return whether one promoted route is structurally compatible with ProductIR."""
    instrument = getattr(ir, "instrument", None)
    exercise = getattr(ir, "exercise_style", "none")
    payoff_family = getattr(ir, "payoff_family", "")
    payoff_traits = set(getattr(ir, "payoff_traits", ()) or ())
    required_market_data = set(getattr(ir, "required_market_data", ()) or ())

    if route.exclude_instruments and instrument in route.exclude_instruments:
        return False
    if route.match_exercise is not None and exercise not in route.match_exercise:
        return False
    if route.exclude_exercise and exercise in route.exclude_exercise:
        return False
    if route.match_required_market_data is not None and not all(
        item in required_market_data for item in route.match_required_market_data
    ):
        return False
    if route.exclude_required_market_data is not None and any(
        item in required_market_data for item in route.exclude_required_market_data
    ):
        return False

    instrument_ok = route.match_instruments is not None and instrument in route.match_instruments
    payoff_family_ok = route.match_payoff_family is not None and payoff_family in route.match_payoff_family
    payoff_traits_ok = route.match_payoff_traits is not None and bool(
        payoff_traits.intersection(route.match_payoff_traits)
    )
    has_positive_filter = (
        route.match_instruments is not None
        or route.match_payoff_family is not None
        or route.match_payoff_traits is not None
    )
    if has_positive_filter and not (instrument_ok or payoff_family_ok or payoff_traits_ok):
        return False
    return has_positive_filter


def _candidate_engine_families_from_route(engine_family: str) -> tuple[str, ...]:
    """Map one route engine-family label onto ProductIR engine-family hints."""
    normalized = str(engine_family or "").strip().lower()
    mapping = {
        "analytical": ("analytical",),
        "monte_carlo": ("monte_carlo",),
        "qmc": ("qmc",),
        "pde_solver": ("pde",),
        "pde": ("pde",),
        "rate_tree": ("lattice",),
        "tree": ("lattice",),
        "lattice": ("lattice",),
        "copula": ("copula",),
        "waterfall": ("cashflow",),
        "cashflow": ("cashflow",),
    }
    return mapping.get(normalized, (normalized,) if normalized else ())


def _looks_like_fx_option_context(
    description: str | None,
    *,
    instrument_type: str | None = None,
) -> bool:
    """Detect a vanilla FX-option context from free-form request text."""
    if instrument_type == "fx_option":
        return True
    if not description:
        return False
    lower = description.lower()
    if any(
        token in lower
        for token in ("fx option", "fx vanilla", "forex option", "garman-kohlhagen", "gk analytical")
    ):
        return True
    return re.search(r"\b[A-Z]{6}\b", description) is not None


def _route_families_for(
    instrument: str,
    payoff_family: str,
    exercise_style: str,
    model_family: str,
) -> tuple[str, ...]:
    """Return the exact route-family labels that remain semantically valid."""
    families: list[str] = []
    if payoff_family == EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY:
        families.append(EVENT_TRIGGERED_TWO_LEGGED_CONTRACT_FAMILY)
    if instrument == "nth_to_default":
        families.append("nth_to_default")
    if (
        payoff_family == "vanilla_option"
        and exercise_style in {"american", "bermudan"}
        and model_family == "equity_diffusion"
    ):
        families.append("exercise")
        families.append("equity_tree")
    if instrument == "barrier_option" and model_family == "equity_diffusion":
        families.append("pde_solver")
    if (
        instrument in {"callable_bond", "puttable_bond", "bermudan_swaption"}
        or (
            exercise_style in {"issuer_call", "holder_put", "bermudan"}
            and model_family == "interest_rate"
        )
    ):
        families.append("rate_lattice")
    return tuple(dict.fromkeys(families))


def _market_data_for_traits(
    payoff_traits: tuple[str, ...],
    store: KnowledgeStore,
) -> list[str]:
    """Infer market-data needs from known features."""
    market_data: set[str] = set()
    for trait in payoff_traits:
        feature = store._features.get(trait)
        if feature is not None:
            market_data.update(feature.market_data)
    if "barrier" in payoff_traits or "asian" in payoff_traits or "vanilla_option" in payoff_traits:
        market_data.update({"discount_curve", "black_vol_surface"})
    return sorted(market_data)


def _reusable_primitives_for(
    payoff_traits: tuple[str, ...],
    model_family: str,
) -> tuple[str, ...]:
    """Return likely reusable primitives for unsupported composites."""
    primitives: list[str] = []
    if any(trait in payoff_traits for trait in ("barrier", "asian", "lookback")):
        primitives.extend([
            "trellis.models.monte_carlo.engine",
            "trellis.models.processes.gbm",
        ])
    if model_family == "stochastic_volatility":
        primitives.append("trellis.models.processes.heston")
    if "american" in payoff_traits:
        primitives.extend([
            "trellis.models.monte_carlo.lsm",
            "trellis.models.monte_carlo.schemes",
        ])
    return tuple(dict.fromkeys(primitives))


def _unresolved_primitives_for(
    payoff_traits: tuple[str, ...],
    exercise_style: str,
    model_family: str,
) -> tuple[str, ...]:
    """Identify explicit blockers for unsupported composite products."""
    unresolved: list[str] = []
    path_dependent = any(trait in payoff_traits for trait in ("barrier", "asian", "lookback", "path_dependent"))
    if exercise_style == "american" and path_dependent and model_family == "stochastic_volatility":
        unresolved.append("path_dependent_early_exercise_under_stochastic_vol")
    elif exercise_style == "american" and path_dependent:
        unresolved.append("path_dependent_early_exercise")
    elif exercise_style == "american" and model_family == "stochastic_volatility":
        unresolved.append("exercise_under_stochastic_vol")
    return tuple(unresolved)


def _decompose_via_llm(
    description: str,
    key: str,
    store: KnowledgeStore,
    model: str | None,
) -> ProductDecomposition:
    """Use LLM to decompose a novel product into known features."""
    from trellis.agent.config import llm_generate_json, load_env
    from trellis.agent.knowledge.retrieval import (
        format_decomposition_knowledge_for_prompt,
    )
    load_env()

    # Build feature taxonomy context
    feature_list = "\n".join(
        f"- {f.id}: {f.description}"
        + (f" (implies: {', '.join(f.implies)})" if f.implies else "")
        + (f" (method_hint: {f.method_hint})" if f.method_hint else "")
        for f in store._features.values()
    )

    # Available methods
    methods = sorted({d.method for d in store._decompositions.values()})
    heuristic_features = list(_traits_from_text(_normalise(description)))
    instrument_hint = key if key in store._decompositions else _infer_instrument(description, None)
    prior_knowledge = store.retrieve_for_task(
        RetrievalSpec(
            method=None,
            features=heuristic_features,
            instrument=instrument_hint,
            max_lessons=5,
        )
    )
    knowledge_text = format_decomposition_knowledge_for_prompt(prior_knowledge)
    knowledge_section = ""
    if knowledge_text:
        knowledge_section = f"\n\n## Shared Knowledge\n{knowledge_text}"

    prompt = f"""You are a quantitative finance expert decomposing a financial instrument
into its constituent features for a pricing library.

## Available Features
{feature_list}

## Available Pricing Methods
{', '.join(methods)}
{knowledge_section}

## Instrument to Decompose
"{description}"

## Your Task
Decompose this instrument into a list of features from the taxonomy above.
You may also propose NEW feature IDs if needed (use snake_case).
Select the most appropriate pricing method.

Return JSON:
{{
    "features": ["feature1", "feature2", ...],
    "method": "pricing_method",
    "method_modules": ["trellis.models.module1", ...],
    "required_market_data": ["discount_curve", "black_vol_surface", ...],
    "reasoning": "Brief explanation of why this decomposition and method",
    "notes": "Any known complexities or edge cases"
}}"""

    try:
        data = llm_generate_json(prompt, model)
    except Exception:
        # If LLM fails, return a minimal decomposition
        return ProductDecomposition(
            instrument=key,
            features=("discounting",),
            method=normalize_method("analytical"),
            reasoning="LLM decomposition failed — falling back to analytical.",
            learned=True,
        )

    return ProductDecomposition(
        instrument=key,
        features=tuple(data.get("features", ["discounting"])),
        method=normalize_method(data.get("method", "analytical")),
        method_modules=tuple(data.get("method_modules", [])),
        required_market_data=frozenset(data.get("required_market_data", ["discount_curve"])),
        reasoning=data.get("reasoning", ""),
        notes=data.get("notes", ""),
        learned=True,
    )
