"""Cap/floor FpML normalization into scheduled static-strip semantic IR."""

from __future__ import annotations

from datetime import date

from trellis.agent.static_leg_contract import (
    NotionalSchedule,
    NotionalStep,
    PeriodRateOptionPeriod,
    PeriodRateOptionStripLeg,
    SettlementRule,
    SignedLeg,
    StaticLegContractIR,
)
from trellis.io.fpml._normalization_common import (
    _ALLOWED_STREAM_CHILDREN,
    _STREAM_METADATA_REFERENCES,
    _STUB_FIELDS,
    _contains_any,
    _descendant,
    _descendants,
    _fail,
    _finite_float,
    _normalize_option_premiums,
    _normalize_stream,
    _provenance,
    _reject_nested_metadata_children,
    _reject_unadmitted_direct_children,
    _required_child,
    _required_text,
)
from trellis.io.fpml.contracts import FpMLFieldProvenance, FpMLPremiumMetadata
from trellis.io.fpml.importer import _direct_children


_ALLOWED_CAP_FLOOR_CHILDREN = {
    "additionalPayment",
    "assetClass",
    "capFloorStream",
    "earlyTerminationProvision",
    "premium",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
}
_CAP_FLOOR_METADATA_CHILDREN = {
    "assetClass",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
}
_ALLOWED_STRIKE_SCHEDULE_CHILDREN = {"buyer", "initialValue", "seller", "step"}


def _normalize_cap_floor(
    cap_floor,
    *,
    namespace: str | None,
    valuation_party_id: str,
    valuation_date: date | None,
    known_party_ids: tuple[str, ...],
) -> tuple[
    StaticLegContractIR,
    tuple[FpMLFieldProvenance, ...],
    tuple[FpMLPremiumMetadata, ...],
]:
    """Normalize one bounded cap or floor into scheduled strip semantics."""

    _reject_unadmitted_direct_children(
        cap_floor,
        allowed=_ALLOWED_CAP_FLOOR_CHILDREN,
        scope="cap_floor",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        cap_floor,
        names=_CAP_FLOOR_METADATA_CHILDREN,
        namespace=namespace,
    )
    if _direct_children(cap_floor, "additionalPayment", namespace=namespace):
        _fail(
            "external_import:fpml_cap_floor_additional_payment_unsupported",
            "unsupported_contract",
            "Additional cap/floor payments are outside the admitted cohort.",
        )
    if _direct_children(
        cap_floor,
        "earlyTerminationProvision",
        namespace=namespace,
    ):
        _fail(
            "external_import:fpml_cap_floor_early_termination_unsupported",
            "unsupported_contract",
            "Early-termination provisions are outside the admitted cap/floor cohort.",
        )
    streams = _direct_children(cap_floor, "capFloorStream", namespace=namespace)
    if len(streams) != 1:
        _fail(
            "missing_contract_field:fpml_cap_floor_stream"
            if not streams
            else "contract_ambiguity:fpml_cap_floor_stream",
            "contract_gap" if not streams else "contract_ambiguity",
            "An admitted cap or floor requires exactly one capFloorStream.",
            missing_fields=("cap_floor_stream",) if not streams else (),
            ambiguous_fields=("cap_floor_stream",) if len(streams) > 1 else (),
        )
    stream = streams[0]
    _reject_unadmitted_direct_children(
        stream,
        allowed=_ALLOWED_STREAM_CHILDREN,
        scope="cap_floor_stream",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        stream,
        names=_STREAM_METADATA_REFERENCES,
        namespace=namespace,
    )
    if _contains_any(stream, _STUB_FIELDS, namespace=namespace):
        _fail(
            "external_import:fpml_stub_period_unsupported",
            "unsupported_contract",
            "Explicit first or last stub periods are outside the admitted cap/floor cohort.",
        )
    if _descendants(stream, "compoundingMethod", namespace=namespace):
        _fail(
            "external_import:fpml_compounding_unsupported",
            "unsupported_contract",
            "Compounded cap/floor periods are outside the admitted cohort.",
        )
    if _descendants(stream, "fixedRateSchedule", namespace=namespace):
        _fail(
            "external_import:fpml_cap_floor_fixed_coupon_unsupported",
            "unsupported_contract",
            "A fixed coupon cannot be combined with the admitted cap/floor strip.",
        )
    notional_schedule = _descendant(
        stream,
        "notionalStepSchedule",
        namespace=namespace,
    )
    if notional_schedule is not None and _direct_children(
        notional_schedule,
        "step",
        namespace=namespace,
    ):
        _fail(
            "external_import:fpml_amortizing_notional_unsupported",
            "unsupported_contract",
            "Amortizing or accreting notionals are outside the admitted cap/floor cohort.",
        )

    floating = _required_child(
        _required_child(
            _required_child(
                stream,
                "calculationPeriodAmount",
                namespace=namespace,
                missing_field="calculation_period_amount",
            ),
            "calculation",
            namespace=namespace,
            missing_field="calculation",
        ),
        "floatingRateCalculation",
        namespace=namespace,
        missing_field="floating_rate_calculation",
    )
    cap_schedules = _direct_children(
        floating,
        "capRateSchedule",
        namespace=namespace,
    )
    floor_schedules = _direct_children(
        floating,
        "floorRateSchedule",
        namespace=namespace,
    )
    if _direct_children(floating, "averagingMethod", namespace=namespace):
        _fail(
            "external_import:fpml_cap_floor_averaging_unsupported",
            "unsupported_contract",
            "Averaged rate observations are outside the admitted cap/floor cohort.",
        )
    if cap_schedules and floor_schedules:
        _fail(
            "external_import:fpml_cap_floor_collar_unsupported",
            "unsupported_contract",
            "A combined cap and floor schedule is a collar outside the admitted cohort.",
        )
    schedules = cap_schedules or floor_schedules
    if not schedules:
        _fail(
            "missing_contract_field:fpml_cap_floor_strike_schedule",
            "contract_gap",
            "An admitted cap or floor requires one strike schedule.",
            missing_fields=("strike_schedule",),
        )
    if len(schedules) != 1:
        _fail(
            "contract_ambiguity:fpml_cap_floor_strike_schedule",
            "contract_ambiguity",
            "An admitted cap or floor requires exactly one strike schedule.",
            ambiguous_fields=("strike_schedule",),
        )
    strike_schedule = schedules[0]
    option_side = "call" if cap_schedules else "put"
    strike, buyer_role, seller_role = _normalize_cap_floor_strike_schedule(
        strike_schedule,
        namespace=namespace,
    )

    terms = _normalize_stream(
        stream,
        kind="floating",
        namespace=namespace,
        valuation_party_id=valuation_party_id,
        allow_rate_option_strikes=True,
        stream_scope="cap_floor_stream",
    )
    if not {terms.payer, terms.receiver}.issubset(set(known_party_ids)):
        _fail(
            "contract_conflict:fpml_cap_floor_party_reference",
            "contract_conflict",
            "The cap/floor stream references a party not identified by the document.",
        )
    if terms.rate_index is None:
        raise AssertionError("normalized cap/floor stream has no rate index")
    if terms.spread != 0.0 or terms.gearing != 1.0:
        _fail(
            "external_import:fpml_cap_floor_coupon_transform_unsupported",
            "unsupported_contract",
            "The admitted pure cap/floor strip requires zero spread and unit gearing.",
        )

    role_parties = {"Payer": terms.payer, "Receiver": terms.receiver}
    option_buyer = role_parties[buyer_role]
    option_seller = role_parties[seller_role]
    if valuation_party_id == option_buyer:
        direction = "receive"
    elif valuation_party_id == option_seller:
        direction = "pay"
    else:  # pragma: no cover - _normalize_stream already proves membership
        raise AssertionError("valuation party is outside cap/floor counterparties")

    notional = NotionalSchedule((NotionalStep(terms.start, terms.end, terms.notional),))
    option_periods = tuple(
        PeriodRateOptionPeriod(
            accrual_start=period.accrual_start,
            accrual_end=period.accrual_end,
            fixing_date=period.fixing_date,
            payment_date=period.payment_date,
        )
        for period in terms.periods
        if period.fixing_date is not None
    )
    if len(option_periods) != len(terms.periods):
        raise AssertionError("normalized cap/floor period is missing a fixing date")
    contract = StaticLegContractIR(
        legs=(
            SignedLeg(
                direction,
                PeriodRateOptionStripLeg(
                    currency=terms.currency,
                    notional_schedule=notional,
                    option_periods=option_periods,
                    rate_index=terms.rate_index,
                    strike=strike,
                    option_side=option_side,
                    day_count=terms.day_count,
                    payment_frequency=terms.frequency_name,
                    label="cap_strip" if option_side == "call" else "floor_strip",
                ),
            ),
        ),
        settlement=SettlementRule(payout_currency=terms.currency),
        metadata={"semantic_family": "period_rate_option_strip"},
    )
    strike_name = "capRateSchedule" if option_side == "call" else "floorRateSchedule"
    base = "/dataDocument/trade/capFloor/capFloorStream"
    floating_base = (
        f"{base}/calculationPeriodAmount/calculation/floatingRateCalculation"
    )
    provenance = (
        _provenance("", "valuation_party_id", valuation_party_id),
        _provenance(
            f"{floating_base}/{strike_name}/buyer",
            "legs[0].direction",
            direction,
        ),
        _provenance(
            f"{base}/calculationPeriodDates",
            "legs[0].option_periods",
            len(option_periods),
        ),
        _provenance(
            f"{base}/calculationPeriodDates/effectiveDate/unadjustedDate",
            "legs[0].notional_schedule.steps[0].start_date",
            terms.start,
        ),
        _provenance(
            f"{base}/calculationPeriodDates/terminationDate/unadjustedDate",
            "legs[0].notional_schedule.steps[0].end_date",
            terms.end,
        ),
        _provenance(
            f"{base}/calculationPeriodAmount/calculation/notionalSchedule/"
            "notionalStepSchedule/initialValue",
            "legs[0].notional_schedule.steps[0].amount",
            terms.notional,
        ),
        _provenance(
            f"{base}/calculationPeriodAmount/calculation/notionalSchedule/"
            "notionalStepSchedule/currency",
            "legs[0].currency",
            terms.currency,
        ),
        _provenance(
            f"{base}/calculationPeriodAmount/calculation/dayCountFraction",
            "legs[0].day_count",
            terms.day_count,
        ),
        _provenance(
            f"{base}/paymentDates/paymentFrequency",
            "legs[0].payment_frequency",
            terms.frequency_name,
        ),
        _provenance(
            f"{floating_base}/floatingRateIndex",
            "legs[0].rate_index",
            f"{terms.rate_index.name}:{terms.rate_index.tenor}",
        ),
        _provenance(
            f"{floating_base}/{strike_name}/initialValue",
            "legs[0].strike",
            strike,
        ),
        _provenance(
            f"{floating_base}/{strike_name}",
            "legs[0].option_side",
            option_side,
        ),
    )
    premium_metadata = _normalize_option_premiums(
        cap_floor,
        namespace=namespace,
        valuation_date=valuation_date,
        known_party_ids=known_party_ids,
        option_party_ids=(option_buyer, option_seller),
        product_scope="cap_floor",
        product_label="Cap/floor",
    )
    return contract, provenance, premium_metadata


def _normalize_cap_floor_strike_schedule(
    schedule,
    *,
    namespace: str | None,
) -> tuple[float, str, str]:
    if _direct_children(schedule, "step", namespace=namespace):
        _fail(
            "external_import:fpml_step_strike_schedule_unsupported",
            "unsupported_contract",
            "Stepped cap/floor strikes are outside the admitted cohort.",
        )
    _reject_unadmitted_direct_children(
        schedule,
        allowed=_ALLOWED_STRIKE_SCHEDULE_CHILDREN,
        scope="cap_floor_strike_schedule",
        namespace=namespace,
    )
    strike = _finite_float(
        _required_text(
            schedule,
            "initialValue",
            namespace=namespace,
            missing_field="strike",
        ),
        field_name="strike",
    )
    buyer = _required_text(
        schedule,
        "buyer",
        namespace=namespace,
        missing_field="strike_buyer",
    )
    seller = _required_text(
        schedule,
        "seller",
        namespace=namespace,
        missing_field="strike_seller",
    )
    if {buyer, seller} != {"Payer", "Receiver"}:
        _fail(
            "contract_conflict:fpml_cap_floor_option_parties",
            "contract_conflict",
            "Cap/floor strike buyer and seller must be opposite Payer/Receiver roles.",
        )
    return strike, buyer, seller
