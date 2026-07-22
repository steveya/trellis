"""Fixed-float swap FpML normalization into static-leg semantic IR."""

from __future__ import annotations

from datetime import date

from trellis.agent.static_leg_contract import (
    CouponLeg,
    FixedCouponFormula,
    FloatingCouponFormula,
    NotionalSchedule,
    NotionalStep,
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
    _has_unresolved_historical_fixing,
    _normalize_stream,
    _provenance,
    _reject_nested_metadata_children,
    _reject_unadmitted_direct_children,
)
from trellis.io.fpml.contracts import FpMLFieldProvenance
from trellis.io.fpml.importer import _direct_children


_ALLOWED_SWAP_CHILDREN = {
    "assetClass",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
    "swapStream",
}
_SWAP_METADATA_CHILDREN = _ALLOWED_SWAP_CHILDREN - {"swapStream"}


def _normalize_fixed_float_swap(
    swap,
    *,
    namespace: str | None,
    valuation_party_id: str,
    known_party_ids: tuple[str, ...],
    required_party_pair: frozenset[str] | None = None,
    xml_base_path: str = "/dataDocument/trade/swap",
) -> tuple[StaticLegContractIR, tuple[FpMLFieldProvenance, ...]]:
    _reject_unadmitted_direct_children(
        swap,
        allowed=_ALLOWED_SWAP_CHILDREN,
        scope="swap",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        swap,
        names=_SWAP_METADATA_CHILDREN,
        namespace=namespace,
    )
    streams = _direct_children(swap, "swapStream", namespace=namespace)
    if not streams:
        _fail(
            "missing_contract_field:fpml_swap_stream",
            "contract_gap",
            "A fixed-float swap requires swapStream economics.",
            missing_fields=("swap_stream",),
        )
    if len(streams) != 2:
        _fail(
            "external_import:fpml_swap_stream_count_unsupported",
            "unsupported_contract",
            "The admitted fixed-float swap cohort requires exactly two swap streams.",
        )
    if _contains_any(swap, _STUB_FIELDS, namespace=namespace):
        _fail(
            "external_import:fpml_stub_period_unsupported",
            "unsupported_contract",
            "Explicit first or last stub periods are outside the admitted swap cohort.",
        )
    if _descendants(swap, "compoundingMethod", namespace=namespace):
        _fail(
            "external_import:fpml_compounding_unsupported",
            "unsupported_contract",
            "Compounded swap payments are outside the admitted swap cohort.",
        )
    for stream in streams:
        _reject_unadmitted_direct_children(
            stream,
            allowed=_ALLOWED_STREAM_CHILDREN,
            scope="swap_stream",
            namespace=namespace,
        )
        _reject_nested_metadata_children(
            stream,
            names=_STREAM_METADATA_REFERENCES,
            namespace=namespace,
        )
        notional_schedule = _descendant(
            stream, "notionalStepSchedule", namespace=namespace
        )
        if notional_schedule is not None and _direct_children(
            notional_schedule,
            "step",
            namespace=namespace,
        ):
            _fail(
                "external_import:fpml_amortizing_notional_unsupported",
                "unsupported_contract",
                "Amortizing or accreting notionals are outside the admitted swap cohort.",
            )

    fixed_streams = [
        stream
        for stream in streams
        if _descendant(stream, "fixedRateSchedule", namespace=namespace) is not None
    ]
    floating_streams = [
        stream
        for stream in streams
        if _descendant(stream, "floatingRateCalculation", namespace=namespace)
        is not None
    ]
    if len(fixed_streams) != 1 or len(floating_streams) != 1:
        _fail(
            "external_import:fpml_fixed_float_leg_shape_unsupported",
            "unsupported_contract",
            "The admitted swap cohort requires one fixed stream and one floating stream.",
        )
    if fixed_streams[0] is floating_streams[0]:
        _fail(
            "external_import:fpml_fixed_float_leg_shape_unsupported",
            "unsupported_contract",
            "Fixed and floating economics must belong to distinct swap streams.",
        )
    if _direct_children(fixed_streams[0], "resetDates", namespace=namespace):
        _fail(
            "external_import:fpml_fixed_stream_reset_dates_unsupported",
            "unsupported_contract",
            "Reset-date schedules are not consumed on the admitted fixed stream.",
        )

    fixed_position = streams.index(fixed_streams[0]) + 1
    floating_position = streams.index(floating_streams[0]) + 1

    fixed = _normalize_stream(
        fixed_streams[0],
        kind="fixed",
        namespace=namespace,
        valuation_party_id=valuation_party_id,
    )
    floating = _normalize_stream(
        floating_streams[0],
        kind="floating",
        namespace=namespace,
        valuation_party_id=valuation_party_id,
    )
    referenced_parties = {
        fixed.payer,
        fixed.receiver,
        floating.payer,
        floating.receiver,
    }
    if not referenced_parties.issubset(set(known_party_ids)):
        _fail(
            "contract_conflict:fpml_swap_stream_party_reference",
            "contract_conflict",
            "A swap stream references a party not identified by the FpML document.",
        )
    stream_party_pair = {fixed.payer, fixed.receiver}
    if stream_party_pair != {floating.payer, floating.receiver}:
        _fail(
            "contract_conflict:fpml_swap_stream_counterparties",
            "contract_conflict",
            "Both admitted swap streams must reference the same counterparty pair.",
        )
    if required_party_pair is not None and stream_party_pair != required_party_pair:
        _fail(
            "contract_conflict:fpml_swaption_underlying_counterparties",
            "contract_conflict",
            "The underlying swap must reference the swaption buyer and seller.",
        )
    if fixed.currency != floating.currency:
        _fail(
            "external_import:fpml_cross_currency_swap_unsupported",
            "unsupported_contract",
            "The admitted swap cohort requires one shared leg currency.",
        )
    if fixed.notional != floating.notional:
        _fail(
            "external_import:fpml_mismatched_swap_notionals_unsupported",
            "unsupported_contract",
            "The admitted swap cohort requires matching constant leg notionals.",
        )
    if fixed.start != floating.start or fixed.end != floating.end:
        _fail(
            "external_import:fpml_mismatched_leg_schedules_unsupported",
            "unsupported_contract",
            "The admitted swap cohort requires matching leg effective and termination dates.",
        )
    if {fixed.direction, floating.direction} != {"pay", "receive"}:
        _fail(
            "contract_conflict:fpml_swap_leg_directions",
            "contract_conflict",
            "The two swap streams do not produce opposite valuation-party directions.",
        )

    notional = NotionalSchedule((NotionalStep(fixed.start, fixed.end, fixed.notional),))
    if fixed.fixed_rate is None or floating.rate_index is None:
        raise AssertionError("normalized fixed-float stream terms are incomplete")
    fixed_leg = CouponLeg(
        currency=fixed.currency,
        notional_schedule=notional,
        coupon_periods=fixed.periods,
        coupon_formula=FixedCouponFormula(fixed.fixed_rate),
        day_count=fixed.day_count,
        payment_frequency=fixed.frequency_name,
        label="fixed_leg",
    )
    floating_leg = CouponLeg(
        currency=floating.currency,
        notional_schedule=notional,
        coupon_periods=floating.periods,
        coupon_formula=FloatingCouponFormula(
            floating.rate_index,
            spread=floating.spread,
            gearing=floating.gearing,
        ),
        day_count=floating.day_count,
        payment_frequency=floating.frequency_name,
        label="floating_leg",
    )
    contract = StaticLegContractIR(
        legs=(
            SignedLeg(fixed.direction, fixed_leg),
            SignedLeg(floating.direction, floating_leg),
        ),
        settlement=SettlementRule(payout_currency=fixed.currency),
        metadata={"semantic_family": "fixed_float_swap"},
    )
    fixed_base = f"{xml_base_path}/swapStream[{fixed_position}]"
    floating_base = f"{xml_base_path}/swapStream[{floating_position}]"
    provenance = (
        _provenance("", "valuation_party_id", valuation_party_id),
        _provenance(
            f"{fixed_base}/payerPartyReference/@href",
            "legs[0].direction",
            fixed.direction,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodDates/effectiveDate/unadjustedDate",
            "legs[0].notional_schedule.steps[0].start_date",
            fixed.start,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodDates/terminationDate/unadjustedDate",
            "legs[0].notional_schedule.steps[0].end_date",
            fixed.end,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodAmount/calculation/notionalSchedule/notionalStepSchedule/initialValue",
            "legs[0].notional_schedule.steps[0].amount",
            fixed.notional,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodAmount/calculation/notionalSchedule/notionalStepSchedule/currency",
            "legs[0].currency",
            fixed.currency,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodDates/calculationPeriodFrequency",
            "legs[0].payment_frequency",
            fixed.frequency_name,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodDates",
            "legs[0].coupon_periods",
            len(fixed.periods),
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodAmount/calculation/dayCountFraction",
            "legs[0].day_count",
            fixed.day_count,
        ),
        _provenance(
            f"{fixed_base}/calculationPeriodAmount/calculation/fixedRateSchedule/initialValue",
            "legs[0].coupon_formula.rate",
            fixed.fixed_rate,
        ),
        _provenance(
            f"{floating_base}/payerPartyReference/@href",
            "legs[1].direction",
            floating.direction,
        ),
        _provenance(
            f"{floating_base}/calculationPeriodAmount/calculation/notionalSchedule/notionalStepSchedule/initialValue",
            "legs[1].notional_schedule.steps[0].amount",
            floating.notional,
        ),
        _provenance(
            f"{floating_base}/calculationPeriodAmount/calculation/notionalSchedule/notionalStepSchedule/currency",
            "legs[1].currency",
            floating.currency,
        ),
        _provenance(
            f"{floating_base}/calculationPeriodDates/calculationPeriodFrequency",
            "legs[1].payment_frequency",
            floating.frequency_name,
        ),
        _provenance(
            f"{floating_base}/resetDates",
            "legs[1].coupon_periods",
            len(floating.periods),
        ),
        _provenance(
            f"{floating_base}/calculationPeriodAmount/calculation/dayCountFraction",
            "legs[1].day_count",
            floating.day_count,
        ),
        _provenance(
            f"{floating_base}/calculationPeriodAmount/calculation/floatingRateCalculation",
            "legs[1].coupon_formula.rate_index",
            f"{floating.rate_index.name}:{floating.rate_index.tenor}",
        ),
    )
    if floating.spread_supplied:
        provenance += (
            _provenance(
                f"{floating_base}/calculationPeriodAmount/calculation/"
                "floatingRateCalculation/spreadSchedule/initialValue",
                "legs[1].coupon_formula.spread",
                floating.spread,
            ),
        )
    if floating.gearing_supplied:
        provenance += (
            _provenance(
                f"{floating_base}/calculationPeriodAmount/calculation/"
                "floatingRateCalculation/floatingRateMultiplierSchedule/initialValue",
                "legs[1].coupon_formula.gearing",
                floating.gearing,
            ),
        )
    return contract, provenance


def _reject_unresolved_swap_historical_fixings(
    contract: StaticLegContractIR,
    valuation_date: date,
) -> None:
    if _has_unresolved_historical_fixing(contract, valuation_date):
        _fail(
            "external_import:fpml_historical_fixing_runtime_unsupported",
            "implementation_gap",
            "The static-leg runtime does not yet consume historical fixings "
            "for unpaid seasoned floating coupons.",
        )
