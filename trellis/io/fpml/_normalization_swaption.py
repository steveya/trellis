"""Physical European swaption FpML normalization into semantic IR."""

from __future__ import annotations

from datetime import date

from trellis.agent.contract_ir import (
    Annuity,
    Constant,
    ContractIR,
    Exercise,
    FiniteSchedule,
    ForwardRate,
    Max,
    Observation,
    Scaled,
    Singleton,
    Strike,
    Sub,
    SwapRate,
    Underlying,
)
from trellis.agent.static_leg_contract import (
    CouponLeg,
    FixedCouponFormula,
    SettlementRule,
    static_leg_economic_identity,
)
from trellis.io.fpml._normalization_common import (
    _ALLOWED_LEAF_CHILDREN,
    _adjustable_date,
    _fail,
    _normalize_option_premiums,
    _provenance,
    _reject_nested_metadata_children,
    _reject_unadmitted_direct_children,
    _required_child,
    _required_href,
)
from trellis.io.fpml._normalization_swap import _normalize_fixed_float_swap
from trellis.io.fpml.contracts import FpMLFieldProvenance, FpMLPremiumMetadata
from trellis.io.fpml.importer import _direct_children, _optional_text, _split_tag


_ALLOWED_SWAPTION_CHILDREN = {
    "assetClass",
    "buyerPartyReference",
    "cashSettlement",
    "europeanExercise",
    "physicalSettlement",
    "premium",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
    "sellerPartyReference",
    "swap",
    "swaptionStraddle",
}
_SWAPTION_METADATA_CHILDREN = {
    "assetClass",
    "primaryAssetClass",
    "productId",
    "productType",
    "secondaryAssetClass",
}
_ALLOWED_EUROPEAN_EXERCISE_CHILDREN = {"expirationDate"}


def _normalize_european_swaption(
    swaption,
    *,
    namespace: str | None,
    valuation_party_id: str,
    valuation_date: date | None,
    known_party_ids: tuple[str, ...],
) -> tuple[
    ContractIR,
    tuple[FpMLFieldProvenance, ...],
    tuple[FpMLPremiumMetadata, ...],
]:
    """Normalize the bounded physical European fixed-float swaption cohort."""

    exercise_styles = tuple(
        local_name
        for child in tuple(swaption)
        for child_namespace, local_name in (_split_tag(child.tag),)
        if child_namespace == namespace
        and local_name in {"americanExercise", "bermudaExercise", "europeanExercise"}
    )
    if not exercise_styles:
        _fail(
            "missing_contract_field:fpml_swaption_exercise",
            "contract_gap",
            "A swaption requires an explicit exercise declaration.",
            missing_fields=("exercise",),
        )
    if exercise_styles != ("europeanExercise",):
        _fail(
            "external_import:fpml_swaption_exercise_style_unsupported",
            "unsupported_contract",
            "The admitted swaption cohort requires exactly one European exercise.",
        )
    if _direct_children(swaption, "cashSettlement", namespace=namespace):
        _fail(
            "external_import:fpml_swaption_cash_settlement_unsupported",
            "unsupported_contract",
            "Cash-settled swaptions require settlement-method semantics outside "
            "the admitted physical cohort.",
        )
    _reject_unadmitted_direct_children(
        swaption,
        allowed=_ALLOWED_SWAPTION_CHILDREN,
        scope="swaption",
        namespace=namespace,
    )
    _reject_nested_metadata_children(
        swaption,
        names=_SWAPTION_METADATA_CHILDREN,
        namespace=namespace,
    )

    buyer = _required_href(
        swaption,
        "buyerPartyReference",
        namespace=namespace,
        missing_field="buyer_party_reference",
    )
    seller = _required_href(
        swaption,
        "sellerPartyReference",
        namespace=namespace,
        missing_field="seller_party_reference",
    )
    if buyer == seller:
        _fail(
            "contract_conflict:fpml_swaption_parties",
            "contract_conflict",
            "Swaption buyer and seller must differ.",
        )
    if not {buyer, seller}.issubset(set(known_party_ids)):
        _fail(
            "contract_conflict:fpml_swaption_party_reference",
            "contract_conflict",
            "The swaption references a party not identified by the FpML document.",
        )
    if valuation_party_id not in {buyer, seller}:
        _fail(
            "contract_conflict:fpml_valuation_party_id",
            "contract_conflict",
            "The valuation party is neither buyer nor seller of the swaption.",
        )

    physical = _required_child(
        swaption,
        "physicalSettlement",
        namespace=namespace,
        missing_field="physical_settlement",
    )
    _reject_unadmitted_direct_children(
        physical,
        allowed=_ALLOWED_LEAF_CHILDREN,
        scope="physical_settlement",
        namespace=namespace,
    )
    straddle_elements = _direct_children(
        swaption,
        "swaptionStraddle",
        namespace=namespace,
    )
    if len(straddle_elements) > 1:
        _fail(
            "contract_ambiguity:fpml_swaption_straddle",
            "contract_ambiguity",
            "FpML normalization found multiple swaptionStraddle declarations.",
            ambiguous_fields=("swaption_straddle",),
        )
    straddle = (
        _optional_text(straddle_elements[0].text).lower()
        if straddle_elements and _optional_text(straddle_elements[0].text)
        else "false"
    )
    if straddle not in {"true", "false", "1", "0"}:
        _fail(
            "external_import:fpml_malformed_swaption_straddle",
            "malformed_document",
            "FpML swaptionStraddle must use XML boolean syntax.",
        )
    if straddle in {"true", "1"}:
        _fail(
            "external_import:fpml_swaption_straddle_unsupported",
            "unsupported_contract",
            "Swaption straddles are outside the admitted payer/receiver cohort.",
        )

    exercise = _required_child(
        swaption,
        "europeanExercise",
        namespace=namespace,
        missing_field="european_exercise",
    )
    _reject_unadmitted_direct_children(
        exercise,
        allowed=_ALLOWED_EUROPEAN_EXERCISE_CHILDREN,
        scope="european_exercise",
        namespace=namespace,
    )
    _, expiry = _adjustable_date(
        exercise,
        "expirationDate",
        namespace=namespace,
    )
    if valuation_date is not None and valuation_date >= expiry:
        _fail(
            "external_import:fpml_expired_swaption_unsupported",
            "unsupported_contract",
            "The admitted swaption pricing cohort requires valuation before expiry.",
        )

    swaps = _direct_children(swaption, "swap", namespace=namespace)
    if len(swaps) != 1:
        _fail(
            "missing_contract_field:fpml_swaption_underlying_swap"
            if not swaps
            else "contract_ambiguity:fpml_swaption_underlying_swap",
            "contract_gap" if not swaps else "contract_ambiguity",
            "A European swaption requires exactly one complete underlying swap.",
            missing_fields=("underlying_swap",) if not swaps else (),
            ambiguous_fields=("underlying_swap",) if len(swaps) > 1 else (),
        )
    underlying_contract, raw_underlying_provenance = _normalize_fixed_float_swap(
        swaps[0],
        namespace=namespace,
        valuation_party_id=buyer,
        known_party_ids=known_party_ids,
        required_party_pair=frozenset((buyer, seller)),
        xml_base_path="/dataDocument/trade/swaption/swap",
    )
    underlying_provenance = tuple(
        FpMLFieldProvenance(
            xml_path=item.xml_path,
            semantic_field=f"underlying_contract.{item.semantic_field}",
            normalized_value=item.normalized_value,
        )
        for item in raw_underlying_provenance
        if item.semantic_field != "valuation_party_id"
    )
    fixed_signed_leg = next(
        (
            item
            for item in underlying_contract.legs
            if isinstance(item.leg, CouponLeg)
            and isinstance(item.leg.coupon_formula, FixedCouponFormula)
        ),
        None,
    )
    if fixed_signed_leg is None:
        raise AssertionError("normalized fixed-float swap has no fixed leg")
    fixed_leg = fixed_signed_leg.leg
    notional_step = fixed_leg.notional_schedule.steps[0]
    if expiry >= notional_step.start_date:
        _fail(
            "contract_conflict:fpml_swaption_expiry_underlying_start",
            "contract_conflict",
            "Swaption expiry must precede the underlying swap effective date.",
        )
    payment_schedule = FiniteSchedule(
        tuple(period.payment_date for period in fixed_leg.coupon_periods)
    )
    underlier_id = (
        f"{fixed_leg.currency}-IRS-"
        f"{notional_step.start_date:%Y%m%d}-{notional_step.end_date:%Y%m%d}"
    )
    rate = fixed_leg.coupon_formula.rate
    intrinsic = (
        Sub(SwapRate(underlier_id, payment_schedule), Strike(rate))
        if fixed_signed_leg.direction == "pay"
        else Sub(Strike(rate), SwapRate(underlier_id, payment_schedule))
    )
    expiry_schedule = Singleton(expiry)
    contract = ContractIR(
        payoff=Scaled(
            Annuity(underlier_id, payment_schedule),
            Max((intrinsic, Constant(0.0))),
        ),
        exercise=Exercise("european", expiry_schedule),
        observation=Observation("terminal", expiry_schedule),
        underlying=Underlying(ForwardRate(underlier_id, "lognormal_forward")),
        position="long" if valuation_party_id == buyer else "short",
        settlement=SettlementRule(
            settlement_kind="physical",
            payout_currency=fixed_leg.currency,
        ),
        underlying_contract=underlying_contract,
    )
    premium_metadata = _normalize_option_premiums(
        swaption,
        namespace=namespace,
        valuation_date=valuation_date,
        known_party_ids=known_party_ids,
        option_party_ids=(buyer, seller),
        product_scope="swaption",
        product_label="Swaption",
    )
    provenance = (
        _provenance("", "valuation_party_id", valuation_party_id),
        _provenance(
            "/dataDocument/trade/swaption/buyerPartyReference/@href",
            "position",
            contract.position,
        ),
        _provenance(
            "/dataDocument/trade/swaption/europeanExercise/expirationDate/"
            "unadjustedDate",
            "exercise.schedule.t",
            expiry,
        ),
        _provenance(
            "/dataDocument/trade/swaption/physicalSettlement",
            "settlement.settlement_kind",
            "physical",
        ),
        _provenance(
            "/dataDocument/trade/swaption/swap",
            "underlying_contract",
            static_leg_economic_identity(underlying_contract),
        ),
        *underlying_provenance,
    )
    return contract, provenance, premium_metadata
