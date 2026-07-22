FpML Import And Normalization
=============================

Trellis treats FpML as an explicit external contract source, not as its
internal derivative representation. The executable surface is deliberately
small: one FpML 5.13 confirmation-view ``dataDocument`` containing one regular,
single-currency, constant-notional fixed-float interest-rate swap, one
physically settled European payer/receiver swaption on that swap cohort, or one
scheduled single-currency cap or floor strip.

Lifecycle
---------

The governed request path is deterministic:

1. ``make_fpml_request(...)`` creates an ``ImportedDocumentPayload`` and a
   separate ``TradeEnvelope``.
2. request preflight requires inline bytes plus declared view and version; it
   never resolves an external URI.
3. ``inspect_fpml_document(...)`` applies the secure XML limits and extracts
   body-free document/trade identity.
4. caller and document provenance are reconciled before economics are used.
5. ``normalize_fpml_document(...)`` maps an admitted swap or cap/floor into
   ``StaticLegContractIR``, or an admitted swaption into ``ContractIR`` with
   the complete swap contract nested under ``underlying_contract``.
6. the ordinary structural selector chooses the existing static-leg or
   ContractIR declaration from normalized semantics.
7. the generic ``ExecutionBackedPayoff`` or ``ContractIRPricingPayoff`` adapter
   prices through the shared ``price_normalized_payoff`` platform action.

The FpML ``swap``, ``swaption``, or ``capFloor`` wrapper is only a normalizer
dispatch fact.
It does not select ``static_leg_fixed_float_swap``, the payer/receiver
swaption declaration, a validation bundle, a model, or a solver. Those
artifacts are selected from normalized structure.
The admitted path performs no natural-language parsing, LLM call, code
generation, cookbook update, or model-validator review.

Valuation Perspective
---------------------

An FpML swap stream identifies its payer and receiver, but the document does
not choose whose NPV a caller wants. A pricing request must therefore identify
exactly one ``TradeParty`` with ``role="valuation_party"``. The normalizer maps
each stream to ``pay`` or ``receive`` relative to that party.

A swap maps leg direction relative to the valuation party. A swaption instead
normalizes its underlying swap from the buyer's perspective, preserving payer
or receiver option orientation, and records the requested buyer/seller value
as ``ContractIR.position = "long"`` or ``"short"``. Seller valuation therefore
does not reverse the underlying swap or change the selected declaration.

A cap or floor uses the ``buyer`` and ``seller`` roles on its strike schedule
to identify the option position. The stream maps those ``Payer`` / ``Receiver``
roles to document parties. Buyer valuation produces a receive-side strip;
seller valuation preserves the same strip economics with a pay-side sign.

A missing valuation party produces
``missing_contract_field:fpml_valuation_party_id``. A party that is not in the
document produces ``contract_conflict:fpml_valuation_party_id``. Trellis never
chooses the first party or assumes the reporting party owns the requested NPV.

Admitted Economics
------------------

The first fixed-float cohort requires:

- exactly two ``swapStream`` elements, one fixed and one floating
- one currency and matching positive constant notionals
- matching effective and termination dates
- regular annual, semiannual, quarterly, or monthly calculation and payment
  schedules with one calculation period per payment
- matching payer/receiver counterparty pairs that reference parties declared
  in the document
- fixed rate, supported day counts, and a term floating-rate index whose one
  index tenor matches the calculation and reset frequency
- reset dates relative to calculation-period starts
- fixed and floating rate schedules without steps; a constant spread or
  multiplier may be represented
- internal FpML schedule references that identify the schedule they claim to
  reference

Pricing requests also require a deterministic valuation date. Until the
static-leg runtime consumes historical fixing histories, Trellis rejects a
swap with an unpaid floating coupon whose fixing date is on or before that
valuation date. Build-only normalization remains independent of valuation
date. When an admitted fixed-float or swaption request declares ``analytics``
or ``greeks`` without explicit outputs, the normalized request receives the
bounded rates defaults before execution planning: price/DV01/duration for
analytics and DV01/duration/convexity for Greeks. Structural solver selection
still requests only ``price`` from the normalized payoff; the executor derives
the requested sensitivities through its governed bump-and-reprice analytics
path.

Supported business-day conventions are ``NONE``, ``FOLLOWING``,
``MODFOLLOWING``, ``PRECEDING``, and ``MODPRECEDING``. Adjusted dates require
an admitted explicit business center. The current center map is bounded to the
corresponding Trellis calendars for ``AUSY``, ``BRSP``, ``CATO``, ``CHZU``,
``EUTA``, ``GBLO``, ``JPTO``, and ``USNY``.

The swaption cohort additionally requires exactly one European exercise date,
physical settlement, absent or false ``swaptionStraddle``, one complete admitted
fixed-float underlying swap between exactly the swaption buyer and seller, and
expiry before the swap effective date. A documented third party cannot replace
either option counterparty on the underlying swap. The fixed-leg direction
determines payer versus receiver orientation; exact notional, strike, fixed
payment schedule, day counts, frequencies, and floating index are taken from
the nested swap. The imported contract selects the existing resolved Black-76
swaption declaration. No FpML-specific pricing helper, generated adapter, or
cookbook route is introduced.

The cap/floor cohort requires one complete ``capFloorStream``, exactly one
constant ``capRateSchedule`` or ``floorRateSchedule``, explicit and opposite
``Payer`` / ``Receiver`` buyer and seller roles, a positive constant notional,
zero spread, unit gearing, and the same bounded regular term-index schedule
discipline as the floating swap cohort. A cap maps to a call strip and a floor
to a put strip. Both normalize directly to the existing
``PeriodRateOptionStripLeg`` semantic contract and select the existing
``static_leg_period_rate_option_strip_analytical`` declaration. No
FpML-specific helper, generated adapter, cookbook, or route is introduced.

A premium settled before the valuation date is reported separately in
``FpMLImportReport.premium_metadata`` for swaptions and cap/floors. It is
excluded from canonical contract identity and structural selection. An
unsettled premium blocks pricing because the current normalized payoff path
does not model that cashflow.

When FpML supplies an ``adjustedDate`` for an effective or termination date,
the normalizer requires it to equal the date recomputed from the unadjusted
date and declared adjustments. Payment dates relative to
``CalculationPeriodEndDate`` are anchored to adjusted calculation-period ends
before their own payment-date adjustment is applied.

Fail-Closed Boundary
--------------------

Unsupported amortization, compounding, stubs, cross-currency legs, stepped
rates or spreads, mismatched schedules or notionals, non-term floating-rate
forms, duplicate optional rate schedules, unpaid seasoned floating coupons,
cash-settled, Bermudan, American, partial-exercise, automatic-exercise, or
straddle swaptions, unsettled premiums, cap/floor collars, stepped strikes,
nonzero cap/floor spreads, nonunit gearing, averaging, fixed coupons,
additional payments, or early termination,
document/package siblings, trade-level payments, duplicate roll declarations,
fixed-leg reset schedules, initial-fixing overrides, mismatched counterparty
pairs, end-of-month or clamped high-day rolls, foreign-namespaced extension
children, unsupported children in trade-header, party, product-metadata,
frequency, tenor, date-adjustment, or business-center containers, children
nested inside scalar or reference leaves, unsupported numeric lexical forms
outside XML decimal/integer syntax, calendars/conventions, lifecycle content,
and missing schedule or valuation-date terms produce exact
``external_import:*``,
``missing_contract_field:*``, ``contract_ambiguity:*``, or
``contract_conflict:*`` blockers. The report carries clarification fields when
the caller can supply missing or disambiguating information.

The importer is not a complete FpML schema validator. It supports one view,
version, root, trade count, and bounded product cohorts; it does not claim
package, lifecycle, recordkeeping, basis, cross-currency, OIS compounding,
inflation, cash/Bermudan/American swaption, or cap/floor forms outside the
scheduled constant-strike strip cohort.

Identity And Provenance
-----------------------

``static_leg_economic_identity(...)`` and
``contract_ir_economic_identity(...)`` hash versioned canonical projections of
normalized economics. Labels, metadata, XML paths, document ids, source format,
and separately reported premium metadata are excluded, so equivalent native
and FpML contracts have the same identity. Position, settlement, nested
underlying economics, signed leg direction, schedules, notionals, coupon
formulas, and indices remain part of the relevant identity.

``FpMLImportReport.mapping_provenance`` separately records XML paths and their
normalized semantic fields. ``fpml_import_report_summary(...)`` exposes the
identity, economic projection, and mapping evidence without retaining raw XML
or parser nodes.
