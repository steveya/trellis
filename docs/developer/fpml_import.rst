FpML Import And Normalization
=============================

Trellis treats FpML as an explicit external contract source, not as its
internal derivative representation. The executable surface is deliberately
small: one FpML 5.13 confirmation-view ``dataDocument`` containing one regular,
single-currency, constant-notional fixed-float interest-rate swap, one
physically settled European payer/receiver swaption on that swap cohort, or one
scheduled single-currency cap or floor strip.

The versioned :doc:`fpml_support_matrix` is the authoritative distinction
between documents that can be inspected, economics that can be normalized,
contracts that can be executed, and pairs with conformance evidence.

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

Internal Module Boundaries
--------------------------

``trellis.io.fpml.importer`` owns secure document inspection and
``trellis.io.fpml.contracts`` owns immutable body-free reports.
``trellis.io.fpml.normalizer`` remains the stable normalization facade and
owns bounded document validation, product dispatch, and report construction.
The internal ``trellis.io.fpml._normalization_swap`` module owns fixed-float
swap validation, semantic mapping, provenance, and the product-specific
historical-fixing rejection. The internal
``trellis.io.fpml._normalization_cap_floor`` module owns cap/floor validation,
strike and option-side mapping, static-strip construction, and provenance. It
normalizes the external ``capFloor`` label to source-neutral
``period_rate_option_strip`` semantics without selecting a pricing route. The
internal ``trellis.io.fpml._normalization_swaption`` module owns physical
European swaption exercise, buyer/seller, settlement, complete underlying-swap
composition, semantic payoff construction, and provenance. It consumes the
same fixed-float mapper as standalone swap normalization and preserves the
complete underlying swap in canonical contract identity. The internal
``trellis.io.fpml._normalization_common`` module owns product-neutral XML
access, exact blocker and provenance construction, calendars, date and
frequency conventions, regular schedule validation, bounded stream parsing,
and option-premium metadata shared by cap/floor and swaption normalization.

The internal dependency graph is deliberately small and enforced by tests:

.. list-table:: FpML normalization ownership
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Owns
     - May depend on
   * - ``normalizer``
     - document validation, valuation context, product dispatch, economic
       identity selection, and immutable report construction
     - ``_normalization_common`` and every admitted product mapper
   * - ``_normalization_common``
     - product-neutral XML, blocker, provenance, convention, schedule, stream,
       and shared premium parsing
     - no product mapper
   * - ``_normalization_swap``
     - fixed-float swap semantics and swap-specific fixing rejection
     - ``_normalization_common``
   * - ``_normalization_cap_floor``
     - scheduled cap/floor strip semantics
     - ``_normalization_common``
   * - ``_normalization_swaption``
     - physical European swaption semantics and complete underlying-swap
       composition
     - ``_normalization_common`` and ``_normalization_swap``

``tests/test_io/test_fpml_normalizer_boundaries.py`` fixes this graph, the two
function facade surface, stable public exports, mapper identity, and exclusion
from code-generation import authority. Shared and product modules may use
semantic IR types, but cannot import the facade, pricing models, generated
adapters, agent knowledge, task runtime, or route-selection authority.

The public entry point remains ``normalize_fpml_document(...)``. The platform
request compiler also uses the internal
``_normalize_inspected_fpml_document(...)`` seam so it can reconcile document
provenance before lowering economics; modularization must preserve both result
contracts. Shared and product normalization modules may depend on semantic IR
types, but they must not import pricing models, generated adapters, agent
knowledge, or route-selection authority. Moving code across these boundaries
does not change the versioned support matrix.

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

.. _fpml-import-fail-closed:

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

Conformance Evidence
--------------------

``TASKS_FPML_CONFORMANCE.yaml`` is the executable evidence ledger for the
bounded cohort. Each positive task pairs one FpML fixture with independently
declared native terms. ``trellis.agent.fpml_conformance`` builds only existing
generic IR values from those terms and requires the imported and native paths
to agree on economic identity, complete economic projection, structural
selection, deterministic market binding, and price. Non-economic envelope
variants must leave every one of those gates unchanged.

Negative tasks certify the exact honest blocker rather than accepting any
failure as success. They cover missing valuation perspective, unsupported
view/version, incomplete swap economics, and incomplete ``genericProduct`` or
``nonSchemaProduct`` payloads. A mismatch in blocker ids is actionable even
though both outcomes are fail-closed.

The conformance task kind dispatches before agent build and review paths.
Results record zero builder, codegen, quant-review, model-validator, recovery,
and token calls. Persisted records carry body-free import provenance and
clarification evidence. The corpus therefore tests the deterministic import
contract without teaching an agent a product-specific helper or promoting a
cookbook entry.

Extension Rules
---------------

An FpML extension begins with Trellis semantic closure, not an XML product
name. Before an import cohort can become executable, it must have an existing
semantic representation, structural declaration, validation bundle, callable
binding, and deterministic pricing evidence. The normalizer may map XML terms
onto those artifacts; it may not create an FpML-specific pricing helper or let
the product wrapper choose them.

Every extension must add secure positive and negative fixtures, exact blocker
tests, native/FpML canonical-identity and selection parity, and a support-matrix
update. ``genericProduct``, ``nonSchemaProduct``, vendor extensions, or labels
can advance only when their full economics have a separately approved Trellis
semantic contract. An agent-authored adapter or cookbook entry cannot upgrade
the public import support claim by itself.

Place new XML access, convention, or schedule logic in
``_normalization_common`` only when it is product-neutral and reused by more
than one admitted mapper. Product-specific admission rules, blockers, semantic
construction, and provenance belong in one product mapper. A product mapper
may compose another mapper only when the nested product is part of canonical
economic identity, as the admitted swaption composes the complete fixed-float
swap. Such a dependency must be explicit in the boundary test and must not
flow back from the nested mapper to its consumer.

The facade may add a product name only after inspection admits it and an
internal mapper exists. The facade remains responsible for orchestration and
report cardinality; product validation must not move back into it. Internal
normalization modules remain absent from the code-generation import registry,
and neither a generated adapter nor learned cookbook can become import-support
authority.
