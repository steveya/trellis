Static-Leg Contract IR
======================

``StaticLegContractIR`` is the bounded sibling semantic surface for static
scheduled cashflow products. It exists for products whose contractual meaning
is "exchange these coupon and principal legs on these schedules," not "evaluate
one payoff expression tree at observation time."

Why It Is Separate
------------------

The Phase 2 ``ContractIR`` tree is the right authority surface for static payoff
expressions such as vanilla options, swaptions, variance swaps, digitals, and
terminal quoted spreads. It is the wrong surface for:

- vanilla fixed-float interest-rate swaps
- float-float basis swaps
- fixed coupon bonds

Those products are defined by coupon periods, notionals, accrual conventions,
fixing references, payment conventions, and settlement rules. Encoding them as
fake payoff-expression leaves would just move route logic into the node names.

Current Surface
---------------

The shipped first slice lives in ``trellis.agent.static_leg_contract`` and uses
frozen dataclasses:

- ``StaticLegContractIR(legs, settlement, metadata)``
- ``SignedLeg(direction, leg)``
- ``CouponLeg(currency, notional_schedule, coupon_periods, coupon_formula, ...)``
- ``ConditionalAccrualLeg(currency, notional_schedule, accrual_periods, coupon_formula, accrual_condition, ...)``
- ``PeriodRateOptionStripLeg(currency, notional_schedule, option_periods, rate_index, strike, option_side, ...)``
- ``KnownCashflowLeg(currency, cashflows, ...)``
- ``NotionalSchedule`` / ``NotionalStep``
- schedule nodes:

  - ``CouponPeriod``
  - ``ConditionalAccrualPeriod``
  - ``PeriodRateOptionPeriod``

- coupon formulas:

  - ``FixedCouponFormula``
  - ``FloatingCouponFormula``
  - ``QuotedCouponFormula`` for later quote-linked coupon work

- rate indices:

  - ``OvernightRateIndex``
  - ``TermRateIndex``
  - ``CmsRateIndex``

The current bounded scope is intentionally static:

- no callability or interruption
- no running target state
- no holder or issuer control

Canonical Economic Identity
---------------------------

``static_leg_economic_summary(...)`` projects a ``StaticLegContractIR`` onto
its source-neutral economics, and ``static_leg_economic_identity(...)`` hashes
that projection under the versioned ``static_leg:v1:`` namespace. The
projection includes signed leg direction, dates, notionals, formulas, indices,
and settlement terms. It excludes labels and metadata so source provenance,
document identifiers, and adapter labels cannot change economic identity.

The identity is intentionally product-neutral. The bounded FpML importer uses
it to prove that an imported fixed-float swap or scheduled cap/floor strip and
an equivalent native ``StaticLegContractIR`` are the same economic position
before both enter the ordinary structural selector. XML mapping provenance
remains in ``FpMLImportReport`` and never participates in identity or route
selection.

The deterministic FpML conformance corpus strengthens that identity check
without adding another pricing layer. For fixed-float swaps and scheduled cap
strips it independently constructs the native ``StaticLegContractIR``, then
requires the imported and native contracts to produce the same canonical
projection, lowering declaration, compiled execution binding, and price under
one market scenario. The physical European swaption pair applies the same
contract through ``ContractIR``. These comparisons are evidence about
normalization coherence; the native oracle is not a product helper or route
authority.

The :doc:`../developer/fpml_support_matrix` records this evidence in four
separate stages: secure inspection, economic normalization, executable
structural lowering, and paired conformance. Quant support should be claimed
only at the highest stage actually proved. Parsing an XML wrapper or preserving
its labels is not mathematical evidence about the normalized contract.

``ConditionalAccrualLeg`` does not change that boundary. It represents an
automatic scheduled coupon whose amount is gated by a predicate over observed
or projected quantities. A plain single-index range accrual therefore belongs
to the static-leg algebra as a conditional coupon, not to the option axis. A
callable range accrual, an interrupted accrual, or a range accrual with barrier
state still needs the dynamic wrapper track.

The cap/floor-strip representation boundary is now also explicit at the
leg level:

- ``period_rate_option_strip`` is the canonical semantic family
- ``cap`` / ``floor`` remain compatibility inputs rather than canonical
  semantic family names
- emitted semantic metadata and lowering IR now stay on
  ``period_rate_option_strip`` across task, semantic, and static-leg surfaces
- the admitted ``_agent`` cap/floor wrappers now delegate through the same
  static-leg execution-backed runtime instead of carrying bespoke repricers

In other words, a schedule-driven cap or floor is represented here as a
strip of period rate options rather than as a helper-shaped wrapper name.
The bounded FpML mapping follows the same rule: ``capRateSchedule`` becomes a
call strip, ``floorRateSchedule`` becomes a put strip, and strike-schedule
buyer/seller roles determine the ``SignedLeg`` direction. XML product labels,
party identifiers, and premium provenance do not create another semantic
family or pricing route.

Conditional Accrual Legs
------------------------

``ConditionalAccrualLeg`` is the checked leg-level abstraction for conditional
scheduled cashflows. Its semantic shape is:

- a notional schedule
- accrual periods with observation, fixing, and payment dates
- a coupon formula
- an ``accrual_condition`` predicate
- an ``accrual_counter_ref`` naming the counter semantics used by the coupon

For the first admitted range-accrual slice, the condition is a
``BetweenPredicate`` over one ``RateIndexObservable``. Each period emits a
coupon only when that observed or projected fixing is in range. The principal
redemption, when present, is a separate ``KnownCashflowLeg`` that must pay on
the final coupon payment date.

The observable and predicate grammar is broader than the first executable
route. ``CmsRateObservable``, ``SpreadObservable``, compound predicates, and
multi-index predicates are representable so the compiler can preserve the
economic shape, but route admission fails closed until checked pricing support
exists.

Route-Free Decomposition
------------------------

``trellis.agent.knowledge.decompose.decompose_to_static_leg_contract_ir(...)``
now provides a fixture-driven, route-independent bridge for the first bounded
cohort:

1. vanilla fixed-float IRS
2. SOFR/FF-style float-float basis swaps
3. fixed coupon bonds
4. scheduled cap/floor strips with explicit start/end dates, strike,
   notional, frequency, day-count, and rate-index terms

For the scheduled cap/floor family, decomposition now emits a
``PeriodRateOptionStripLeg`` as the canonical static-leg representation.
Wrapper labels such as ``cap`` and ``floor`` remain compatibility inputs
only; the emitted semantic object is the scheduled strip itself.

This remains a bounded expansion, not a generic leg-product lane:

- route-free decomposition is now present for the bounded admitted
  request surface
- executable static-leg lowering for that strip family now exists on
  the checked analytical and Monte Carlo cap/floor helpers
- the admitted cohort can now lower into ``ContractExecutionIR`` and price
  through the static execution runtime
- fresh-build authority and the `F003`-`F005` parity repair now run on
  the deterministic exact-binding build path for that family
- caplets/floorlets and other unsupported strip variants still fail
  closed rather than pretending to be admitted static-leg contracts

Representative descriptions:

.. code-block:: text

   Vanilla pay fixed USD IRS notional 1000000 fixed rate 4% effective 2025-06-30
   maturity 2030-06-30 fixed semiannual float quarterly index SOFR

   SOFR-FF basis swap notional 1000000 effective 2025-06-30 maturity 2030-06-30
   pay SOFR quarterly receive FF quarterly plus 0.25%

   Fixed coupon bond USD face 1000000 coupon 5% issue 2025-01-15 maturity 2030-01-15
   semiannual day count ACT/ACT

Quoted-observable payoffs and dynamic wrappers fail closed here and are expected
to use their own sibling semantic tracks instead.

Admission And Materialization
-----------------------------

``trellis.agent.static_leg_admission`` is the bounded first lowering-admission
surface for this track.

The current declarations are:

- ``static_leg_fixed_float_swap``
- ``static_leg_period_rate_option_strip_analytical``
- ``static_leg_period_rate_option_strip_monte_carlo``
- ``static_leg_basis_swap``
- ``static_leg_fixed_coupon_bond``
- ``static_leg_range_accrual_discounted``

All of those declarations can now materialize checked repository engines:

- fixed-float IRS -> ``trellis.instruments.swap.SwapPayoff`` via ``SwapSpec``
- scheduled rate-option strips -> ``trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical``
- scheduled rate-option strips -> ``trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo``
- float-float basis swaps -> ``trellis.models.rate_basis_swap.price_rate_basis_swap``
- fixed coupon bond -> ``trellis.instruments.bond.Bond`` kwargs
- single-index range accrual -> ``trellis.models.range_accrual.price_range_accrual``

Range-accrual admission is intentionally exact. The checked declaration
admits only a receive-side ``ConditionalAccrualLeg`` with fixed coupon,
constant positive notional, period settlement, identity observation/fixing
dates, an ``in_range_coupon_count`` counter, one rate-index between predicate,
and at most one receive-side maturity principal cashflow.

Unsupported neighbors produce ``StaticLegAdmissionBlocker`` records rather
than falling through to the checked route. Current blocker ids include:

- ``conditional_range_accrual_callability_pending``
- ``conditional_range_accrual_interruption_state_pending``
- ``conditional_range_accrual_barrier_state_pending``
- ``conditional_accrual_spread_observable_pending``
- ``conditional_accrual_cms_rate_observable_pending``
- ``conditional_accrual_multi_index_predicate_pending``

Those ids are part of the diagnostic contract. They mean the shape was
represented, but executable admission requires a later dynamic-wrapper,
composite-observable, or multi-index pricing slice.

Execution IR Runtime
--------------------

``trellis.execution.compiler.compile_static_leg_execution_ir(...)`` now lowers
the execution-backed static-leg cohort into a route-free
``ContractExecutionIR``.
The lowered artifact records:

- coupon-leg, known-cashflow, conditional-accrual, or
  period-rate-option-strip obligations
- observation, fixing, and payment events derived from the semantic leg periods
- discount, forward, fixing-history, and volatility requirement hints
- a settlement program that sums the execution obligations

The first visitor/runtime set is intentionally small:

- ``execution_event_schedule(...)`` returns a stable event schedule projection
- ``derive_requirement_hints(...)`` derives route-free market and timeline
  requirements from the execution artifact
- ``known_cashflow_obligations(...)`` exposes deterministic cashflow
  obligations for cashflow expansion checks
- ``price_static_leg_execution_ir(...)`` prices the bounded cohort directly
  from the execution artifact and can reuse the same compiled IR across market
  bumps
- ``trellis.core.payoff.ExecutionBackedPayoff`` exposes that execution artifact
  through the public payoff boundary so ``price_payoff(...)`` can consume the
  admitted static-leg runtime without route ids
- ``compile_factor_state_simulation_ir_from_execution_ir(...)`` projects the
  admitted fixed-float swap execution artifact onto the typed
  ``FactorStateSimulationIR`` future-value substrate contract
- ``build_future_value_cube_from_execution_ir(...)`` reuses that same
  fixed-float swap execution artifact to emit a ``FutureValueCube`` through the
  checked Hull-White swap future-value runtime
- ``summarize_discounted_execution_ir(...)`` and
  ``summarize_future_value_execution_ir(...)`` now expose the same execution
  artifact as discounted and future-value precursor summaries for later
  validation, reporting, and xVA-adjacent workflows

This runtime is still a checked static proving lane, but the admitted
cap/floor ``_agent`` wrappers now use it as thin compatibility shells.
Richer static-leg wrapper families and generic dynamic-wrapper execution
remain later work.

The checked single-index range-accrual route now has both static-leg admission
and execution-IR repricing. It binds ``ConditionalAccrualLeg`` evidence to a
``conditional_accrual_leg`` execution obligation, preserves observation and
payment events plus fixing-history/forward-curve requirements, and prices
through ``trellis.models.range_accrual.price_range_accrual``. Unsupported
conditional-accrual variants still stay explicit through blockers.

For the scheduled strip family, the lowering boundary is:

- structural core terms come from ``PeriodRateOptionStripLeg`` itself
- option side maps structurally to ``cap`` or ``floor``
- non-structural pricing knobs such as ``model`` / ``shift`` / ``sabr``
  arrive through the generic normalized-term surface rather than a
  wrapper route id
- the current executable lane assumes a single signed, constant-notional strip;
  receive-side materialization has multiplier ``+1`` and pay-side
  materialization has multiplier ``-1``
- that bounded family now executes on
  the fresh-build exact-binding path and has closed the `F003`-`F005`
  parity slice

The basis-swap family is now materially executable in the bounded slice:

- it selects route-free on semantic structure
- it materializes a checked floating-vs-floating swap helper with
  explicit coupon periods, day-count conventions, rate-index references,
  and spreads
- it lowers into execution IR with explicit coupon obligations, fixing events,
  payment events, and forward-curve requirements
- the current executable lane is still bounded to two constant-notional
  floating legs on term/overnight indices

The future-value bridge is intentionally narrower than the repricing lane:

- today it admits only the vanilla fixed-float IRS execution cohort
- it recompiles the route-free execution artifact onto ``SwapSpec`` plus the
  typed ``FactorStateSimulationIR`` contract
- it then reuses the checked swap future-value substrate rather than inventing
  a second swap-local exposure representation
- basis swaps, bonds, and scheduled rate-option strips still do not have a
  route-free execution-to-future-value bridge

That boundary is deliberate. Static-leg selection exists now so later compiler
work can bind against semantic structure, but the support contract still does
not pretend that the whole leg cohort has cut over to generic route-free
execution.

Relationship To The Dynamic Track
---------------------------------

``StaticLegContractIR`` is also the static substrate for later dynamic wrappers.
Callable coupon notes, range accruals with interruption, and other schedule-based
exotics should layer:

- static legs from this module
- explicit event / state / control structure from
  :doc:`dynamic_contract_ir`

instead of inventing product-name nodes.
