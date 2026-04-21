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
- ``PeriodRateOptionStripLeg(currency, notional_schedule, option_periods, rate_index, strike, option_side, ...)``
- ``KnownCashflowLeg(currency, cashflows, ...)``
- ``NotionalSchedule`` / ``NotionalStep``
- schedule nodes:

  - ``CouponPeriod``
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

The cap/floor-strip representation boundary is now also explicit at the
leg level:

- ``period_rate_option_strip`` is the canonical semantic family
- ``cap`` / ``floor`` remain wrapper-level compatibility shells
- emitted semantic metadata and lowering IR now stay on
  ``period_rate_option_strip`` across task, semantic, and static-leg surfaces

In other words, a schedule-driven cap or floor is represented here as a
strip of period rate options rather than as a helper-shaped wrapper name.

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

This remains a bounded expansion, not yet a full authoritative
execution lane:

- route-free decomposition is now present for the bounded admitted
  request surface
- executable static-leg lowering for that strip family now exists on
  the checked analytical and Monte Carlo cap/floor helpers
- fresh-build authority and parity repair for that family are still
  follow-on work
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

All of those declarations can now materialize checked repository engines:

- fixed-float IRS -> ``trellis.instruments.swap.SwapPayoff`` via ``SwapSpec``
- scheduled rate-option strips -> ``trellis.models.rate_cap_floor.price_rate_cap_floor_strip_analytical``
- scheduled rate-option strips -> ``trellis.models.rate_cap_floor.price_rate_cap_floor_strip_monte_carlo``
- float-float basis swaps -> ``trellis.models.rate_basis_swap.price_rate_basis_swap``
- fixed coupon bond -> ``trellis.instruments.bond.Bond`` kwargs

For the scheduled strip family, the lowering boundary is:

- structural core terms come from ``PeriodRateOptionStripLeg`` itself
- option side maps structurally to ``cap`` or ``floor``
- non-structural pricing knobs such as ``model`` / ``shift`` / ``sabr``
  arrive through the generic normalized-term surface rather than a
  wrapper route id
- the current executable lane still assumes a single receive-side,
  constant-notional strip and still leaves parity/fresh-build cutover to
  follow-on work

The basis-swap family is now materially executable in the bounded slice:

- it selects route-free on semantic structure
- it materializes a checked floating-vs-floating swap helper with
  explicit coupon periods, day-count conventions, rate-index references,
  and spreads
- the current executable lane is still bounded to two constant-notional
  floating legs on term/overnight indices
- it is still not the authoritative fresh-build path; it is a checked
  bounded lowering lane inside the static-leg program

That boundary is deliberate. Static-leg selection exists now so later compiler
work can bind against semantic structure, but the support contract does not yet
pretend that the whole leg cohort has already cut over to route-free
authoritative execution.

Relationship To The Dynamic Track
---------------------------------

``StaticLegContractIR`` is also the static substrate for later dynamic wrappers.
Callable coupon notes, range accruals with interruption, and other schedule-based
exotics should layer:

- static legs from this module
- explicit event / state / control structure from
  :doc:`dynamic_contract_ir`

instead of inventing product-name nodes.
