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
- ``KnownCashflowLeg(currency, cashflows, ...)``
- ``NotionalSchedule`` / ``NotionalStep``
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

Route-Free Decomposition
------------------------

``trellis.agent.knowledge.decompose.decompose_to_static_leg_contract_ir(...)``
now provides a fixture-driven, route-independent bridge for the first bounded
cohort:

1. vanilla fixed-float IRS
2. SOFR/FF-style float-float basis swaps
3. fixed coupon bonds

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
- ``static_leg_basis_swap``
- ``static_leg_fixed_coupon_bond``

Two of those declarations can already materialize checked repository engines:

- fixed-float IRS -> ``trellis.instruments.swap.SwapPayoff`` via ``SwapSpec``
- fixed coupon bond -> ``trellis.instruments.bond.Bond`` kwargs

The basis-swap family is explicit but still non-executable in this slice:

- it selects route-free on semantic structure
- its materialization fails closed with ``NotImplementedError`` because no
  checked generic basis-swap executable lowering is landed yet

That boundary is deliberate. Static-leg selection exists now so later compiler
work can bind against semantic structure, but the support contract does not yet
pretend that every admitted leg family has a checked end-to-end lane.

Relationship To The Dynamic Track
---------------------------------

``StaticLegContractIR`` is also the static substrate for later dynamic wrappers.
Callable coupon notes, range accruals with interruption, and other schedule-based
exotics should layer:

- static legs from this module
- explicit event / state / control structure from
  :doc:`dynamic_contract_ir`

instead of inventing product-name nodes.
