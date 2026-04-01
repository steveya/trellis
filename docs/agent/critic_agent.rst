Critic Agent & Arbiter
======================

The critic agent reads generated code and selects from a bounded menu of
deterministic review checks. The arbiter executes those checks deterministically.
Legacy critic-authored ``test_code`` remains supported for backward compatibility.

Critic Agent
------------

The critic is a separate LLM call with a focused adversarial prompt:
"find what's wrong with this code."

It looks for:

1. **Discounting errors** — double discounting, or not discounting
2. **Exercise decision errors** — comparing undiscounted values to call price
3. **Missing vol dependence** — requirements include vol but code doesn't use it
4. **Day count errors** — wrong convention applied
5. **Edge cases** — what happens at maturity, all dates past, deep ITM/OTM

Output: a list of ``CriticConcern(check_id, description, severity, evidence, remediation)``.
The critic does not need to write executable Python on the standard path; it
selects a supported ``check_id`` and explains why that check is warranted.

.. autoclass:: trellis.agent.critic.CriticConcern
   :members:

.. autofunction:: trellis.agent.critic.critique

Arbiter
-------

The arbiter runs all validation checks deterministically:

1. **Invariant suite** — non-negativity, vol monotonicity, bounding, rate sensitivity
2. **Critic-selected deterministic checks** — execute each concern's supported ``check_id``
3. **Report** — ``ValidationResult(passed, invariant_failures, critic_failures)``

No LLM judgment — just run the tests and report pass/fail.

.. autoclass:: trellis.agent.arbiter.ValidationResult
   :members:

Invariant Library
-----------------

.. list-table::
   :header-rows: 1

   * - Check
     - What it catches
   * - ``check_non_negativity``
     - Option price < 0
   * - ``check_vol_monotonicity``
     - Price decreases with vol
   * - ``check_bounded_by_reference``
     - Callable > straight bond (across rates)
   * - ``check_rate_monotonicity``
     - Bond price increases with rates
   * - ``check_zero_vol_intrinsic``
     - Zero-vol ≠ intrinsic value

Case Study: Callable Bond
~~~~~~~~~~~~~~~~~~~~~~~~~

The callable bond demo exposed why the critic matters:

- **First attempt**: agent compared undiscounted cashflows to call price.
  Result: callable > straight bond at high rates (impossible).
- **Bounding invariant** caught the error.
- **Retry**: agent used forward discount factor ratios for the call decision.
  Callable ≤ straight at all rates.

Implementation
--------------

.. autofunction:: trellis.agent.invariants.run_invariant_suite
.. autofunction:: trellis.agent.arbiter.validate
