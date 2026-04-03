Analytical Route Cookbook
=========================

This page shows how to add a new closed-form analytical route to Trellis.
All analytical routes follow the same four-layer pattern:

.. code-block:: text

   resolver → support helpers → raw kernel → public adapter

The pattern keeps autograd-safe computation in the raw kernel, market-data
handling in the adapter, and reusable mathematical building blocks in the
shared support layer under ``trellis.models.analytical.support``.


Layer Summary
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Layer
     - What it does
     - Example
     - Location
   * - ``ResolvedXxxInputs``
     - Frozen dataclass of pre-computed scalar market inputs
     - ``ResolvedBarrierInputs(spot, strike, barrier, rate, sigma, T)``
     - Top of the route module
   * - Support helpers
     - Reusable mathematical building blocks (discounting, forwards, payoff decomposition)
     - ``discounted_value``, ``forward_from_discount_factors``, ``terminal_vanilla_from_basis``
     - ``trellis.models.analytical.support``
   * - ``xxx_raw(resolved)``
     - Autograd-safe pricing kernel that accepts resolved inputs plus, at most, a small semantic selector
     - ``down_and_out_call_raw``, ``garman_kohlhagen_price_raw``
     - Route module, private or exported
   * - ``xxx(S, K, …)`` public adapter
     - Accepts raw market inputs, constructs ``ResolvedXxxInputs``, calls the raw kernel
     - ``down_and_out_call``, ``black76_call``
     - Route module, public


Minimal Example
---------------

Here is a stripped-down template for a new analytical route:

.. code-block:: python

    # trellis/models/analytical/my_product.py
    from __future__ import annotations

    from dataclasses import dataclass

    from trellis.models.analytical.support import (
        discount_factor_from_zero_rate,
        discounted_value,
        forward_from_carry_rate,
        safe_time_fraction,
        terminal_intrinsic,
    )


    @dataclass(frozen=True)
    class ResolvedMyProductInputs:
        """Pre-computed scalar inputs for MyProduct pricing."""
        option_type: str
        spot: float
        strike: float
        domestic_rate: float
        carry_rate: float
        T: float


    def my_product_raw(resolved: ResolvedMyProductInputs) -> float:
        """Autograd-safe pricing kernel over resolved inputs only."""
        T = safe_time_fraction(resolved.T)
        forward = forward_from_carry_rate(
            spot=resolved.spot,
            carry_rate=resolved.carry_rate,
            T=T,
        )
        discount_factor = discount_factor_from_zero_rate(
            resolved.domestic_rate,
            T,
        )
        intrinsic = terminal_intrinsic(
            resolved.option_type,
            spot=forward,
            strike=resolved.strike,
        )
        return discounted_value(intrinsic, discount_factor)


    def my_product(
        option_type: str,
        S: float,
        K: float,
        domestic_rate: float,
        carry_rate: float,
        T: float,
    ) -> float:
        """Price MyProduct. Public adapter — handles market inputs."""
        if T <= 0.0:
            return float(terminal_intrinsic(option_type, spot=S, strike=K))
        resolved = ResolvedMyProductInputs(
            option_type=option_type,
            spot=S,
            strike=K,
            domestic_rate=domestic_rate,
            carry_rate=carry_rate,
            T=T,
        )
        return float(my_product_raw(resolved))


Rules for Raw Kernels
---------------------

1. Use autograd-safe numerics inside the raw kernel. That can mean composing
   the checked support helpers or calling ``trellis.core.differentiable.get_numpy()``.
   Never switch to plain ``numpy`` inside traced pricing code.

2. Accept a resolved dataclass plus, at most, a small fixed semantic selector
   such as ``option_type``. No ``MarketState``, no schedule parsing, and no
   open-ended ``**kwargs`` plumbing.

3. Keep the raw kernel trace-friendly. Do not wrap the final value in
   ``float(...)`` inside the traced region; let the public adapter cast if it
   needs a plain Python float.

4. Keep market resolution, date parsing, and hard route selection outside the
   raw kernel. Small semantic branches such as call vs put are fine; traced
   numerical regime handling should prefer array-safe control such as
   ``np.where(...)`` when the branch must remain differentiable.


Shared Support Helpers
----------------------

The support layer provides the standard building blocks:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Helper
     - Purpose
     - Module
   * - ``discount_factor_from_zero_rate(rate, T)``
     - Continuous-compounding discount factor implied by a zero rate
     - ``support.discounting``
   * - ``discounted_value(value, discount_factor, scale=...)``
     - Apply discounting and optional notional scaling to an undiscounted value
     - ``support.discounting``
   * - ``safe_time_fraction(T)``
     - Clamp a model horizon to a non-negative analytical time fraction
     - ``support.discounting``
   * - ``forward_from_carry_rate(spot, carry_rate, T)``
     - Forward implied by a continuous carry rate over ``T``
     - ``support.forwards``
   * - ``forward_from_discount_factors(spot, domestic_df, foreign_df)``
     - Forward bridge from spot and domestic/foreign discount factors
     - ``support.forwards``
   * - ``forward_from_dividend_yield(spot, domestic_rate, dividend_yield, T)``
     - Standard equity forward under continuous dividend yield
     - ``support.forwards``
   * - ``terminal_intrinsic(option_type, spot=..., strike=...)``
     - Terminal intrinsic value for a vanilla call or put
     - ``support.payoffs``
   * - ``cash_or_nothing_intrinsic(option_type, spot=..., strike=..., cash=...)``
     - Terminal cash-or-nothing payoff for a vanilla call or put
     - ``support.payoffs``
   * - ``asset_or_nothing_intrinsic(option_type, spot=..., strike=...)``
     - Terminal asset-or-nothing payoff for a vanilla call or put
     - ``support.payoffs``
   * - ``terminal_vanilla_from_basis(option_type, asset_value=..., cash_value=..., strike=...)``
     - Assemble a vanilla payoff from exact basis claims
     - ``support.payoffs``
   * - ``foreign_to_domestic_forward_bridge(spot, domestic_df, foreign_df)``
     - Bridge foreign-carry spot into a domestic forward level
     - ``support.cross_asset``
   * - ``quanto_adjusted_forward(spot, domestic_df, foreign_df, corr, sigma_underlier, sigma_fx, T)``
     - Domestic-payout quanto forward after covariance adjustment
     - ``support.cross_asset``
   * - ``exchange_option_effective_vol(sigma_1, sigma_2, corr)``
     - Margrabe-style effective volatility for exchange-style payoffs
     - ``support.cross_asset``


Checked-In Route Examples
-------------------------

Recent analytical routes in Trellis follow the same resolver-to-raw split:

- FX vanilla: ``ResolvedGarmanKohlhagenInputs`` ->
  ``garman_kohlhagen_price_raw(...)`` in
  ``trellis.models.analytical.fx``.
- Quanto vanilla: ``resolve_quanto_inputs(...)`` ->
  ``ResolvedQuantoInputs`` -> ``price_quanto_option_raw(...)`` in
  ``trellis.models.resolution.quanto`` and
  ``trellis.models.analytical.quanto``. The resolver accepts canonical foreign
  carry keys such as ``EUR-DISC`` directly, but any noncanonical foreign-curve
  reuse now requires an explicit ``quanto_foreign_curve_policy`` bridge instead
  of silently falling back to the only forecast curve or the domestic discount
  curve.
- European rate-style swaption: ``resolve_swaption_black76_inputs(...)`` ->
  ``ResolvedSwaptionBlack76Inputs`` ->
  ``price_swaption_black76_raw(...)`` in
  ``trellis.models.rate_style_swaption``.
- Jamshidian zero-coupon bond option:
  ``resolve_zcb_option_hw_inputs(...)`` ->
  ``ResolvedJamshidianInputs`` / ``zcb_option_hw_raw(...)`` under the public
  wrapper ``price_zcb_option_jamshidian(...)`` in
  ``trellis.models.zcb_option`` and
  ``trellis.models.analytical.jamshidian``.

When a checked-in helper already owns market binding plus a raw kernel, agent
generated adapters should delegate to that helper-backed surface instead of
reconstructing annuity, forward, or strike-normalization logic inline.


Gradient Verification
---------------------

Every new raw kernel should have an autograd gradient test alongside the
pricing test:

.. code-block:: python

    from trellis.core.differentiable import gradient
    from dataclasses import replace

    def test_my_product_vega_autograd(resolved):
        autodiff = gradient(
            lambda vol: my_product_raw(replace(resolved, sigma=vol))
        )(resolved.sigma)
        fd = (my_product_raw(replace(resolved, sigma=resolved.sigma + 1e-6))
              - my_product_raw(replace(resolved, sigma=resolved.sigma - 1e-6))) / 2e-6
        assert autodiff == pytest.approx(fd, rel=1e-6, abs=1e-8)

See ``tests/test_models/test_analytical_support.py`` and
``tests/test_models/test_fx_analytical_support.py`` for checked-in examples.


Barrier Monitoring
------------------

Barrier primitives (survival probability, rebate payment) are **route-local**
until two independent routes share the same formula verbatim.

The current barrier implementation in ``trellis.models.analytical.barrier``
provides:

- ``vanilla_call_raw``, ``barrier_image_raw``, ``rebate_raw`` — raw-kernel pack
- ``down_and_out_call_raw``, ``down_and_in_call_raw`` — assembled route kernels
- ``barrier_option_price(S, K, B, r, sigma, T, barrier_type, option_type, rebate)``
  — generic Reiner-Rubinstein dispatcher (all 4 barrier types × 2 option types)
- ``down_and_out_call``, ``down_and_in_call`` — public adapters for the T09 route

``trellis.models.analytical.support.barriers`` is the designated home for
shared barrier helpers once a second consumer appears.  See
``docs/quant/basis_claim_patterns.md`` §Barrier Monitoring for the extraction
policy and checklist.


Agent Integration
-----------------

When the agent builds a new analytical route it should:

1. Import reusable helpers from ``trellis.models.analytical.support`` rather
   than reimplementing basic discounting or payoff decomposition.
2. Place the raw kernel in a file under ``trellis/models/analytical/`` or in
   the agent-generated instruments directory (``trellis/instruments/_agent/``).
3. Register the route in ``trellis/agent/knowledge/canonical/routes.yaml`` with
   the appropriate ``family``, ``method``, and ``score_hints`` block.

The builder-facing cookbook entries live in
``trellis/agent/knowledge/canonical/cookbooks.yaml`` and are rendered through
the agent prompt layer. Keep those cookbook entries aligned with the checked-in
helper surface whenever a new analytical route lands.

Related Reading
---------------

- :doc:`pricing_stack` — how the analytical layer fits into the full pricing stack
- :doc:`differentiable_pricing` — autograd rules and helper reference
- :doc:`basis_claim_patterns` — terminal, barrier, and path-dependent extraction policy
- :doc:`extending_trellis` — broader guide for adding instruments and routes
