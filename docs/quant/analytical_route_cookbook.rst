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
     - ``terminal_intrinsic``, ``df_continuous``, ``forward_from_spot``
     - ``trellis.models.analytical.support``
   * - ``xxx_raw(resolved)``
     - Autograd-safe pricing kernel that accepts only the resolved dataclass
     - ``down_and_out_call_raw``, ``black76_call_raw``
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

    import autograd.numpy as np

    from trellis.models.analytical.support import (
        df_continuous,
        forward_from_spot,
        terminal_intrinsic,
    )


    @dataclass(frozen=True)
    class ResolvedMyProductInputs:
        """Pre-computed scalar inputs for MyProduct pricing."""
        spot: float
        strike: float
        rate: float
        sigma: float
        T: float


    def my_product_raw(resolved: ResolvedMyProductInputs) -> float:
        """Autograd-safe pricing kernel. Only accepts the resolved dataclass."""
        fwd = forward_from_spot(resolved.spot, resolved.rate, resolved.T)
        df = df_continuous(resolved.rate, resolved.T)
        intrinsic = terminal_intrinsic(fwd, resolved.strike)
        # … add model-specific terms …
        return float(df * intrinsic)


    def my_product(
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
    ) -> float:
        """Price MyProduct. Public adapter — handles market inputs."""
        if T <= 0:
            return max(S - K, 0.0)
        resolved = ResolvedMyProductInputs(spot=S, strike=K, rate=r, sigma=sigma, T=T)
        return my_product_raw(resolved)


Rules for Raw Kernels
---------------------

1. **Only** ``autograd.numpy`` (aliased as ``np``) inside the raw kernel.
   Never use plain ``numpy`` — it breaks the gradient tape.

2. Accept exactly **one** argument: the resolved dataclass.  No ``**kwargs``,
   no market-state threading.

3. Return a plain Python ``float``.  Use ``float(...)`` on the final result.

4. No branching on input values inside the raw kernel (e.g. ``if S <= B``).
   Move branching to the public adapter **before** constructing the resolved
   dataclass.  Branching breaks autograd through the kernel.


Shared Support Helpers
----------------------

The support layer provides the standard building blocks:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Helper
     - Purpose
     - Module
   * - ``df_continuous(r, T)``
     - Continuous-compounding discount factor ``exp(-rT)``
     - ``support.discounting``
   * - ``df_from_rate(rate, T, convention)``
     - Discount factor from a rate under a named day-count convention
     - ``support.discounting``
   * - ``safe_sqrt_T(T)``
     - ``sqrt(T)`` clamped to a small floor to avoid ``0/0``
     - ``support.discounting``
   * - ``forward_from_spot(S, r, T)``
     - Forward price ``S * exp(rT)`` (no dividends)
     - ``support.forwards``
   * - ``forward_from_discount_factors(S, df_r, df_q)``
     - Forward price given separate risk-free and dividend discount factors
     - ``support.forwards``
   * - ``terminal_intrinsic(F, K)``
     - ``max(F - K, 0)`` — vanilla call terminal payoff from forward
     - ``support.payoffs``
   * - ``cash_or_nothing_intrinsic(F, K)``
     - ``1 if F > K else 0`` — digital cash payoff from forward
     - ``support.payoffs``
   * - ``asset_or_nothing_intrinsic(F, K)``
     - ``F if F > K else 0`` — digital asset payoff from forward
     - ``support.payoffs``
   * - ``terminal_vanilla_from_basis(asset_or_nothing, cash_or_nothing, K)``
     - Assembles vanilla payoff from basis claims
     - ``support.payoffs``
   * - ``quanto_adjusted_forward(S, r_d, r_f, rho, sigma_S, sigma_FX, T)``
     - Forward under quanto measure adjustment
     - ``support.cross_asset``
   * - ``effective_vol_cross_asset(sigma_S, sigma_FX, rho)``
     - Combined volatility for cross-currency products
     - ``support.cross_asset``


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

See ``tests/test_models/test_analytical_support.py`` for full examples.


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

The builder cookbook ``rate_tree``, ``monte_carlo``, ``analytical``, and
other templates in ``trellis/agent/cookbooks.py`` show the full
``evaluate()`` body structure expected by the agent validator.

Related Reading
---------------

- :doc:`pricing_stack` — how the analytical layer fits into the full pricing stack
- :doc:`differentiable_pricing` — autograd rules and helper reference
- :doc:`basis_claim_patterns` — terminal, barrier, and path-dependent extraction policy
- :doc:`extending_trellis` — broader guide for adding instruments and routes
