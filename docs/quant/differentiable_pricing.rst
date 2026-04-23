Differentiable Pricing
======================

Trellis promotes autograd only where it has a clear payoff:

- closed-form pricing kernels that currently drive real Greeks and calibration
- cap/floor strips and FX/quanto analytics built on top of those kernels
- flat-vol Vega extraction in ``Session.analyze()``
- SABR calibration, where a gradient is more useful than repeated finite-difference sweeps
- binomial/trinomial tree pricing for smooth payoffs when the tree state is built
  from autograd-aware inputs
- Monte Carlo pathwise pricing when shocks are supplied explicitly and the
  payoff stays on the smooth side of the kink

FX vanilla under Garman-Kohlhagen is the clearest example of the basis
assembly engine in action: spot FX and domestic/foreign discount factors are
bridged to a forward, the option is valued from Black76 asset-or-nothing and
cash-or-nothing basis claims, and the terminal vanilla payoff is assembled with
``terminal_vanilla_from_basis(...)``. The checked helper-facing entrypoint for
that route is now ``trellis.models.analytical.fx.garman_kohlhagen_price_raw(...)``,
so the decomposition stays explicit without asking every adapter to rebuild the
same payoff algebra inline.

The reusable closed-form support surface now lives under
``trellis.models.analytical.support``. Raw analytical helpers such as
``trellis.models.analytical.fx.garman_kohlhagen_price_raw``,
``trellis.models.analytical.quanto.price_quanto_option_raw``,
``trellis.models.analytical.jamshidian.zcb_option_hw_raw``,
``trellis.models.rate_style_swaption.price_swaption_black76_raw``, and the
barrier route kernels compose that support layer rather than re-implementing
the same discounting, forward, and terminal-payoff glue in each route.

For quanto, the raw kernel is intentionally paired with a stricter runtime
resolver boundary: ``resolve_quanto_inputs(...)`` now records per-input
provenance for the underlier spot, FX spot, domestic curve, foreign curve,
volatility lookups, and correlation, and it only accepts noncanonical foreign
carry reuse when an explicit ``quanto_foreign_curve_policy`` bridge is present.

The goal is not to make every numerical routine differentiable. It is to remove
unnecessary bump/reprice loops, stabilize calibration, and keep the exact same
pricing logic available to both value and sensitivity workflows.

Where Autograd Helps
--------------------

- Black76 and Garman-Kohlhagen calls/puts
- FX vanilla pricing can be assembled explicitly from Black76 terminal basis
  claims via ``terminal_vanilla_from_basis(...)`` after mapping spot FX and
  domestic/foreign discount factors to a forward.
- cap and floor strip valuation
- quanto adjustment and resolved quanto pricing inputs
- the T09 barrier kernel pack for the down-and-out / down-and-in call route
- raw resolved-input analytical helpers such as
  ``trellis.models.analytical.fx.garman_kohlhagen_price_raw``,
  ``trellis.models.analytical.jamshidian.zcb_option_hw_raw``,
  ``trellis.models.rate_style_swaption.price_swaption_black76_raw``,
  ``trellis.models.analytical.quanto.price_quanto_option_raw``, and
  ``trellis.models.analytical.barrier.down_and_out_call_raw`` /
  ``trellis.models.analytical.barrier.down_and_in_call_raw``
- public ``YieldCurve`` / ``CreditCurve`` node-value sensitivities and
  ``GridVolSurface`` node-value sensitivities
- runtime rate-risk extraction on public ``YieldCurve`` node grids, with
  ``resolved_derivative_method="autodiff_public_curve"`` recorded on the
  resulting analytics outputs
- flat-vol Vega extraction in the analytics layer
- SABR calibration through a gradient-assisted objective
- simple binomial/trinomial tree rollback through ``backward_induction(..., differentiable=True)``
- pathwise Monte Carlo pricing through ``simulate_with_shocks(..., differentiable=True)``

These paths now use autograd-friendly primitives and avoid scalarization inside
the traced region.

Where Trellis Still Stays Forward-Only
--------------------------------------

- generic lattice calibration and reduced-storage Monte Carlo path-state accumulation
- Numba-accelerated tree, Monte Carlo, and PDE kernels
- discontinuous payoffs that would need smoothing or a custom adjoint
- broader European barrier families beyond the T09 route, which remain
  forward-only until a second consumer justifies shared barrier support
- scalar vega on unsupported smile surfaces, which now reports an explicit
  representative-flat-vol fallback instead of silently pretending to be a
  surface-native Greek
- reduced-storage state-aware Monte Carlo payoffs, which still coerce results
  back to plain ``float`` arrays
- custom discretization schemes that are not autograd-aware themselves

That split is deliberate: the compiled engines stay fast for production pricing,
while the closed-form layer exposes gradients where they are genuinely useful.

Implementation Rules
--------------------

- use ``autograd.numpy`` via ``trellis.core.differentiable.get_numpy()``
- avoid ``float(...)`` or other scalarization inside traced pricing code
- for Monte Carlo gradients, pass explicit shocks so the path is deterministic
- keep the public pricing adapter trace-safe and reserve ``float(...)`` for
  explicit reporting or solver boundaries; expose raw resolved-input kernels as
  ``*_raw`` helpers when they improve reuse
- keep generator prompts, cookbook templates, and payoff skeletons aligned to
  that same public contract so learned routes preserve traced PVs by default
- for piecewise-linear curves and surfaces, target node-value derivatives as
  the supported contract and treat query-location derivatives as piecewise only
  away from knot boundaries
- keep Numba kernels as forward engines, not gradient engines
- preserve the full ``MarketState`` when cloning a traced pricing state
- prefer autodiff when it replaces bump/reprice loops, parallel DV01s, or
  optimizer Jacobians
- when a runtime measure falls back to bumps, record that derivative-method
  choice in the public result metadata instead of hiding it behind a plain
  scalar

Implementation References
-------------------------

.. autofunction:: trellis.models.black.black76_call
.. autofunction:: trellis.models.black.garman_kohlhagen_call
.. autofunction:: trellis.models.analytical.fx.garman_kohlhagen_price_raw
.. autofunction:: trellis.models.analytical.terminal_vanilla_from_basis
.. autofunction:: trellis.models.analytical.jamshidian.zcb_option_hw_raw
.. autofunction:: trellis.models.rate_style_swaption.price_swaption_black76_raw
.. autofunction:: trellis.models.analytical.quanto.price_quanto_option_raw
.. autofunction:: trellis.models.analytical.barrier.down_and_out_call_raw
.. autofunction:: trellis.models.analytical.barrier.down_and_in_call_raw
.. automethod:: trellis.instruments.cap.CapPayoff.evaluate
.. automethod:: trellis.analytics.measures.Vega.compute
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr
.. autofunction:: trellis.models.trees.backward_induction.backward_induction
.. automethod:: trellis.models.monte_carlo.engine.MonteCarloEngine.simulate_with_shocks
.. automethod:: trellis.models.monte_carlo.engine.MonteCarloEngine.price

Related Reading
---------------

- :doc:`../mathematical/black76`
- :doc:`../mathematical/calibration`
- :doc:`extending_trellis`
- :doc:`pricing_stack`
