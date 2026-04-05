Calibration Methods
===================

Calibration maps market-observed prices to model parameters.

Solve-Request Substrate
-----------------------

Calibration and inversion helpers now normalize solver inputs onto a typed
``SolveRequest`` carrying:

- named parameters and initial guesses
- explicit lower/upper bounds
- optional constraint metadata and warm-start hints
- an ``ObjectiveBundle`` describing the scalar root or vector least-squares
  target, derivative-hook availability, and replay metadata

The current executor remains SciPy-backed, but the solver-facing contract is
now explicit and serializable before backend dispatch. That means calibration
results can record the solve request itself in provenance instead of forcing
replay tools to infer solver inputs from ad hoc backend calls.

Backend Registry And Capability Policy
--------------------------------------

The solve-request layer now dispatches through ``SolveBackendRegistry`` rather
than assuming one concrete optimizer path in each calibration helper. A backend
record declares:

- which objective shapes it can solve (currently scalar roots and/or vector
  least-squares)
- whether it supports bounds, constraints, warm starts, and derivative hooks
- the executor callable that turns a ``SolveRequest`` into a ``SolveResult``

Capability mismatches are now part of the contract. If a request asks for a
feature that the selected backend does not support, Trellis raises
``UnsupportedSolveCapabilityError`` instead of quietly ignoring that feature.
Fallback is possible, but only when the caller explicitly names a fallback
backend; the resulting solve metadata records both the requested backend and
the backend that actually executed the request.

Implied Volatility
------------------

Given an observed option price :math:`C_{\text{mkt}}`, find :math:`\sigma` such that:

.. math::

   \text{BS}(S, K, T, r, \sigma) = C_{\text{mkt}}

Newton-Raphson Method
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - C_{\text{mkt}}}{\mathcal{V}(\sigma_n)}

where :math:`\mathcal{V} = \partial C / \partial\sigma` is the **vega**:

.. math::

   \mathcal{V} = S\sqrt{T}\,\phi(d_1)

**Initial guess** (Brenner-Subrahmanyam):

.. math::

   \sigma_0 \approx \sqrt{\frac{2\pi}{T}} \cdot \frac{C_{\text{mkt}}}{S}

Falls back to Brent's method for edge cases (deep ITM/OTM).

Jaeckel Rational Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses a rational polynomial approximation for the initial guess, then
Newton refinement. More robust than pure Newton for extreme strikes.

SABR Calibration
----------------

Given market implied vols :math:`\sigma_{\text{mkt}}(K_i)` at strikes
:math:`K_i`, calibrate SABR parameters :math:`(\alpha, \rho, \nu)` with
:math:`\beta` typically fixed:

.. math::

   \min_{\alpha, \rho, \nu} \sum_i \left[\sigma_{\text{SABR}}(K_i; \alpha, \beta, \rho, \nu) - \sigma_{\text{mkt}}(K_i)\right]^2

subject to :math:`\alpha > 0`, :math:`|\rho| < 1`, :math:`\nu > 0`.

Trellis solves this with gradient-assisted L-BFGS-B optimization. The Hagan
approximation is differentiable in the calibrated parameters, so the optimizer
can use exact gradients instead of repeated finite-difference sweeps.
The raw implied-vol kernel stays separate from the optimizer adapter; the
calibration loop consumes the gradient rather than duplicating the pricing math.

The smile-input surface is now explicit as well. ``build_sabr_smile_surface(...)``
packages one ordered strike/vol grid together with stable point labels, optional
weights, and fit warnings such as:

- the forward is not observed exactly, so the nearest strike is used for ATM diagnostics
- the smile does not bracket the forward, so the fit is extrapolative around ATM

``fit_sabr_smile_surface(...)`` then consumes that surface and returns a
reusable calibration artifact containing:

- the assembled smile surface payload
- the typed ``solve_request`` and normalized ``solve_result``
- governed ``solver_provenance`` and ``solver_replay_artifact``
- stable fit diagnostics such as per-strike residuals, max absolute vol error,
  RMS error, weighted RMS error, and ATM error
- explicit warnings that later replay and trader-review workflows can surface

The supported raw-input workflow now sits one level above that substrate.
``calibrate_sabr_smile_workflow(...)`` takes the forward, expiry, strike grid,
market vols, and optional labels/weights, builds the reusable smile surface,
fits the SABR parameters, and returns the full
``SABRSmileCalibrationResult`` in one step. The older ``calibrate_sabr(...)``
helper remains available as a compatibility wrapper that extracts the fitted
``SABRProcess`` while preserving the richer provenance and diagnostic payloads
on the returned object.

The ATM vol provides a good initial guess for :math:`\alpha`:

.. math::

   \alpha_0 \approx \sigma_{\text{ATM}} \cdot F^{1-\beta}

Heston Smile Calibration
------------------------

The supported Heston workflow fits one single-expiry implied-vol smile onto
the five runtime parameters :math:`(\kappa, \theta, \xi, \rho, v_0)`. Trellis
packages that calibration as an explicit smile surface first:

- ``build_heston_smile_surface(...)`` stores the spot, rate, dividend yield,
  ordered strike/vol points, optional weights, and warning flags
- ``fit_heston_smile_surface(...)`` lowers that smile onto the typed
  ``SolveRequest`` substrate and runs a least-squares fit in implied-vol space
- ``calibrate_heston_smile_workflow(...)`` is the supported raw-input wrapper
  that returns the full ``HestonSmileCalibrationResult`` in one step

The Heston workflow keeps the same audit contract as the rates and SABR paths:

- typed ``solve_request`` and normalized ``solve_result``
- governed ``solver_provenance`` and ``solver_replay_artifact``
- stable fit diagnostics including per-strike residuals, RMS errors, and ATM error
- a reusable ``model_parameters`` payload and ``runtime_binding`` that can be
  projected back onto ``MarketState`` for later pricing or simulation

This remains a supported single-smile fit rather than a full term-structure or
surface calibration plant, but the checked workflow is now explicit enough for
replay, runtime reuse, and later performance benchmarking.

Dupire Local Volatility
------------------------

Dupire's formula extracts the local volatility surface from the implied
vol surface :math:`\sigma_{\text{impl}}(K, T)`:

.. math::

   \sigma_{\text{loc}}^2(K, T) = \frac{\sigma^2 + 2\sigma T\!\left(\frac{\partial\sigma}{\partial T} + rK\frac{\partial\sigma}{\partial K}\right)}
   {\left(1 + Kd_1\sqrt{T}\frac{\partial\sigma}{\partial K}\right)^2 + K^2 T\sigma\!\left(\frac{\partial^2\sigma}{\partial K^2} - d_1\sqrt{T}\!\left(\frac{\partial\sigma}{\partial K}\right)^2\right)}

where :math:`d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}`.

The derivatives are computed from a smooth interpolation of the implied
vol surface (bicubic spline).

**Consistency check**: for a flat implied vol surface, :math:`\sigma_{\text{loc}} = \sigma` everywhere.

The hardened checked workflow now exposes that surface as a structured result
instead of only a callable. ``dupire_local_vol_result(...)`` returns the local
volatility callable together with:

- explicit calibration-target metadata for the implied-vol grid
- persisted diagnostics for the Dupire numerator/denominator terms
- a warning list when unstable grid regions were detected
- a compact summary and provenance payload suitable for replay or validation

When the Dupire terms are unstable at a queried point, the compatibility
wrapper still falls back to the interpolated implied volatility there, but that
fallback is no longer silent: the calibration result and callable-facing
artifacts expose the unstable-point count plus sampled problematic regions so
review tooling can flag them.

The supported workflow wrapper now sits one step above that hardened substrate.
``calibrate_local_vol_surface_workflow(...)`` returns the same structured
``LocalVolCalibrationResult`` but tags it as the supported workflow surface and
lets callers apply the calibrated local-vol surface back onto ``MarketState``
without rebuilding an ad hoc wrapper.

Validation And Replay Expectations
----------------------------------

The supported calibration workflows now ship with a deterministic replay and
tolerance pack that locks the current checked quality envelope on the synthetic
benchmark fixtures used for workflow validation:

- Hull-White replay keeps the typed request/provenance contract stable and
  reprices the supported strip with quote residuals below :math:`10^{-8}`
- SABR replay keeps the same solver-provenance contract and stays within a
  max absolute smile-fit error of ``5e-4`` volatility points on the checked
  single-smile fixture
- Heston replay keeps the same typed request/runtime-binding handoff and stays
  within a max absolute smile-fit error of ``1e-4`` volatility points on the
  checked single-smile fixture
- Local-vol replay keeps the workflow provenance contract stable and requires
  zero unstable calibration-grid points on the checked stable surface fixture

The benchmark baseline in ``docs/benchmarks/calibration_workflows.{json,md}``
complements those fit-quality gates with cold-start versus warm-start timing
expectations for the supported workflows.

Curve Bootstrapping
-------------------

Calibrate a zero-rate curve from market instruments (deposits, futures, swaps).

The bootstrap input surface is now explicit:

- ``BootstrapCurveInputBundle`` names the curve and carries currency,
  rate-index, metadata, and the ordered market instruments
- ``BootstrapConventionBundle`` carries the supported convention inputs for the
  current bootstrap path: simple deposit compounding, future quote style, and
  fixed/float leg frequencies plus day-count labels for swaps
- ``BootstrapInstrument`` carries the market quote together with tenor,
  optional start tenor, optional accrual override, and a stable instrument label

The current implementation still works on year-fraction tenors rather than
full dated schedules, so the convention bundle governs the implied accrual grid
used by the bootstrap rather than a full calendar roll engine.

The bootstrap now lowers onto the same typed solve-request substrate used by
the other calibration helpers. ``build_bootstrap_solve_request(...)`` packages
the ordered curve bundle as an instrument-labeled vector least-squares problem:

.. math::

   \min_{\mathbf{r}} \left\|\text{model}(\mathbf{r}) - \text{quotes}\right\|_2^2

The repricer is still assembled from differentiable primitives, so Trellis can
build the bootstrap residual Jacobian
:math:`J = \partial\,\text{model} / \partial \mathbf{r}` directly from the
typed curve bundle instead of reintroducing finite-difference sweeps. When the
executor sees that full residual Jacobian, the SciPy backend uses a vector
least-squares solve path rather than scalarizing the bootstrap into an opaque
optimizer call.

``bootstrap_curve_result(...)`` then exposes the solved curve together with the
typed ``solve_request``, normalized ``solve_result``, governed
``solver_provenance``, replay artifact, and per-run diagnostics such as the
residual vector, Jacobian matrix, condition number, and Jacobian rank. The
legacy ``bootstrap(...)`` and ``bootstrap_yield_curve(...)`` helpers remain as
thin compatibility wrappers over that richer result surface.

Rates Option Calibration
------------------------

Cap/floor and European swaption quotes are often calibrated as implied Black
volatilities under a multi-curve environment. Trellis keeps the calibration
surface explicit: the result preserves the selected curve names from
``MarketState`` and records any caller-supplied volatility or correlation
source labels.

Given an observed price :math:`P_{\text{mkt}}`, the rates calibration helpers
solve:

.. math::

   P_{\text{cap/floor}}(\sigma) = P_{\text{mkt}}
   \qquad\text{or}\qquad
   P_{\text{swaption}}(\sigma) = P_{\text{mkt}}

where the pricing side uses the same OIS discount and forecast-curve selection
as the underlying route. For cap/floor workflows, the price is the discounted
sum of Black76 caplets or floorlets. For European swaptions, the price is:

.. math::

   PV = N \cdot A \cdot \text{Black76}(S, K, \sigma, T)

with :math:`S` the forward swap rate and :math:`A` the annuity assembled from
the selected discount and forecast curves.

Calibration regression checks should treat the recovered volatility and the
reported residual as the primary success criteria. Small PV drift at the level
of the root-finding tolerance is expected, so route-appropriate tolerances are
preferred over machine-epsilon equality on the repriced present value.

The swaption payoff route and the swaption calibration helper share the same
term builder for expiry, annuity, forward swap rate, and payment count. That
shared algebra keeps route pricing and calibration aligned while still letting
the calibration result report a small numerical residual.

Both the rates helpers and the SABR least-squares helper now record their typed
solve-request payloads in provenance. The cap/floor and swaption workflows use
scalar root solves over price residuals, while the rates bootstrap and SABR
workflows package their fits as vector objectives with stable labels and
optional derivative hooks.

Those helpers also normalize the solver execution details into two governed
artifacts:

- ``solver_provenance`` records backend identity, requested options,
  termination summary, and residual diagnostics
- ``solver_replay_artifact`` packages the typed solve request together with the
  normalized provenance and the raw solve result so replay and review flows can
  inspect one deterministic solve path without reverse-engineering model-local
  payloads

Hull-White Strip Calibration
----------------------------

The supported Hull-White workflow now calibrates one reusable
``(mean_reversion, sigma)`` parameter pair to a strip of swaption-style quotes
instead of requiring helper routes to hard-code ``mean_reversion = 0.1``.

Given instrument labels :math:`i = 1,\dots,n`, market targets
:math:`Q_i^{\text{mkt}}`, and tree prices :math:`P_i^{\text{HW}}(a,\sigma)`,
Trellis solves a weighted least-squares problem on the typed solve-request
substrate:

.. math::

   \min_{a,\sigma} \sum_i w_i \left(P_i^{\text{HW}}(a,\sigma) - P_i^{\text{mkt}}\right)^2

The current supported strip uses one-exercise Bermudan swaption contracts as
the calibration instrument surface so the calibration route and the callable
rates helpers share the same underlying Hull-White lattice machinery.

The workflow accepts either:

- direct price quotes, or
- Black-vol quotes, which are first normalized onto target prices and then
  converted back into quote units for residual reporting

Like the bootstrap workflow, ``calibrate_hull_white(...)`` records:

- the typed ``solve_request`` and normalized ``solve_result``
- governed ``solver_provenance`` and ``solver_replay_artifact``
- the reused market ``bootstrap_runs`` provenance when the input curves were
  themselves bootstrapped
- a serializable ``model_parameters`` payload that can be attached back to
  ``MarketState``

``HullWhiteCalibrationResult.apply_to_market_state(...)`` writes the calibrated
payload onto both ``market_state.model_parameters`` and
``market_state.model_parameter_sets[parameter_set_name]``. The callable-bond,
Bermudan-swaption, and zero-coupon-bond-option helpers resolve those attached
parameters before falling back to legacy heuristics.

This remains a constant-parameter Hull-White fit across the supported strip,
not a full time-dependent :math:`\sigma(t)` calibration.

Implementation
--------------

.. autofunction:: trellis.models.calibration.implied_vol.implied_vol
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr_smile_workflow
.. autofunction:: trellis.models.calibration.sabr_fit.build_sabr_smile_surface
.. autofunction:: trellis.models.calibration.sabr_fit.fit_sabr_smile_surface
.. autofunction:: trellis.models.calibration.heston_fit.calibrate_heston_smile_workflow
.. autofunction:: trellis.models.calibration.heston_fit.build_heston_smile_surface
.. autofunction:: trellis.models.calibration.heston_fit.fit_heston_smile_surface
.. autofunction:: trellis.models.calibration.local_vol.calibrate_local_vol_surface_workflow
.. autofunction:: trellis.models.calibration.local_vol.dupire_local_vol_result
.. autofunction:: trellis.models.calibration.local_vol.dupire_local_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_cap_floor_black_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_swaption_black_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_hull_white
.. autoclass:: trellis.models.calibration.heston_fit.HestonSmileSurface
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSmileFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSmileCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.sabr_fit.SABRSmileSurface
   :members:
.. autoclass:: trellis.models.calibration.sabr_fit.SABRSmileFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.sabr_fit.SABRSmileCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.local_vol.LocalVolCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveRequest
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveBackendRegistry
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveProvenance
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveReplayArtifact
   :members:
.. autofunction:: trellis.curves.bootstrap.bootstrap_yield_curve
   :no-index:

References
----------

- Jaeckel, P. (2015). "Let's be rational." *Wilmott*, 2015(75), 40-53.
- Hagan, P. et al. (2002). "Managing smile risk." *Wilmott Magazine*, Sep 2002.
- Dupire, B. (1994). "Pricing with a smile." *Risk*, 7(1), 18-20.
