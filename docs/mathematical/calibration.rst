Calibration Methods
===================

Trellis treats calibration as a market-inference layer inside the pricing
stack. It is not a separate library, and it is not only "fit model parameters
to market quotes." For many liquid products, the authoritative calibrated
result is a Trellis market object such as a curve, vol surface, credit curve,
or correlation surface that downstream pricing should consume directly.
Reduced-form model parameters are a later layer used when Trellis needs model
compression for pricing, simulation, or risk.

Calibration Architecture
------------------------

The calibration sleeve has three Trellis-native layers:

- **Market reconstruction** reconstructs liquid market objects from quoted
  elementary products.
- **Model compression** fits tractable pricing models to those calibrated
  market objects when Trellis needs reusable reduced-form parameter packs.
- **Hybrid composition** combines already-calibrated single-asset objects into
  higher-order or cross-asset systems.

Simple liquid products often use the same pricing engine for both pricing and
calibration: the engine is the translation machine between observed quotes and
the runtime object being reconstructed or fitted. Trellis still treats that as
typed calibration work because it needs explicit quote maps, fitting
instruments, acceptance criteria, runtime materialization, and provenance.

Trellis-Native Runtime Outputs
------------------------------

Calibration outputs land back on ``MarketState`` through
``trellis.models.calibration.materialization`` and related runtime binding
surfaces. That makes calibrated curves, surfaces, credit objects, and model
parameter packs first-class Trellis runtime capabilities instead of route-local
payloads.

The practical split is:

- market reconstruction outputs are often the authoritative runtime result
- model-compression outputs are optional follow-on fits that consume those
  market objects
- hybrid workflows should consume already-materialized single-asset outputs
  rather than bypassing them with one opaque direct fit

Representative Product-Family Mapping
-------------------------------------

.. list-table::
   :header-rows: 1

   * - Product family
     - Liquid inputs
     - Primary calibrated output
     - Optional reduced-model output
   * - OIS / IRS / FX swap
     - deposits, OIS, IRS, FX swaps, basis swaps
     - discount and forecast curves, basis structures
     - short-rate or curve-dynamics factors
   * - Vanilla equity / FX options
     - option prices or quoted vols
     - implied-vol surface or cube
     - local vol, Heston, stochastic-vol parameter packs
   * - Caps/floors / swaptions
     - option prices or quoted vols
     - caplet strips, swaption cubes
     - SABR, Hull-White, G2++, or LMM parameter packs
   * - CDS
     - running and upfront CDS quotes
     - credit curve
     - reduced-form factor model
   * - Basket credit tranches
     - tranche spreads
     - base-correlation or correlation surface
     - copula or factor correlation model
   * - Hybrid liquid sets
     - linked single-asset objects
     - cross-asset binding and correlation objects
     - hybrid state model

Current Bounded Workflow Boundary
---------------------------------

This note documents the currently shipped bounded calibration workflows. It
does not claim that the repo already has a full industrial curve, surface, or
correlation plant for every asset class. Where the current support boundary is
still single-smile, quote-local, or proving-grade, that boundary is stated
explicitly below.

For the latent-state and generator grammar inside the broader sleeve, see
``docs/unified_pricing_engine_model_grammar.md``. For the Trellis implementation
bridge that maps calibration contracts onto runtime bindings, see
``docs/developer/composition_calibration_design.md``.

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

Quote Maps And Target Transforms
--------------------------------

Calibration workflows now use an explicit bounded quote-map surface in
``trellis.models.calibration.quote_maps`` to separate quote conventions from
the stochastic/pricing kernels.

The shipped quote-map vocabulary is intentionally bounded:

- ``Price``
- ``ImpliedVol(Black)``
- ``ImpliedVol(Normal)``
- ``ParRate``
- ``Spread``
- ``Hazard``

Each quote map carries two directional transforms where applicable:

- quote-to-price (for objective-target assembly)
- price-to-quote (for residual reporting in market quote units)

Transform outputs are explicit about failure and warnings, so inverse-transform
issues are surfaced in calibration provenance rather than being hidden inside
route-local helper logic.

Rates calibration workflows now record multi-curve role assumptions directly in
quote-map provenance (discount curve, forecast curve, and rate-index binding),
which keeps quote semantics aligned with OIS discounting and forward-curve
selection used by the pricing kernels.

Typed Materialization Onto ``MarketState``
------------------------------------------

Migrated calibration workflows now materialize runtime outputs through the
bounded helpers in ``trellis.models.calibration.materialization`` instead of
open-coding route-local ``MarketState`` mutation.

The shipped materialization kinds are intentionally narrow:

- ``model_parameter_set``
- ``black_vol_surface``
- ``local_vol_surface``
- ``credit_curve``

Each materialization writes the compatibility fields that existing pricing code
already consumes, but it also records one typed provenance packet under
``market_provenance["calibrated_objects"]`` together with the current selection
under ``market_provenance["selected_calibrated_objects"]``.

That means migrated workflows can preserve:

- the bound runtime object family
- the named runtime object that was selected
- source-kind / source-ref metadata
- multi-curve discount and forecast role selections where applicable
- workflow-specific metadata such as model family or surface shape

``MarketState.materialized_calibrated_object(...)`` is the supported lookup
surface for those authoritative records.

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

SABR fits live in the model-compression layer. The current shipped workflow is
still bounded to one smile at a time rather than a full expiry-tenor surface or
cube reconstruction plant.

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

The supported Heston path is also a model-compression workflow. It fits one
single-expiry smile and then materializes a reusable parameter pack back onto
``MarketState``; it is not yet a full equity-vol surface authority.

The supported Heston workflow fits one single-expiry implied-vol smile onto
the five runtime parameters :math:`(\kappa, \theta, \xi, \rho, v_0)`. Trellis
packages that calibration as an explicit smile surface first:

- ``build_heston_smile_surface(...)`` stores the spot, rate, dividend yield,
  ordered strike/vol points, optional weights, and warning flags
- ``fit_heston_smile_surface(...)`` lowers that smile onto the typed
  ``SolveRequest`` substrate and runs a least-squares fit in implied-vol space
- ``calibrate_heston_smile_workflow(...)`` is the supported raw-input wrapper
  that returns the full ``HestonSmileCalibrationResult`` in one step

The shipped Heston workflow now keeps the carry convention aligned across the
entire priced-to-implied-vol path for this supported slice:

- Black-style implied-vol inversion accepts either ``dividend_yield`` or an
  explicit continuous carry rate
- the Heston fit now passes the smile's ``dividend_yield`` through both FFT
  pricing and implied-vol inversion instead of pricing with carry and inverting
  without it
- fit diagnostics now separate quote-convention mismatches from numerical
  pricing or inversion failures so review tooling can distinguish the two

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

The local-vol path sits on the boundary between market reconstruction and model
compression. In the current shipped workflow it remains a bounded local-vol
extraction layer over a supplied implied-vol grid rather than a full
arbitrage-repaired surface authority.

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

Carry inputs are now recorded explicitly on the local-vol calibration target,
summary, and provenance payloads. For this checked slice the local-vol workflow
still treats those inputs as a continuous-yield carry convention and emits an
explicit warning when such inputs are supplied. That is a bounded support
contract, not yet a full discrete-dividend or repo-specific equity-surface
calibration plant.

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
- Single-name credit replay keeps the same typed CDS-par-spread least-squares
  contract and requires near-zero repricing and quote residuals on the checked
  spread-grid fixture

The benchmark baseline in ``docs/benchmarks/calibration_workflows.{json,md}``
complements those fit-quality gates with cold-start versus warm-start timing
expectations for the supported workflows. The benchmark pack now covers five
workflows (Hull-White, SABR, Heston, local vol, and single-name credit), with
warm-start timing tracked on the three workflows that expose explicit warm
seeds (Hull-White, SABR, Heston).

Those checked fixtures now read from the same bounded mock-snapshot contracts
used by replay and proving paths instead of from a separate benchmark-only
surface. The SABR canary reads the seeded ``usd_rates_smile`` surface and its
rate-vol hints from ``prior_parameters.synthetic_generation_contract``. The
Heston and local-vol canaries read the seeded ``spx_heston_implied_vol``
surface and the derived ``spx_local_vol`` linkage from that same synthetic
contract. The single-name credit canary continues to read spread/recovery
inputs from the derived ``model_consistency_contract`` compatibility packet.

This keeps the checked calibration boundary aligned with the same bounded
synthetic market assumptions that task and proving workflows see at runtime.

Curve Bootstrapping
-------------------

Curve bootstrapping is a market-reconstruction workflow. The calibrated curve
is the authoritative runtime object; later short-rate or curve-dynamics fits
are optional follow-on model-compression steps.

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

The current shipped rates-vol slice is still primarily quote-local. It
normalizes individual cap/floor or swaption quotes and preserves the selected
curve roles, but it does not yet claim a full caplet-strip or swaption-cube
reconstruction plant.

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

The rates Black-vol helpers now use explicit shared term builders for both
supported routes:

- cap/floor calibration resolves one period-term strip
  (:math:`\tau`, fixing time, payment time, discount factor, forward) and
  reuses it for both repricing and result summaries
- swaption calibration resolves the shared expiry/annuity/forward/payment-count
  tuple through the same term-builder used by the analytical swaption route

This keeps route pricing and calibration aligned while still letting the
calibration result report a small numerical residual.

Both helpers also expose the same residual policy fields in
``RatesCalibrationResult.summary``:

- ``residual_tolerance_abs`` gives the route-appropriate absolute PV tolerance
  derived from the solve tolerance and quote scale
- ``residual_within_tolerance`` reports whether the repricing residual is
  within that shared bound

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

The supported Hull-White strip fit is a model-compression workflow layered on
top of rates curve and option calibration inputs. It materializes one reusable
parameter pack, not a full time-dependent rates-model surface.

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

Reduced-Form Credit Calibration
-------------------------------

The supported credit slice is intentionally bounded to single-name reduced-form
hazard calibration for CDS-style quotes. Trellis now exposes
``calibrate_single_name_credit_curve_workflow(...)`` as the typed entry point.

This remains a bounded single-name running-CDS workflow rather than the later
full industrial CDS bootstrap path. The current implementation now calibrates
through a CDS pricing engine, but it still uses one canonical quarterly
``ACT/360`` schedule built from ``market_state.settlement`` with tenor
maturities rounded to calendar months. It does not yet cover the broader
standard-coupon/upfront, IMM-roll, holiday-calendar, or recovery-governance
surface that a production CDS curve plant would need.

The workflow accepts tenor-labeled quotes in either of the shipped credit quote
families:

- ``Spread`` for running-spread quotes, normalized onto decimal running-spread
  fit space
- ``Hazard`` for direct hazard-rate targets, normalized through the same CDS
  pricing stack on a bounded flat-hazard surrogate

The solve objective now fits model-implied CDS par spreads against those
normalized target running spreads rather than solving the identity map in
hazard space. The pricing semantics remain explicit in provenance. The credit
workflow records a potential-binding payload that makes the reduced-form
discount/default contract visible:

.. math::

   D_{\text{risky}}(t) = D(t)\,S(t)

where the selected discount curve and the calibrated credit curve are both
named in the runtime binding metadata.

Like the other migrated calibration helpers, the credit workflow returns:

- the typed ``solve_request`` and normalized ``solve_result``
- governed ``solver_provenance`` and ``solver_replay_artifact``
- quote-map metadata and any inverse-transform warnings
- bounded repricing diagnostics, including target/model running spreads,
  repricing errors, survival probabilities, and forward hazards on the tenor
  grid
- a reusable ``CreditCurve`` together with ``apply_to_market_state(...)`` for
  shared runtime materialization

This slice still does not widen into full schedule-aware CDS bootstrap,
standard-coupon plus upfront calibration, basket credit, structural-credit, or
hybrid credit-equity calibration.

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
.. autofunction:: trellis.models.calibration.credit.calibrate_single_name_credit_curve_workflow
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
.. autoclass:: trellis.models.calibration.credit.CreditHazardCalibrationQuote
   :members:
.. autoclass:: trellis.models.calibration.credit.CreditHazardCalibrationResult
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
