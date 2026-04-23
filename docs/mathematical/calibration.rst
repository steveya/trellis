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

Bounded Hybrid Quanto Correlation Slice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first checked hybrid calibration workflow is intentionally narrow:
``calibrate_quanto_correlation_workflow(...)`` fits one scalar
``quanto_correlation`` from one or more quanto option price quotes.

The workflow composes already-bound runtime market inputs rather than building
a new cross-asset plant. It requires the existing quanto resolver to consume:

- a domestic discount curve
- a canonical or explicitly bridged foreign carry curve
- underlier and FX spots
- a volatility surface for the underlier / FX-vol lookups
- market quotes for the bounded quanto option target

The workflow records a ``CalibrationDependencyGraph`` with dependency-first
ordering, solves the scalar correlation through the shared ``SolveRequest``
surface, reports quote-level repricing residuals, and materializes the
calibrated value back onto ``MarketState`` as a ``model_parameter_set`` named
by the caller. Downstream quanto pricing then consumes the same
``market_state.model_parameters["quanto_correlation"]`` binding used by direct,
empirical, or user-supplied correlation inputs.

This is not a universal rates/equity/FX calibration engine. It does not
calibrate cross-currency curves, separate FX-vol surfaces, multi-underlier
quanto baskets, or a globally smoothed hybrid state model.

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
- ``correlation_surface``

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
``MarketState``; it is not itself the authoritative equity-vol market object.

Repaired Equity-Vol Surface Authority
-------------------------------------

The first bounded market-reconstruction lane for equity vol now sits one layer
above the old single-smile Heston and raw Dupire workflows. Trellis exposes
``calibrate_equity_vol_surface_workflow(...)`` as a repaired multi-expiry
equity-vol surface authority.

The current shipped authority is intentionally bounded:

- raw observed quotes first pass through
  ``clean_equity_vol_surface_quotes(...)``, which keeps raw-versus-cleaned
  provenance explicit and applies a local outlier-governance pass on the
  observed grid before model repair
- each expiry smile is fit with a raw-SVI parameterization in total-variance
  space
- the fit uses explicit positive-residual penalties for smile-level
  positivity, call-price monotonicity, and call-price convexity on a
  diagnostic strike grid
- the assembled surface applies a calendar-total-variance repair across
  expiries before answering ``black_vol(expiry, strike)`` queries

That gives Trellis a first market-object-first equity-vol surface that can be
materialized back onto ``MarketState`` as a ``black_vol_surface`` instead of
jumping directly to a reduced-model parameter pack.

The workflow returns:

- the raw multi-expiry input surface
- the cleaned multi-expiry input surface plus node-level quote-cleaning
  diagnostics and adjustments
- one typed solve request and replay artifact per fitted SVI smile
- smile-level no-arbitrage diagnostics before and after repair
- surface-level calendar diagnostics before and after repair
- the repaired implied-vol surface itself for downstream pricing or later
  model-compression stages

This remains a bounded first surface authority rather than a full production
SPX plant. The checked implementation now has an explicit quote-governance
stage, but it is still raw-SVI-based, not yet a full SSVI, bid/ask, liquidity,
or exchange-convention governance stack.

The supported Heston workflow now comes in two bounded forms:

- ``calibrate_heston_smile_workflow(...)`` fits one single-expiry implied-vol
  smile onto the five runtime parameters :math:`(\kappa, \theta, \xi, \rho, v_0)`
- ``calibrate_heston_surface_from_equity_vol_surface_workflow(...)`` compresses
  the repaired multi-expiry equity-vol authority into one reusable Heston
  parameter pack across the full fitted grid

Trellis packages that calibration as explicit smile or surface targets first:

- ``build_heston_smile_surface(...)`` stores the spot, rate, dividend yield,
  ordered strike/vol points, optional weights, and warning flags
- ``fit_heston_smile_surface(...)`` lowers that smile onto the typed
  ``SolveRequest`` substrate and runs a least-squares fit in implied-vol space
- ``calibrate_heston_smile_workflow(...)`` is the supported raw-input wrapper
  that returns the full ``HestonSmileCalibrationResult`` in one step
- ``build_heston_surface_input(...)`` stores the full ordered expiry/strike
  grid for a multi-expiry fit
- ``fit_heston_surface(...)`` lowers the full grid onto one typed
  ``SolveRequest`` and calibrates one global Heston parameter pack across the
  whole surface
- ``calibrate_heston_surface_workflow(...)`` is the supported raw-input wrapper
  for that full-grid fit
- ``calibrate_heston_surface_from_equity_vol_surface_workflow(...)`` is the
  staged model-compression wrapper that consumes the repaired surface
  authority rather than raw quotes directly

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
- stable fit diagnostics including per-strike or per-expiry residuals, RMS
  errors, and ATM error
- a reusable ``model_parameters`` payload and ``runtime_binding`` that can be
  projected back onto ``MarketState`` for later pricing or simulation

Trellis now also exposes
``compare_heston_to_equity_vol_surface_workflow(...)`` for one bounded staged
comparison: the repaired equity-vol surface can be compared directly against a
later Heston compression fit on a selected expiry slice so the market-object
stage and the reduced-model stage are not conflated.

Trellis also exposes
``compare_heston_surface_to_equity_vol_surface_workflow(...)`` for the same
stage separation on the full observed expiry/strike grid.

This remains a bounded global-surface compression fit rather than a full
time-dependent Heston term structure, stochastic-local-vol bridge, or broader
stochastic-vol plant, but the checked workflow is now explicit enough for
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

The local-vol lane can now also consume the repaired equity-vol authority
instead of raw implied-vol quotes directly.
``calibrate_local_vol_surface_from_equity_vol_surface_workflow(...)`` samples
the repaired surface on a chosen grid and then runs the same bounded Dupire
workflow from that repaired implied-vol surface.

Carry inputs are now recorded explicitly on the local-vol calibration target,
summary, and provenance payloads. For this checked slice the local-vol workflow
still treats those inputs as a continuous-yield carry convention and emits an
explicit warning when such inputs are supplied. That is a bounded support
contract, not yet a full discrete-dividend or repo-specific equity-surface
calibration plant.

Validation And Replay Expectations
----------------------------------

Most supported calibration workflows now ship with a deterministic replay and
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
- Basket-credit replay consumes the materialized single-name credit curve,
  fits the desk-like two-maturity tranche grid through the homogeneous
  tranche-implied correlation workflow, and requires near-zero quote residuals
  with no root failures
- Quanto-correlation calibration is currently covered by targeted workflow
  regression tests that check the dependency DAG, repricing residuals,
  materialization, and missing-input diagnostics; it is not yet in the
  benchmark pack

The benchmark baseline in ``docs/benchmarks/calibration_workflows.{json,md}``
complements those fit-quality gates with cold-start versus warm-start timing
expectations for the benchmarked workflows. The benchmark pack now covers ten
workflows: Hull-White, caplet stripping, SABR, swaption cube assembly,
equity-vol surface authority, Heston single-smile fitting, Heston
surface-compression fitting, local vol, single-name credit, and basket-credit
tranche-implied correlation. Warm-start timing is tracked on the four
workflows that expose explicit warm seeds (Hull-White, SABR, Heston
single-smile, and Heston surface compression).

The first validation tranche also carries one desk-like basket-credit fixture
instead of only synthetic single-asset fixtures. That fixture links a
materialized single-name credit curve to six tranche expected-loss quotes
across two maturities and three attachment/detachment bands. Its benchmark
payload records an explicit perturbation diagnostic for a parallel quote bump
and a latency envelope for the cold-start solve, so quote-instability and
performance drift are visible in the persisted artifact instead of being
inferred from pass/fail replay alone.

Those checked fixtures now read from the same bounded mock-snapshot contracts
used by replay and proving paths instead of from a separate benchmark-only
surface. The SABR canary reads the seeded ``usd_rates_smile`` surface and its
rate-vol hints from ``prior_parameters.synthetic_generation_contract``. The
Heston and local-vol canaries read the seeded ``spx_heston_implied_vol``
surface and the derived ``spx_local_vol`` linkage from that same synthetic
contract. The single-name credit canary continues to read spread/recovery
inputs from the derived ``model_consistency_contract`` compatibility packet.
The basket-credit canary layers its desk-like tranche surface on top of that
single-name credit materialization to keep the representative-curve linkage
explicit.

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

The legacy bootstrap lane still works on year-fraction tenors. Trellis now
also ships one first dated multi-curve lane:

- ``DatedBootstrapInstrument`` carries explicit ``start_date`` and
  ``end_date`` together with optional schedule overrides such as stub type,
  roll convention, and business-day adjustment
- ``DatedBootstrapCurveInputBundle`` groups those dated instruments, names the
  curve role, and records hard dependencies such as a forecast curve depending
  on a discount curve
- ``MultiCurveBootstrapProgram`` packages the settlement date and the ordered
  curve bundles into one explicit chained calibration program

That dated path is still bounded rather than a full desk calendar engine, but
it moves the first supported rates slice away from tenor-only abstractions and
toward explicit dated instruments and dependency-aware curve reconstruction.

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

The dated path mirrors that contract:

- ``bootstrap_dated_curve_result(...)`` solves one dated curve bundle and
  returns the same diagnostics and replay surfaces as the tenor-only lane
- ``bootstrap_multi_curve_program(...)`` executes the explicit dependency graph
  in topological order, preserving the dependency order and dependency graph in
  the returned ``MultiCurveBootstrapResult``
- ``MultiCurveBootstrapResult.apply_to_market_state(...)`` materializes the
  selected discount and forecast curves back onto ``MarketState`` so downstream
  pricing and later rates-vol fits consume the calibrated runtime objects
  directly

This is still a first bounded OIS-plus-forecast chain rather than a full basis,
cross-currency, smoothing, or exchange-grade futures plant.

Rates Option Calibration
------------------------

The current shipped rates-vol slice now has a first market-object layer in
addition to the older quote-local helpers.

Trellis still supports the flat-Black quote inversion helpers
(``calibrate_cap_floor_black_vol(...)`` and ``calibrate_swaption_black_vol(...)``),
but it now also exposes:

- ``calibrate_caplet_vol_strip_workflow(...)`` for bounded caplet stripping
  from sequential cap quotes into a reusable caplet-vol surface
- ``calibrate_swaption_vol_cube_workflow(...)`` for bounded tenor-aware
  swaption-cube assembly from normalized swaption quotes
- ``compare_sabr_to_swaption_cube_workflow(...)`` for staged comparison between
  that reconstructed market cube and per-slice SABR compression

Those market objects materialize back onto ``MarketState`` through the existing
black-vol-surface binding path, so downstream cap/floor and rate-style
swaption pricing can consume the calibrated outputs directly instead of
reconstructing helper-local vol assumptions.

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

The newer rates-vol market-object workflows sit one layer above those
single-quote helpers.

Bounded caplet stripping
~~~~~~~~~~~~~~~~~~~~~~~~

``calibrate_caplet_vol_strip_workflow(...)`` treats cap quotes as a sequential
ladder where each maturity adds one new caplet period. Within that bounded
support contract, the workflow:

1. normalizes each cap quote onto price space,
2. strips the incremental caplet PV from the maturity ladder,
3. inverts the last caplet on each strike line back to Black vol, and
4. assembles the stripped nodes into a reusable ``GridVolSurface``.

This gives Trellis a first checked caplet-vol market object that downstream
cap/floor pricing can consume through ``market_state.vol_surface``.

The current support is intentionally bounded: it assumes a one-step cap ladder,
Black-lognormal quotes, and no desk-style caplet bootstrap smoothing or
normal/shifted-vol governance.

Bounded swaption cube assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``calibrate_swaption_vol_cube_workflow(...)`` widens the rates-vol layer from
single quote inversion into a tenor-aware cube authority.

For each swaption quote, the workflow:

1. resolves expiry, forward swap rate, and underlier tenor from the selected
   discount and forecast curves,
2. normalizes price quotes back onto Black vol when necessary,
3. assembles the normalized quotes onto an expiry-tenor-strike cube, and
4. materializes a tenor-aware ``SwaptionVolCube`` back onto ``MarketState``.

The rate-style swaption route detects that tenor-aware cube and resolves the
Black vol through its ``swaption_black_vol(...)`` hook before falling back to
the generic two-dimensional ``black_vol(...)`` surface protocol. That keeps
the rates-vol authority inside the Trellis runtime abstraction instead of
introducing a route-local side channel.

The current support is still bounded to a rectangular absolute-strike cube.
It does not yet claim the broader desk surface stack around relative-moneyness
grids, bid/ask governance, arbitrage repair, or normal/shifted conventions.

Staged SABR compression
~~~~~~~~~~~~~~~~~~~~~~

``compare_sabr_to_swaption_cube_workflow(...)`` then treats the reconstructed
swaption cube as the market object and fits one SABR smile per expiry-tenor
slice. The comparison result records the fitted slice pack together with the
full cube residuals so the rates-vol layer can distinguish:

- the reconstructed market cube itself, and
- the reduced-form SABR compression built on top of it

This is the same market-object-first staging that the hardened equity-vol slice
now uses, but applied to a bounded rates-vol program.

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

This remains a bounded single-name CDS workflow rather than a full production
CDS curve plant, but the supported surface is now schedule-aware. Legacy tenor
quotes still default to a quarterly ``ACT/360`` schedule from
``market_state.settlement``. Quotes can also carry explicit CDS effective and
maturity dates, frequency, day count, calendar, business-day adjustment, roll
convention, stub rule, and payment lag. Those conventions flow into both quote
normalization and repricing diagnostics instead of being reconstructed locally
by downstream consumers.

The workflow accepts tenor-labeled quotes in the shipped credit quote families:

- ``Spread`` for running-spread quotes, normalized onto decimal running-spread
  fit space
- ``Upfront`` for standard-coupon-plus-upfront quotes, normalized through the
  CDS pricing stack onto an equivalent fitted quote-style value while repricing
  the supplied standard coupon and upfront amount directly
- ``Hazard`` for direct hazard-rate targets, normalized through the same CDS
  pricing stack on a bounded flat-hazard surrogate

Running-spread quotes greater than ``1.0`` are interpreted as basis points.
Upfront quotes with absolute magnitude greater than ``1.0`` are interpreted as
upfront points, so ``5.0`` means ``5%`` of notional.

The solve objective fits quote-style values through the CDS pricer rather than
solving an identity map in hazard space. Running-spread and hazard quotes fit
model-implied CDS par spreads. Upfront quotes fit the clean upfront PV under
the supplied standard running coupon. The pricing semantics remain explicit in
provenance. The credit workflow records a potential-binding payload that makes
the reduced-form discount/default contract visible:

.. math::

   D_{\text{risky}}(t) = D(t)\,S(t)

where the selected discount curve and the calibrated credit curve are both
named in the runtime binding metadata.

Like the other migrated calibration helpers, the credit workflow returns:

- the typed ``solve_request`` and normalized ``solve_result``
- governed ``solver_provenance`` and ``solver_replay_artifact``
- quote-map metadata and any inverse-transform warnings
- bounded repricing diagnostics, including target/model running spreads,
  quote-style fit values, repricing errors, survival probabilities, forward
  hazards on the tenor grid, and a compact hazard-governance payload
- a reusable ``CreditCurve`` together with ``apply_to_market_state(...)`` for
  shared runtime materialization

The single-name slice still does not itself widen into structural-credit,
hybrid credit-equity calibration, bid/ask governance, index-credit
conventions, or a smoothed/regularized production CDS curve plant. Basket
tranche correlation calibration is handled by the separate bounded workflow
below.

Basket Credit Tranche-Implied Correlation
-----------------------------------------

Trellis now has a bounded basket-credit calibration sleeve for homogeneous
tranche fixtures. The entry point is
``calibrate_homogeneous_basket_tranche_correlation_workflow(...)``. It consumes
a ``MarketState`` that already carries a representative single-name
``CreditCurve`` and a discount curve, normally by applying the result of the
single-name CDS workflow first. The basket calibration does not infer or
rebuild hidden marginal default curves.

The supported quote axes are explicit:

- maturity in years
- attachment
- detachment
- quote value
- quote family and style

The first slice accepts the shipped quote-map vocabulary where it applies:
``Price`` quotes for tranche present value or expected-loss-fraction targets,
and ``Spread`` quotes for fair-spread-in-basis-points targets. These quote maps
are identity maps in quote space; the pricing kernel is the existing
homogeneous basket-credit tranche helper.

For each quote, Trellis solves one tranche-implied correlation under the
one-factor Gaussian copula by scanning the configured correlation interval and
then applying a Brent root solve on a bracketed interval. The calibrated object
is an exact-node ``BasketCreditCorrelationSurface`` keyed by maturity,
attachment, and detachment. It can be materialized onto ``MarketState`` as
``correlation_surface`` and then consumed by basket-tranche pricing when a
contract does not provide an explicit correlation.

The governance diagnostics are intentionally visible:

- quote residuals in the original quote units
- impossible quote / root-bracketing failures
- simple tranche-bound sanity warnings
- monotonicity warnings across fitted detachment or maturity slices
- smoothness warnings when adjacent fitted correlations jump above the
  configured threshold

This is not a full production base-correlation bootstrap. Trellis does not yet
fit heterogeneous single-name portfolios, index-credit conventions, bid/ask
tranche surfaces, interpolated/smoothed base-correlation curves, or arbitrage
repair across a market tranche grid. The supported claim is narrower:
homogeneous representative-curve tranche-implied correlations with governed
diagnostics and first-class runtime materialization.

Implementation
--------------

.. autofunction:: trellis.models.calibration.implied_vol.implied_vol
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr_smile_workflow
.. autofunction:: trellis.models.calibration.sabr_fit.build_sabr_smile_surface
.. autofunction:: trellis.models.calibration.sabr_fit.fit_sabr_smile_surface
.. autofunction:: trellis.models.calibration.equity_vol_surface.clean_equity_vol_surface_quotes
.. autofunction:: trellis.models.calibration.equity_vol_surface.calibrate_equity_vol_surface_workflow
.. autofunction:: trellis.models.calibration.equity_vol_surface.fit_equity_vol_surface
.. autofunction:: trellis.models.calibration.equity_vol_surface.build_equity_vol_surface_input
.. autofunction:: trellis.models.calibration.equity_vol_surface.calibrate_local_vol_surface_from_equity_vol_surface_workflow
.. autofunction:: trellis.models.calibration.equity_vol_surface.compare_heston_to_equity_vol_surface_workflow
.. autofunction:: trellis.models.calibration.equity_vol_surface.compare_heston_surface_to_equity_vol_surface_workflow
.. autofunction:: trellis.models.calibration.heston_fit.calibrate_heston_smile_workflow
.. autofunction:: trellis.models.calibration.heston_fit.build_heston_smile_surface
.. autofunction:: trellis.models.calibration.heston_fit.fit_heston_smile_surface
.. autofunction:: trellis.models.calibration.heston_fit.calibrate_heston_surface_workflow
.. autofunction:: trellis.models.calibration.heston_fit.calibrate_heston_surface_from_equity_vol_surface_workflow
.. autofunction:: trellis.models.calibration.heston_fit.build_heston_surface_input
.. autofunction:: trellis.models.calibration.heston_fit.fit_heston_surface
.. autofunction:: trellis.models.calibration.local_vol.calibrate_local_vol_surface_workflow
.. autofunction:: trellis.models.calibration.local_vol.dupire_local_vol_result
.. autofunction:: trellis.models.calibration.local_vol.dupire_local_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_cap_floor_black_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_swaption_black_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_hull_white
.. autofunction:: trellis.models.calibration.rates_vol_surface.calibrate_caplet_vol_strip_workflow
.. autofunction:: trellis.models.calibration.rates_vol_surface.calibrate_swaption_vol_cube_workflow
.. autofunction:: trellis.models.calibration.rates_vol_surface.compare_sabr_to_swaption_cube_workflow
.. autofunction:: trellis.models.calibration.credit.calibrate_single_name_credit_curve_workflow
.. autofunction:: trellis.models.calibration.basket_credit.calibrate_homogeneous_basket_tranche_correlation_workflow
.. autoclass:: trellis.models.calibration.dependency_graph.CalibrationDependencyGraph
.. autoclass:: trellis.models.calibration.dependency_graph.CalibrationDependencyNode
.. autofunction:: trellis.models.calibration.quanto.calibrate_quanto_correlation_workflow
.. autoclass:: trellis.models.calibration.heston_fit.HestonSmileSurface
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSmileFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSmileCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSurfaceInput
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSurfaceFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.heston_fit.HestonSurfaceCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolSurfaceInput
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolQuoteAdjustment
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolQuoteCleaningDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolQuoteCleaningResult
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.SVISmileParameters
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.SVISmileFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.SVISmileFitResult
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolSurfaceFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolSurfaceAuthorityResult
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolStageComparisonResult
   :members:
.. autoclass:: trellis.models.calibration.equity_vol_surface.EquityVolSurfaceStageComparisonResult
   :members:
.. autoclass:: trellis.models.calibration.sabr_fit.SABRSmileSurface
   :members:
.. autoclass:: trellis.models.calibration.sabr_fit.SABRSmileFitDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.sabr_fit.SABRSmileCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.CapletStripQuote
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.CapletVolStripDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.CapletVolStripAuthorityResult
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.SwaptionCubeQuote
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.SwaptionVolCube
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.SwaptionVolCubeDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.SwaptionVolCubeAuthorityResult
   :members:
.. autoclass:: trellis.models.calibration.rates_vol_surface.SwaptionCubeStageComparisonResult
   :members:
.. autoclass:: trellis.models.calibration.local_vol.LocalVolCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.credit.CreditHazardCalibrationQuote
   :members:
.. autoclass:: trellis.models.calibration.credit.CreditHazardCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.basket_credit.BasketCreditTrancheQuote
   :members:
.. autoclass:: trellis.models.calibration.basket_credit.BasketCreditCorrelationSurface
   :members:
.. autoclass:: trellis.models.calibration.basket_credit.BasketCreditCalibrationDiagnostics
   :members:
.. autoclass:: trellis.models.calibration.basket_credit.BasketCreditCorrelationCalibrationResult
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveRequest
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveBackendRegistry
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveProvenance
   :members:
.. autoclass:: trellis.models.calibration.solve_request.SolveReplayArtifact
   :members:
.. autofunction:: trellis.curves.bootstrap.bootstrap_dated_curve_result
.. autofunction:: trellis.curves.bootstrap.bootstrap_multi_curve_program
.. autofunction:: trellis.curves.bootstrap.bootstrap_yield_curve
   :no-index:
.. autoclass:: trellis.curves.bootstrap.DatedBootstrapInstrument
   :members:
.. autoclass:: trellis.curves.bootstrap.DatedBootstrapCurveInputBundle
   :members:
.. autoclass:: trellis.curves.bootstrap.MultiCurveBootstrapProgram
   :members:
.. autoclass:: trellis.curves.bootstrap.MultiCurveBootstrapResult
   :members:

References
----------

- Jaeckel, P. (2015). "Let's be rational." *Wilmott*, 2015(75), 40-53.
- Hagan, P. et al. (2002). "Managing smile risk." *Wilmott Magazine*, Sep 2002.
- Dupire, B. (1994). "Pricing with a smile." *Risk*, 7(1), 18-20.
