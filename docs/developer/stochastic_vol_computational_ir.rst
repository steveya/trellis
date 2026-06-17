Stochastic-Vol Computational IR Guide
=====================================

This guide is the operational contract for stochastic-volatility pricing
tasks. It is written for developers and future pricing-function agents that
need to decide whether a task can be built from existing Trellis components,
should fail closed with a useful repair packet, or needs a new numerical
abstraction.

The boundary is internal. Public pricing APIs should not depend on these
diagnostic objects directly.

Canonical notation
------------------

Use these names consistently when writing tasks, diagnostics, tests, or docs:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Term
     - Meaning
   * - ``derivative_family="option"``
     - The broad derivative axis. It is not a route key by itself.
   * - ``payoff_family="vanilla_option"``
     - The terminal payoff shape. It can stay broad while other axes carry
       the underlier, exercise, and model facts.
   * - ``underlying_asset_class``
     - The underlier axis, such as ``equity``, ``fx``, ``rate``, ``future``,
       ``credit``, or ``commodity``.
   * - ``exercise_style``
     - The exercise/control axis, such as ``european``, ``american``,
       ``bermudan``, ``issuer_call``, or ``holder_put``.
   * - ``option_type``
     - The call/put axis when the payoff side is part of the contract.
   * - ``process_family``
     - The stochastic process family used by the numerical problem, such as
       ``heston``, ``bates``, or ``slv_lsv``.
   * - ``solver_target``
     - The computational method target, such as ``fft_transform``,
       ``cos_transform``, ``monte_carlo_qe``, ``pde_adi``,
       ``surface_calibration``, or ``gauss_laguerre_quadrature``.
   * - ``model_parameters``
     - Explicit stochastic-vol model parameters. For Heston these are
       ``kappa``, ``theta``, ``xi``, ``rho``, and ``v0`` plus any route-owned
       drift/carry inputs.
   * - ``black_vol_surface``
     - Market quote evidence or a calibration target. It is not a live
       substitute for Heston model parameters.
   * - ``calibration_problem``
     - The recorded bridge that consumes market quotes and produces a
       ``calibrated_model_parameter_set``.
   * - ``validation_bundle``
     - The deterministic validation contract for the selected computational
       family, for example ``heston:transform`` or ``slv_lsv:leverage_function``.
   * - ``repair_packet``
     - A machine-readable missing-primitive or unsupported-class packet used
       by remediation tooling before raw exception text.
   * - ``expected_honest_block``
     - An intentional fail-closed result for a known unsupported composite
       shape.

Do not narrow option products into synthetic product families just to recover
route identity. An American equity put can remain
``derivative_family="option"``, ``payoff_family="vanilla_option"``,
``underlying_asset_class="equity"``, ``exercise_style="american"``, and
``option_type="put"``. The route, helper, and validation bundle must consume
those axes before accepting an adapter.

Lifecycle
---------

Stochastic-vol task execution should move through these stages:

1. The request or task manifest supplies semantic axes such as payoff family,
   underlier, exercise style, option side, model family, and requested methods.
2. ``trellis.agent.computational_problem_ir`` classifies each comparison
   target into one computational bucket and records target-local
   ``process_family``, ``solver_target``, market-binding needs, and
   ``validation_bundle``.
3. Route selection checks whether an admitted helper or route already satisfies
   the exact computational problem. Broader adapters are not acceptable just
   because the terminal payoff is numerically vanilla-shaped.
4. Deterministic semantic gates run before a task is declared successful. For
   Heston model-parameter routes, generated code that reads
   ``market_state.vol_surface`` is rejected as
   ``heston_black_vol_surface_mismatch``.
5. Method builds and cross-method comparison produce task-result evidence. If
   all builds succeed but comparison fails, the failure is comparison evidence,
   not a prompt to invent a new route.
6. ``trellis.agent.task_diagnostics`` persists the diagnosis packet and
   Markdown dossier. The packet carries outcome, failure bucket, primary
   failure, computational problem evidence, and repair packets.
7. Remediation tools group by structured bucket and repair packet before using
   raw error strings.

Production-like behavior remains deterministic and fail-closed. Any
AI-assisted repair belongs in offline task, canary, developer, or remediation
workflows; it is not live self-modifying pricing logic.

Computational buckets
---------------------

``classify_stochastic_vol_task(...)`` currently emits these stable buckets:

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Bucket
     - Common targets
     - Contract
   * - ``stochastic_vol_transform``
     - ``fft_transform``, ``cos_transform``, ``gauss_laguerre_quadrature``
     - Terminal payoff under a characteristic-function family such as
       ``heston_log_spot``. FFT and COS bind to checked Heston helpers;
       Gauss-Laguerre is a typed blocker until the quadrature kernel exists.
   * - ``stochastic_vol_monte_carlo``
     - ``monte_carlo_euler``, ``monte_carlo_qe``
     - European Heston vanilla Monte Carlo with explicit Heston model
       parameters and an explicit scheme.
   * - ``stochastic_vol_pde``
     - ``pde`` or ``pde_adi``
     - A two-state stochastic-vol PDE target. It must carry explicit model
       parameters and the ``heston:pde`` validation bundle before any PDE route
       can be admitted.
   * - ``calibration_to_surface``
     - ``surface_calibration``
     - A market-surface-to-model-parameter problem. This is the bridge that may
       consume Black implied vols or option prices and output Heston model
       parameters.
   * - ``affine_jump_stochastic_vol``
     - ``affine_jump_transform``, ``affine_jump_monte_carlo``
     - Bates-style Heston plus compound-Poisson lognormal jumps. Current tasks
       get a repair packet for ``bates_affine_jump_stochastic_vol_kernel``.
   * - ``slv_lsv``
     - ``leverage_function_pde``, ``leverage_function_monte_carlo``
     - Stochastic-local-vol or local-stochastic-vol targets. They require
       local-vol and Black-vol surface authority, Heston parameters, leverage
       calibration provenance, and solver-specific contracts.
   * - ``unsupported_path_dependent_control``
     - Heston American Asian barrier PDE, MC, or transform comparisons
     - A composite control problem requiring path-state simulation, event
       monitor, payoff summary, early-exercise policy, and Heston path-state
       coupling. These are expected honest blocks today.

If a task compares several targets, the task-level bucket may be
``stochastic_vol_mixed`` while each target keeps one of the concrete buckets.

Heston parameters and Black vol surfaces
----------------------------------------

Heston pricing routes consume explicit model parameters. A task can provide
them directly on the spec, through ``market_state.model_parameters``, from a
synthetic fixture, or from a recorded calibration result.

Black implied-vol surfaces are different objects. They are market evidence,
comparison evidence, or calibration targets. They become Heston model
parameters only when a ``calibration_problem`` records that bridge and produces
a ``calibrated_model_parameter_set``. Therefore:

- bumping ``kappa``, ``theta``, ``xi``, ``rho``, or ``v0`` is a model-parameter
  sensitivity
- bumping a Black volatility surface is market-quote sensitivity
- recalibrating Heston after a Black surface bump is a calibration workflow,
  not an automatic side effect of a pricing route

This distinction is what protects T20-style Heston parameter tasks from being
judged as if the runtime had silently recalibrated after a Black-vol bump.

Route responsibilities
----------------------

Use the existing helper surface before generating adapters:

- ``trellis.models.transforms.heston.price_heston_option_transform(...)`` owns
  checked Heston FFT and COS transform pricing.
- ``trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo(...)``
  owns bounded European Heston Monte Carlo with ``scheme="euler"`` or
  ``scheme="heston_qe"``.
- ``trellis.models.calibration.heston_fit`` owns the bounded Heston smile and
  surface compression workflows that can produce reusable model parameters.
- Heston Gauss-Laguerre, Bates, SLV/LSV, and path-dependent Heston control
  targets should produce repair packets or honest blocks until the named
  primitives and validation bundles exist.

Thin route adapters may bind task-specific specs to these helpers, but they
must not reimplement the stochastic process, transform kernel, Monte Carlo
scheme, calibration bridge, or validation semantics locally.

Validation and reviewer policy
------------------------------

Deterministic validation is authoritative for known semantic mismatches. The
model validator is not a substitute for typed route, market-binding, or
validation-bundle checks.

Use this policy:

- Run deterministic semantic and validation-bundle checks first.
- Skip model-validator LLM review when no payoff exists, when the target is an
  expected honest block, or when deterministic validation already identifies a
  concrete route, bundle, market-binding, or missing-primitive issue.
- Keep build-time model-validator LLM review behind ``validation="thorough"``.
  In ``standard`` validation, critic/model-validator surfaces are deterministic
  or advisory according to the review policy.
- For cross-method failures where all methods built successfully, preserve the
  method prices, reference target, tolerances, payoff classes, selected routes,
  validation bundles, and computational problem evidence. That evidence can
  support quant or model-validator review in assisted/remediation workflows,
  but strict production-like runs still fail closed.

Repair packets
--------------

Repair packets should name reusable missing abstractions, not one-off generated
adapter fixes. Current stochastic-vol packets include:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Packet or primitive
     - Meaning
   * - ``heston_gauss_laguerre_transform_kernel``
     - Add a checked Heston characteristic-function quadrature kernel, damping
       or contour policy, stabilization diagnostics, and validation bundle.
   * - ``bates_affine_jump_stochastic_vol_kernel``
     - Add Bates characteristic-function and simulation capability over Heston
       base parameters plus lognormal jump parameters.
   * - ``leverage_function_contract``
     - Add an executable SLV/LSV leverage calibration and solver contract over
       local-vol, Black-vol, and Heston inputs.
   * - ``path_dependent_heston_control_contract``
     - Add a path-state, event-monitor, payoff-summary, early-exercise-control,
       and stochastic-vol coupling contract before admitting the composite
       route.

When a new kernel is needed, add it at the numerical or computational layer,
then bind it through a thin adapter. Do not patch one task by embedding a
private formula into a generated ``trellis.instruments._agent`` module.

Task map
--------

Use the recent task pack as regression examples:

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - Task
     - Bucket
     - Lesson
   * - ``T20``
     - ``stochastic_vol_pde`` and ``stochastic_vol_monte_carlo``
     - Explicit Heston parameters are not Black-vol surface bumps. Recalibration
       requires a recorded calibration problem.
   * - ``T28``
     - ``stochastic_vol_monte_carlo`` and ``stochastic_vol_transform``
     - Euler/QE Heston MC and Heston FFT target binding belong on checked helper
       surfaces.
   * - ``T40`` and ``T76``
     - ``stochastic_vol_mixed``
     - Transform, MC, and PDE targets need target-local process, solver, and
       validation-bundle evidence.
   * - ``T67``
     - ``calibration_to_surface``
     - Market prices or implied vols are calibration inputs until a fit records
       output Heston model parameters.
   * - ``T114``
     - ``stochastic_vol_transform``
     - Heston Gauss-Laguerre is a missing quadrature primitive, not a reason to
       fall back to an FX or Black-vol vanilla adapter.
   * - ``T44``
     - ``affine_jump_stochastic_vol``
     - Bates requires Heston plus jump parameters and a named affine jump
       stochastic-vol primitive.
   * - ``T60`` and ``T117``
     - ``slv_lsv``
     - SLV/LSV requires leverage-function authority and solver contracts.
   * - ``E27``
     - ``unsupported_path_dependent_control``
     - Heston American Asian barrier requests are composite control problems
       and should block honestly until the abstractions exist.

Extension checklist
-------------------

Before marking a stochastic-vol task green, verify:

1. The option axes are explicit and no route depends on a broad product label
   alone.
2. Every comparison target has the right computational bucket, process family,
   solver target, market-binding semantics, and validation bundle.
3. Heston model-parameter routes do not read a Black vol surface unless a
   calibration problem explicitly produced the parameters.
4. Unsupported families emit repair packets or expected honest blocks with
   named missing primitives.
5. Method prices, tolerances, selected routes, payoff classes, and validation
   bundles are persisted when cross-method comparison fails.
6. Docs and ``LIMITATIONS.md`` are updated when support contracts move.

Related pages:

- :doc:`task_diagnostics` for the diagnosis packet and dossier schema.
- :doc:`task_and_eval_loops` for task-run and learning-loop operations.
- :doc:`../quant/contract_ir` for option-axis and structural contract
  semantics.
- :doc:`../quant/pricing_stack` for the quant-facing pricing-stack overview.
- :doc:`../mathematical/transforms`, :doc:`../mathematical/monte_carlo`,
  :doc:`../mathematical/calibration`, and :doc:`../mathematical/processes` for
  the numerical method references.
