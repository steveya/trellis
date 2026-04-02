Lattice Algebra
===============

This note records the lattice algebra that Trellis ships and the general
framework under development. It is the lattice counterpart to
:doc:`contract_algebra`.

A Trellis lattice is a calibrated family of positive one-step pricing operators
on finite recombining state spaces. Contracts are linear cashflow claims wrapped
by finite-state overlays and single-controller obstacle operators.

The goal is not one universal tree implementation. The goal is one checked API
boundary that multiple lattice families can target, so that the semantic DSL,
route registry, and agent system interact with lattice pricing through
consistent, type-safe contracts rather than ad hoc per-product wiring.

Mathematical Foundation
-----------------------

The general lattice algebra is defined as a **finite-horizon positive Markov
pricing system on a recombining graded state graph**. Classical binomial and
trinomial trees are first-class specializations.

Fix a time grid :math:`\Pi = \{0=t_0 < t_1 < \cdots < t_n = T\}`. At each
time step :math:`m`, let :math:`E_m` be a finite state set. Each node carries:

- **Latent state** :math:`X_m : E_m \to \mathbb{R}^d` — where the model
  evolves and moment matching is defined.
- **Observable bundle** :math:`Y_m : E_m \to \mathcal{O}` — what contracts
  actually inspect.

The one-step pricing operator kernel is:

.. math::

   M_m(i,k) \ge 0, \qquad i \in E_m,\; k \in E_{m+1}

acting on a value vector :math:`f` defined on :math:`E_{m+1}` by:

.. math::

   (\mathcal{M}_m f)(i) = \sum_{k \in E_{m+1}} M_m(i,k)\, f(k)

Under the money-market numeraire, this factorizes as
:math:`M_m(i,k) = D_m(i) P_m(i,k)` with :math:`P_m(i,\cdot)` row-stochastic
and :math:`D_m(i) > 0`. The operator :math:`\mathcal{M}_m` is the true
invariant object; the factorization is one convenient special case.

**Recombination** means the state graph collapses exponentially many paths into
polynomially many nodes: :math:`|E_m| = O(m^q)` for fixed factor dimension.

**Local consistency** for diffusion-style families requires first- and
second-moment matching of the discrete transitions against the continuous-time
target process in latent-state space.

**Admissibility** requires positivity (:math:`P_m(i,k) \ge 0`) and
normalization (:math:`\sum_k P_m(i,k) = 1`). Silent probability clipping is a
fallback policy with diagnostics, not part of the mathematical definition.

Topology and Mesh
-----------------

Topology and numerical coordinates are separate objects.

**Topology** answers: which nodes exist at each step, which edges connect
parent to child, whether the graph recombines, and how indexing works.

**Mesh** provides coordinate maps
:math:`\chi_m : E_m \to \mathbb{R}^d` parameterized by step sizes, truncation
rules, and state transforms. Classical formulas like
:math:`x^{\text{bin}}_m(j) = (2j-m)\Delta x_m` are specific mesh instances.

This separation supports non-uniform grids, truncated lattices, log-spot
meshes, and product meshes for two-factor problems.

Observable State Mapping
~~~~~~~~~~~~~~~~~~~~~~~~

The observable bundle is obtained from the latent state through model-specific
maps:

- Additive normal: :math:`r_m(i) = \varphi_m + x_m(i)`
- Multiplicative lognormal: :math:`r_m(i) = \exp(\varphi_m + x_m(i))`
- Equity/spot: :math:`S_m(i) = \psi(x_m(i))`, often :math:`\psi(x) = e^x`

Model Registry
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 15 15 15

   * - Model
     - Latent State
     - Observable
     - Calibration
     - Topologies
     - Mean Reversion
   * - Hull-White 1F
     - additive Gaussian
     - :math:`r = \varphi + x`
     - term structure
     - bin, tri
     - explicit
   * - Ho-Lee
     - additive Gaussian
     - :math:`r = \varphi + x`
     - term structure
     - bin, tri
     - none
   * - BDT
     - additive latent
     - :math:`r = e^{\varphi+x}`
     - term structure
     - bin, tri
     - implied
   * - Black-Karasinski
     - additive latent
     - :math:`r = e^{\varphi+x}`
     - term structure
     - bin, tri
     - explicit
   * - CRR
     - log-spot
     - :math:`S = e^x`
     - analytical
     - bin
     - none
   * - Jarrow-Rudd
     - log-spot
     - :math:`S = e^x`
     - analytical
     - bin
     - none
   * - Local Vol
     - spot / log-spot
     - :math:`S`
     - vol surface
     - bin, tri
     - none

Calibration
-----------

Calibration is defined at the level of the lattice family as constrained
fitting of the pricing operator, not as a model-specific side procedure.

Arrow-Debreu state prices are forward propagation of the same one-step pricing
operators used in rollback:

.. math::

   q_0 = \delta_{i_0}, \qquad
   q_{m+1}(k) = \sum_{i \in E_m} q_m(i)\, M_m(i,k)

The calibration round-trip: :math:`P(0,t_m) = \sum_{k \in E_m} q_m(k)` matches
market discount factors at all calibrated maturities.

Three calibration regimes exist:

- **Term structure** (short-rate models): Arrow-Debreu exact fit, closed-form
  for additive-normal, Newton for lognormal.
- **Analytical** (CRR, Jarrow-Rudd): parameters from carry, vol, time step.
- **Vol surface** (local vol): ill-posed inverse problem with smoothing /
  regularization as part of calibration target.

The three-pass protocol (seed → calibrate drift → update probabilities →
re-calibrate) is a specific solver for one-factor rate trees, not the general
definition.

Contract Algebra
----------------

The general lattice algebra distinguishes three composition layers.

Linear Claim Layer
~~~~~~~~~~~~~~~~~~

Terminal payoff :math:`H_n`, node cashflows :math:`c_m(i)`, optional edge
cashflows :math:`g_m(i,k)`. Backward induction:

.. math::

   V_n(i) = H_n(i), \qquad
   U_m(i) = c_m(i) + \sum_k M_m(i,k)\bigl(g_m(i,k) + V_{m+1}(k)\bigr)

This layer is linear and closed under addition and scalar multiplication.

Single-Controller Obstacle Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nonlinear control wrappers around the continuation value:

.. math::

   V_m(i) = \begin{cases}
   U_m(i), & m \notin A \\
   \max\{E_m^{\mathrm{ex}}(i),\, U_m(i)\}, & \text{holder maximize} \\
   \min\{E_m^{\mathrm{ex}}(i),\, U_m(i)\}, & \text{issuer minimize}
   \end{cases}

This is the discrete obstacle / Snell-envelope structure. It is not part of the
linear cashflow algebra.

In the current code this is represented by:

- ``ControllerProtocol`` on the semantic contract
- ``ExerciseLatticeIR.control_style``
- ``trellis.models.trees.control.LatticeExercisePolicy``
- ``resolve_lattice_exercise_policy_from_control_style(...)``

Event-Overlay Layer
~~~~~~~~~~~~~~~~~~~

Event overlays enlarge the state space by a finite event state
:math:`z \in Z_m`. The augmented pricing operator is:

.. math::

   \widetilde{M}_m\bigl((i,z),(k,z')\bigr)
   = M_m(i,k)\,\Gamma_m(z,z' \mid i,k)

where :math:`\Gamma_m` is an event transition kernel conditioned on the edge
:math:`(i,k)`. The edge dependence is important: barrier and related overlays
depend on what happens between parent and child states, not only on node
labels.

Applicable to: knock-in/knock-out barriers, default/no-default with recovery,
range-accrual counters, coupon-memory and notice-state mechanics.

Closure and Boundary
~~~~~~~~~~~~~~~~~~~~

The framework is closed under:

1. Addition and scalar multiplication of linear claims.
2. Application of a single-controller obstacle wrapper.
3. Finite-state augmentation by an event overlay, provided recombination and
   polynomial node growth are preserved.

The framework is **not** closed under arbitrary nesting of nonlinear control
games.

Explicit inclusion criteria:

- Finite-horizon
- Markov after finite augmentation
- Recombining with polynomial node growth
- Factor count ≤ 2
- Single numeraire
- Single-controller obstacle problems only

Shipped Lattice Slice
---------------------

The current shipped lattice boundary is:

.. code-block:: text

   SemanticContract
     -> ProductIR
     -> LatticeRecipe
     -> compile_lattice_recipe()
     -> (LatticeTopologySpec, LatticeMeshSpec, LatticeModelSpec, LatticeContractSpec)
     -> build_lattice()
     -> price_on_lattice()

The shipped ``ExerciseLatticeIR`` covers tranche-1 strategic-rights products:

- callable bonds
- Bermudan swaptions

Timing And Phase Order
~~~~~~~~~~~~~~~~~~~~~~

Same-day ordering is first-class in the shipped lattice slice.

The default tranche-1 phase order is:

- ``EVENT``
- ``OBSERVATION``
- ``DECISION``
- ``DETERMINATION``
- ``SETTLEMENT``
- ``STATE_UPDATE``

The lattice lowering validates, at minimum:

- observation before decision
- decision before settlement
- settlement dates not earlier than decision dates

Typed Contract Inputs
~~~~~~~~~~~~~~~~~~~~~

The current lattice slice relies on typed semantic inputs:

- ``SemanticTimeline``
- ``ObservableSpec``
- ``StateField``
- ``ObligationSpec``
- ``ControllerProtocol``

For callable bonds, the required typed observables include:

- ``discount_curve``
- ``cashflow_schedule``

For Bermudan swaptions, the required typed observables include:

- ``forward_rate``
- ``discount_curve``

Admissibility
~~~~~~~~~~~~~

Lattice admissibility is typed through ``RouteSpec.admissibility`` and checked
through ``BuildGateDecision``.

The lattice checks cover:

- topology/model compatibility
- calibration-target compatibility
- supported control styles (``holder_max``, ``issuer_min``, ``identity``)
- factor-count limits
- overlay state-growth limits
- event support
- Numba-compatibility of compiled rollback shape
- reporting and multicurrency limits

Current Helper-Backed Routes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checked helper-backed lattice routes remain:

- ``trellis.models.callable_bond_tree.price_callable_bond_tree``
- ``trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree``

Target API Design
-----------------

The target API exposes the mathematical layers directly.

``LatticeTopologySpec``
   Pure graph structure: branching, node count, child/parent indices.
   No displacement or coordinates.

``LatticeMeshSpec``
   Coordinate maps, step sizes, metric functions, truncation rules,
   state transforms.

``LatticeModelSpec``
   Model family with pricing_operator_fn (first-class), observable_fn,
   numeraire, calibration strategy, state space type.

``LatticeLinearClaimSpec``
   Terminal payoff, node cashflows, edge cashflows.

``LatticeControlSpec``
   Objective (holder_max / issuer_min / identity), exercise steps,
   exercise value function.

``EventOverlaySpec``
   Edge-aware event transition kernel for barriers, defaults, accruals.

``LatticeContractSpec``
   Composite: claim + control + overlay + timeline.

``LatticeRecipe``
   Declarative agent-facing surface with named families and parameter
   objects. Compiled to internal specs via ``compile_lattice_recipe()``.

``CalibrationStrategy``
   Protocol with implementations: ``TermStructureCalibration``,
   ``AnalyticalCalibration``, ``LocalVolCalibration``.

Unified entry points:

- ``build_lattice(topology, mesh, model, calibration_target)``
- ``price_on_lattice(lattice, contract_spec)``

Implemented Advanced Families
-----------------------------

The currently checked advanced extensions are:

- ``LocalVolCalibration`` on a recombining trinomial log-spot mesh, with
  structured probability diagnostics and validation against PDE/Monte Carlo
- ``product_binomial_2f`` for low-dimensional correlated spot products
- shared lattice/PDE adapters in ``trellis.models.grid_protocols`` for common
  spatial-grid and exercise-boundary concepts

These are still intentionally narrower than the full design appendix. The code
supports the shipped local-vol and two-factor contracts that fit the boundary;
it does not claim universal closure over every implied tree, hybrid, or PDE
scheme.

Two-Factor Extension
--------------------

A two-factor lattice uses product state sets
:math:`E_m = E_m^{(1)} \times E_m^{(2)}` with joint kernel satisfying
moment/covariance matching. Implementations may use decorrelation transforms
or sparse joint kernel construction. For trinomial × trinomial:
:math:`|E_m| = (2m+1)^2`, giving :math:`O(n^3)` total nodes.

Lattice-PDE Bridge
------------------

The common bridge is not "trees and PDEs are the same thing." The bridge is a
shared vocabulary:

- ``SpatialGrid`` adapters for lattice and PDE domains
- shared exercise-boundary objects for single-obstacle problems
- backend-specific rollback engines consuming the same obstacle semantics

The bridge is implemented for low-dimensional obstacle problems such as the
American put canary. It is not a claim that every lattice is literally an
explicit finite-difference scheme.

Proof Obligations
-----------------

The lattice algebra carries explicit proof obligations:

- **Operator positivity**: :math:`M_m(i,k) \ge 0` everywhere.
- **State-price consistency**: row-sums, Arrow-Debreu sums match discount
  factors.
- **Martingale test**: tradables are martingales under chosen numeraire.
- **Expressiveness**: all registered families through unified interface.
- **Closure**: overlay augmentation preserves well-formed lattice.
- **Cross-route agreement**: lattice vs analytical/PDE/MC within tolerance.
- **Refinement behavior**: convergence envelope at n, 2n, 4n steps;
  Richardson-style reference. Lattice convergence is often oscillatory.
- **Performance**: 1D fast paths stay within a bounded regression envelope
  relative to direct rollback; fallback is warned, not silent; overlays and
  non-fast-path multidimensional contracts are explicit.

Performance Contract
--------------------

The generalized API exposes the following pricing-path policy:

- ``fast_linear_*`` for plain 1D linear claims
- ``fast_obstacle_*`` for 1D single-controller obstacle problems
- ``fast_product_2d_*`` for terminal-only two-factor product claims
- ``python_*_fallback`` with a runtime warning for overlays and other
  contracts outside the compiled fast path

Carve-Outs
----------

The following families are explicitly outside the general lattice algebra:

- Non-recombining trees (HJM/LMM without Markov projection)
- Non-Markov trees (rough-vol, H < 0.5)
- High-dimensional baskets (5+ underlyings)
- Multi-controller game structures (callable-puttable-convertible)

See :doc:`../developer/lattice_algebra_boundary_adr` for the formal boundary.

Implementation Roadmap
----------------------

.. code-block:: text

   Wave 1 (Foundation):  QUA-484, QUA-506, QUA-487
   Wave 2 (Migration):   QUA-485, QUA-486, QUA-510
   Wave 3 (Advanced):    QUA-507, QUA-508, QUA-509
   Wave 4 (Validation):  QUA-488, QUA-489

References
----------

- Nelson & Ramaswamy (1990), *Simple Binomial Processes as Diffusion
  Approximations in Financial Models*
- Derman, Ergener & Kani (1995), *Enhanced Numerical Methods for Options
  with Barriers*
- Derman, Kani & Zou (1996), *The Local Volatility Surface*
- Kirkby, Dang & Nguyen (2020), *A General CTMC Approximation for
  Multi-Asset Option Pricing*
- See ``trellis_lattice_algebra_design.md`` for full reference list
