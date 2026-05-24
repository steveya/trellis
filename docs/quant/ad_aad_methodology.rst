AD And AAD Methodology
======================

This document explains the differentiable-pricing methodology used by Trellis.
It is written for quants who know pricing and risk, but who may be new to
automatic differentiation (AD), adjoint automatic differentiation (AAD),
vector-Jacobian products (VJP), and the bounded hybrid AD machinery in this
library.

The short version is:

- Trellis treats a supported pricing route as a mathematical map from market
  and model coordinates to value.
- A derivative is only reported when the route, coordinate chart, and backend
  operator are checked and executable.
- Book risk is represented as sparse sensitivities keyed by stable
  ``RiskFactorId`` objects, not as anonymous positional arrays.
- Unsupported trades, discontinuities, missing graph dependencies, and
  unsupported derivative operators fail closed with structured metadata instead
  of silently falling back inside an AAD result.

This is deliberately narrower than "differentiate everything". The aim is to
replace wasteful bump/reprice loops where the derivative is mathematically
well-defined and implementation-supported, while keeping discontinuities,
boundary cases, and unsupported product families honest.

Pricing As A Coordinate Map
---------------------------

For AD, the first question is not "can Python differentiate this function?".
The first question is "what are the differentiable coordinates?"

Trellis separates market objects from coordinates on those objects. A yield
curve, volatility surface, correlation parameter, or model-parameter pack is
not itself a single scalar. It is an object with a coordinate chart:

.. math::

   \theta = (\theta_1, \ldots, \theta_n)

where each :math:`\theta_j` is a supported market or model coordinate such as:

- a zero-rate node on a named ``YieldCurve``
- a scalar value on a shared ``FlatVol``
- a node value on a ``GridVolSurface`` at expiry and strike
- a scalar model parameter, such as a bounded quanto correlation
- an off-diagonal coordinate in a checked correlation matrix chart

For one trade or pricing route, Trellis wants a map:

.. math::

   V : D \subset \mathbb{R}^n \rightarrow \mathbb{R}

where :math:`V(\theta)` is the present value after all non-differentiated inputs
have been fixed. Those fixed inputs can include dates, calendars, exercise
schedules, payoff type, notional, already-bound market conventions, and any
unsupported market objects that are intentionally held fixed for the chosen
lane.

The domain :math:`D` matters. A flat-vol coordinate may live naturally in a
positive-volatility domain, a correlation coordinate may live in
:math:`(-1, 1)`, and a matrix-coordinate chart may be valid only inside the
positive-semidefinite cone with a safety margin. Trellis therefore treats a
coordinate chart as part of the derivative contract, not as presentation
metadata.

For vector-valued outputs, such as a residual vector in calibration, Trellis
uses:

.. math::

   F : D \subset \mathbb{R}^n \rightarrow \mathbb{R}^m

The Jacobian is then:

.. math::

   J_F(\theta) =
   \begin{bmatrix}
   \partial F_1 / \partial \theta_1 & \cdots &
   \partial F_1 / \partial \theta_n \\
   \vdots & \ddots & \vdots \\
   \partial F_m / \partial \theta_1 & \cdots &
   \partial F_m / \partial \theta_n
   \end{bmatrix}

Dense Jacobians are useful for small calibration problems and tests. They are
not the preferred representation for broad book risk because most books touch
only a small subset of all known factors.

Local Differentiability
-----------------------

The mathematical object behind AD is the local linearization of a pricing map.
For a scalar route, differentiability at :math:`\theta` means there is a linear
map :math:`DV(\theta)` such that:

.. math::

   V(\theta + h) =
   V(\theta) + DV(\theta)[h] + o(\lVert h \rVert)

For a scalar map, this linear functional is represented by the gradient:

.. math::

   DV(\theta)[h] = \nabla_\theta V(\theta)^\top h

For a vector map :math:`F`, the derivative is the Jacobian as a linear map:

.. math::

   DF(\theta)[h] = J_F(\theta) h

This local statement is why Trellis is careful around discontinuities,
exercise boundaries, interpolation knots, and PSD boundaries. A function can be
perfectly valid for pricing but not differentiable at every point where it is
valid for valuation.

Trellis generally works with three spaces:

- primal coordinates :math:`\theta`, the market or model inputs being
  differentiated
- tangent directions :math:`v`, perturbations of those coordinates
- cotangent weights :math:`w`, adjoint weights on outputs that are pulled back
  to input-coordinate sensitivities

JVP pushes tangent directions forward. VJP pulls cotangent weights backward.
That tangent/cotangent distinction is the mathematical reason VJP and AAD are
natural for book risk: a book value is scalar, so one reverse pass can pull the
value adjoint back to many input coordinates.

Coordinate Identity
-------------------

The implementation object for a coordinate is ``RiskFactorId``. It is a stable
identity for a differentiable coordinate, not a pointer to a Python object.
Conceptually:

.. code-block:: text

   RiskFactorId(
       object_type="yield_curve",
       object_name="USD-OIS",
       coordinate_type="zero_rate",
       axes={"tenor_years": 5.0},
       currency="USD",
       provenance_namespace="base"
   )

The exact fields vary by factor family, but the rule is the same: if a
sensitivity is reported, the coordinate must be named in a way that survives
aggregation, serialization, reporting, and later selection by request.

This is why Trellis avoids reporting large anonymous arrays as book-level AAD
risk. A positional derivative vector:

.. math::

   (0.12, -0.03, 0.00, 4.2)

is not useful unless every consumer knows which coordinate each element
represents. Trellis instead uses a sparse map:

.. code-block:: text

   SparseRiskVector = {
       RiskFactorId(
           object_type="yield_curve",
           object_name="USD-OIS",
           coordinate_type="zero_rate",
           axes={"tenor_years": "2"},
       ): -1200.0,
       RiskFactorId(
           object_type="yield_curve",
           object_name="USD-OIS",
           coordinate_type="zero_rate",
           axes={"tenor_years": "5"},
       ): -3100.0,
       RiskFactorId(
           object_type="vol_surface",
           object_name="EQ-VOL",
           coordinate_type="black_vol",
           axes={"expiry_years": "1", "strike": "100"},
       ): 42.0,
   }

Zero entries are normally omitted. A missing entry is not automatically an
error; it can mean the factor is unavailable, unsupported, selected but
insensitive, or excluded by policy. The result metadata distinguishes those
cases.

Scalar AD, Greeks, And Trace Safety
-----------------------------------

For a scalar smooth route:

.. math::

   V = V(\theta)

the gradient is:

.. math::

   \nabla_\theta V =
   \left(
   \frac{\partial V}{\partial \theta_1},
   \ldots,
   \frac{\partial V}{\partial \theta_n}
   \right)

Classical Greeks are examples of this idea under particular coordinates:

- delta is a derivative with respect to spot
- vega is a derivative with respect to a volatility coordinate
- rho is a derivative with respect to a rate coordinate
- correlation risk is a derivative with respect to a correlation coordinate

The implementation requirement is trace safety. A pricing route must not
collapse traced scalars with ``float(...)`` or force NumPy arrays to
``dtype=float`` inside the differentiable region. Trellis code that is meant to
be differentiable uses ``trellis.core.differentiable.get_numpy()`` so the same
kernel can run on ordinary numeric values and on ``autograd`` traced values.

The public support contract is intentionally narrower than the mathematical
idea. Trellis supports AD only for routes and market objects that have checked
tests and explicit derivative-method metadata.

Financial Meaning Of The Map
----------------------------

Many pricing routes can be written abstractly as a risk-neutral expectation:

.. math::

   V(\theta)
   =
   \mathbb{E}^{Q_\theta}
   \left[
     D(\theta, \omega)\,
     \Phi(X(\theta, \omega); \alpha)
   \right]

where :math:`D` is discounting, :math:`X` is the simulated or analytically
integrated state, :math:`\Phi` is the payoff, :math:`\alpha` are fixed contract
terms, and :math:`\theta` are the selected market or model coordinates.

Trellis differentiates the implemented pricing map after the route has made
its modeling choices explicit. In a closed-form route, that may mean
differentiating through a normal-CDF formula. In a tree, it may mean
differentiating through a smooth rollback. In Monte Carlo, it means
differentiating a deterministic estimator with fixed shocks:

.. math::

   \widehat{V}_N(\theta)
   =
   \frac{1}{N}
   \sum_{k=1}^{N}
   D(\theta, z_k)\,
   \Phi(X(\theta, z_k); \alpha)

where :math:`z_k` are supplied shocks. Under ordinary smoothness and dominated
convergence conditions, the pathwise derivative of this estimator is a
consistent estimator of the derivative of the expectation. When those
conditions fail, for example at barrier crossings or exercise-boundary changes,
Trellis does not report ordinary pathwise AD as if it were mathematically
valid.

This is also why model and measure choices must already be encoded in the
route. AD differentiates the chosen computational model. It does not decide
whether the model, measure, or calibration target is economically appropriate.

Curve And Surface Coordinates
-----------------------------

Curve and surface derivatives need special care. A public curve object is a
function of both node values and query location:

.. math::

   z(t; \theta_1, \ldots, \theta_n)

where :math:`t` is the maturity or query coordinate and :math:`\theta_j` are
curve node values. Trellis' supported curve and grid-vol AD contracts are node
value contracts. They answer questions such as:

.. math::

   \frac{\partial V}{\partial \theta_j}

for fixed trade dates, observation times, strikes, and interpolation policy.
They are not broad claims about differentiating with respect to knot locations
or changing the interpolation topology.

For piecewise-linear or gridded objects, derivatives with respect to query
location can be piecewise-defined and can change at knots. Trellis therefore
treats query-location derivatives as outside the ordinary node-risk contract
unless a lane explicitly says otherwise. This is why grid-vol European option
AAD can report node-vol risk, while grid-vol early-exercise or grid-vol
path-summary derivatives remain planned or unsupported.

Directional Operators
---------------------

Trellis exposes several derivative operators through
``trellis.core.differentiable``. The important ones for the current
implementation are gradients, Jacobians, VJPs, and scalar-objective HVPs.

JVP
~~~

A Jacobian-vector product pushes a tangent direction forward:

.. math::

   \operatorname{JVP}_F(\theta, v) = J_F(\theta) v

This answers: "if the input coordinates move in direction :math:`v`, how does
the output move?"

JVP is natural for forward-mode AD and small input dimension. Trellis does not
currently claim checked JVP support for pricing primitives. The active backend
reports ``jvp=False`` because stock ``autograd.make_jvp`` does not provide
checked forward-mode coverage for pricing primitives Trellis uses, including
``norm.cdf``. JVP requests in bounded hybrid AD therefore fail closed.

VJP
~~~

A vector-Jacobian product pulls an output weight vector backward:

.. math::

   \operatorname{VJP}_F(\theta, w) = J_F(\theta)^\top w

This answers: "what input-coordinate sensitivity corresponds to this weighted
output?"

For scalar valuation :math:`V(\theta)`, the VJP with weight :math:`1` is the
gradient:

.. math::

   \operatorname{VJP}_V(\theta, 1) = \nabla_\theta V

For a vector residual :math:`r(\theta)`, a scalar least-squares objective can
be written as:

.. math::

   L(\theta) = \frac{1}{2} r(\theta)^\top W r(\theta)

and its gradient is:

.. math::

   \nabla_\theta L(\theta) =
   J_r(\theta)^\top W r(\theta)

when :math:`W` is symmetric. If a nonsymmetric weighting matrix were used, the
gradient would involve :math:`(W + W^\top) / 2`. In calibration practice,
:math:`W` is normally diagonal or symmetric positive semidefinite, so the VJP
form above is the intended contract.

This is exactly a VJP with output weight :math:`w = W r(\theta)`. That is why
reverse-mode AD is valuable for risk and calibration: it avoids materializing a
dense Jacobian when the consumer only needs a weighted pullback.

HVP
~~~

A Hessian-vector product applies the Hessian of a scalar objective to a
direction:

.. math::

   \operatorname{HVP}_V(\theta, v) =
   \nabla^2_\theta V(\theta) v

The current backend supports scalar-objective HVP through a checked
reverse-over-reverse construction. Trellis uses HVP directionally rather than
building full Hessians for broad books. Full Hessian matrices can be useful for
small diagnostics, but they are not the scalable default for pricing stacks
with many risk factors.

Cost Model
~~~~~~~~~~

For a scalar value :math:`V : D \subset \mathbb{R}^n \rightarrow \mathbb{R}`,
finite-difference risk estimates need one or two extra valuations per
coordinate, depending on whether one-sided or central differences are used:

.. math::

   \frac{\partial V}{\partial \theta_j}
   \approx
   \frac{V(\theta + h e_j) - V(\theta - h e_j)}{2h}

That is :math:`O(n)` pricing calls for one trade or book. Reverse-mode AD has
a different cost profile: it evaluates the primal computation and then pulls a
single scalar adjoint backward to all active input coordinates. The constant
factor depends on the route and backend, but the scaling is attractive when
there are many input coordinates and one scalar output.

For vector outputs :math:`F : \mathbb{R}^n \rightarrow \mathbb{R}^m`, the best
operator depends on the consumer. If a solver needs the full dense
:math:`m \times n` Jacobian, dense Jacobian construction may be appropriate for
small :math:`m` and :math:`n`. If a solver or risk report only needs
:math:`J_F^\top w`, VJP is usually the better primitive.

AAD As Reverse-Mode Book Risk
-----------------------------

In quant finance, AAD usually means reverse-mode sensitivity calculation over a
pricing computation. In Trellis, the bounded portfolio-AAD lanes use that idea
but wrap it in explicit product, factor, and support contracts.

For a book with trade values :math:`V_i(\theta)` and signed quantities or
notionals :math:`q_i`, the book value is:

.. math::

   B(\theta) = \sum_i q_i V_i(\theta)

The book gradient is:

.. math::

   \nabla_\theta B(\theta)
   =
   \sum_i q_i \nabla_\theta V_i(\theta)

A bump/reprice implementation estimates each coordinate separately:

.. math::

   \frac{\partial B}{\partial \theta_j}
   \approx
   \frac{B(\theta + h e_j) - B(\theta - h e_j)}{2h}

That costs roughly one or two extra book reprices per factor, depending on the
finite-difference stencil. If a book has many factors, this becomes expensive
and can introduce finite-difference noise.

AAD instead evaluates the supported pricing computation once and pulls the
adjoint weights backward to the factor coordinates. More precisely, for each
supported lane Trellis builds a trace-safe scalar or vector map over the
lane's admitted coordinates, applies the backend VJP/HVP operator, and then
converts the resulting positional derivative into ``RiskFactorId`` keyed
entries. For vector-valued trade outputs:

.. math::

   y_i = F_i(\theta)

and a scalar weighted contribution:

.. math::

   L_i = w_i^\top y_i

the trade contribution is:

.. math::

   \nabla_\theta L_i = J_{F_i}(\theta)^\top w_i

That is a VJP. In Trellis, supported AAD lanes compute these pullbacks and
return sparse factorized risk:

.. math::

   R[f] = \sum_i q_i R_i[f]

where :math:`f` is a ``RiskFactorId``. This is the mathematical basis of
``SparseRiskVector`` aggregation.

This also explains a subtle but important implementation decision: Trellis does
not expose one global mutable pricing tape for arbitrary books. It exposes
bounded adapters whose coordinate families, product semantics, and unsupported
policy are explicit. The result is less general than a bank-wide industrial
AAD plant, but it is auditable and incrementally extensible.

Implementation Contract For Portfolio AAD
-----------------------------------------

The implementation is organized around a few durable objects:

``RiskFactorRegistry``
   Discovers supported coordinates on market and model objects. It can produce
   executable coordinates, such as ``YieldCurve`` zero-rate nodes for the
   bond-book lane, and discovery-only coordinates for future adapters.

``SparseRiskVector``
   Stores sensitivities keyed by ``RiskFactorId``. This is the canonical
   derivative payload for AAD-style risk.

``PortfolioAADRequest``
   Carries factor selection and unsupported-position policy. A request may ask
   for only some factors, and the result must report selected factors that were
   absent.

``PortfolioAADResult``
   Carries portfolio value, sparse risk, factor coordinates, unsupported
   positions, method metadata, diagnostics, and optional aggregation payloads.

``UnsupportedAADPosition``
   Records a position that is not admitted into AAD risk. Unsupported positions
   are not silently finite-differenced inside an AAD result.

The current bounded portfolio-AAD implementation supports explicit lanes rather
than a universal book compiler:

- shared-curve bond books
- European vanilla option books over shared ``FlatVol``
- European vanilla option books over shared ``GridVolSurface`` node grids
- bounded vanilla American/Bermudan option books over ``FlatVol`` under a
  smooth-interior hard exercise-projection policy
- bounded arithmetic-average Asian option books over ``FlatVol`` under a
  smooth path-summary policy
- scalar quanto-correlation books
- explicitly configured mixed supported books

Each adapter owns the pricing map it differentiates and the factor family it
returns. The mixed-book dispatcher aggregates only lanes that were explicitly
enabled by the caller's market context. This makes the support boundary visible
at runtime.

Bucket Aggregation
------------------

Risk reporting often needs buckets rather than raw coordinates. If
:math:`g = \nabla_\theta B` is the low-level sparse gradient and :math:`A` is a
factor-by-bucket incidence or weighting matrix, then bucket risk is:

.. math::

   b = A^\top g

In Trellis this is represented by ``RiskAggregationMap`` and
``risk_bucket_totals``. Examples include:

- curve zero-rate nodes aggregated into key-rate tenors
- volatility nodes aggregated by expiry, strike, or reporting surface
- correlation coordinates aggregated by factor pair
- credit coordinates aggregated by issuer or tenor

The important point is that aggregation is a reporting map over already-named
coordinates. It is not allowed to erase coordinate identity before the risk
vector is produced.

Semantic Admission Before AAD
-----------------------------

Trellis does not let every product shape enter an AAD lane just because some
pricing function exists. Product semantics are checked first.

``trellis.analytics.admit_portfolio_aad_lane(...)`` classifies
``ContractIR`` and ``DynamicContractIR`` requests as supported, planned, or
unsupported for portfolio AAD. The admission payload records:

- the product shape being considered
- the derivative lane requested
- the required market-coordinate family
- the support decision
- the reason for planned or unsupported status

This is a guardrail against accidental widening. For example, a terminal
European vanilla option over grid-vol nodes can be admitted to the bounded
grid-vol option lane. A discontinuous barrier monitor cannot be admitted to
ordinary pathwise AAD unless a checked smoothing, custom-adjoint, or
alternative estimator exists.

Discontinuities And Exercise Boundaries
---------------------------------------

AD computes ordinary derivatives of the implemented computation. That is not
the same as saying every financial payoff has a useful derivative everywhere.

A digital-style payoff:

.. math::

   \phi(S_T) = \mathbf{1}_{S_T > K}

has a distributional derivative, not an ordinary pathwise derivative at the
strike. A barrier monitor has event discontinuities. Early exercise creates a
control boundary where the exercise decision may change under a perturbation.

Trellis therefore requires a derivative policy for discontinuous or
piecewise-smooth structures. The allowed policies are:

- analytical derivative, when a checked formula exists
- ordinary AD, only on a supported smooth region
- finite-difference bump/reprice, with metadata saying it was a fallback
- governed smoothing or custom adjoint, once implemented and tested
- fail-closed unsupported status

For early-exercise options, the current supported AAD and hybrid AD lanes are
bounded smooth-interior lanes. They differentiate the value conditional on a
stable exercise projection or control policy. They do not differentiate the
``argmax`` or re-solve a changing exercise boundary inside the adjoint. The
exercise projection must remain stable under the configured tolerance.
Exercise-boundary ties fail closed rather than reporting unstable adjoints.

For Monte Carlo pathwise derivatives, the analogous rule is that randomness is
held fixed by explicit shocks. Trellis differentiates the deterministic map
from inputs and shocks to payoff when the path payoff is smooth enough. It does
not claim pathwise derivatives for discontinuous stopping or barrier events
unless a specific smoothing or custom estimator has been admitted.

Hybrid AD: Factor Graphs And Coordinate Charts
----------------------------------------------

Hybrid AD in Trellis means graph-owned derivatives over a bounded hybrid route.
It is separate from book-level portfolio AAD. The core object is
``HybridFactorGraph``:

- it records the market objects and model parameters visible to the route
- each differentiable coordinate has a ``MarketObjectCoordinateChart``
- unsupported dependencies remain visible in diagnostics
- semantic admission can be attached to the runtime derivative request

For the bounded terminal quanto route, the graph can expose scalar coordinates:

- underlier spot
- FX spot
- domestic and foreign curve zero-rate nodes
- flat or grid volatility nodes
- scalar underlier/FX correlation

The scalar correlation chart can be constrained directly as :math:`\rho`, or
represented by an unconstrained coordinate :math:`x`:

.. math::

   \rho = \tanh(x)

The chain rule controls how the reported sensitivity changes under the chart:

.. math::

   \frac{\partial V}{\partial x}
   =
   \frac{\partial V}{\partial \rho}
   \frac{\partial \rho}{\partial x}
   =
   \frac{\partial V}{\partial \rho}
   (1 - \tanh^2(x))

This is why the coordinate chart is part of the risk contract. A derivative
with respect to constrained :math:`\rho` is not numerically the same as a
derivative with respect to unconstrained :math:`x`.

Correlation Matrix Coordinates
------------------------------

The matrix-coordinate hybrid lane treats a correlation matrix as a checked
coordinate chart. The request must provide a finite symmetric unit-diagonal
matrix with bounded entries and enough positive-semidefinite margin:

.. math::

   C = C^\top,\quad C_{ii}=1,\quad -1 \le C_{ij} \le 1,
   \quad \lambda_{\min}(C) > \epsilon

For a factor pair :math:`(a,b)`, the active off-diagonal coordinate is:

.. math::

   c_{ab} = C_{ab} = C_{ba}

The current executable lane differentiates the terminal quanto route with
respect to direct off-diagonal matrix coordinates away from the PSD boundary.
It does not project, repair, or smooth invalid matrices. Requests near the PSD
boundary, projected/repaired charts, and correlation surfaces fail closed.

Bounded Hybrid VJP And HVP
--------------------------

For a hybrid route value:

.. math::

   V = V(\theta_G)

where :math:`\theta_G` are graph-owned coordinates, Trellis can compute:

.. math::

   \operatorname{VJP}_V(\theta_G, 1) = \nabla_{\theta_G} V

and, for supported scalar-objective HVP lanes:

.. math::

   \operatorname{HVP}_V(\theta_G, v)
   =
   \nabla^2_{\theta_G} V \, v

The HVP direction :math:`v` is sparse and keyed by ``RiskFactorId``. Missing or
empty HVP directions fail closed. This is important because a directional
second derivative has no meaning unless its coordinate basis is explicit.

Current bounded hybrid derivative entrypoints include:

- ``differentiate_quanto_scalar_correlation(...)``
- ``differentiate_quanto_scalar_inputs(...)``
- ``differentiate_quanto_correlation_matrix(...)``
- ``differentiate_arithmetic_asian_path_summary(...)``
- ``differentiate_vanilla_early_exercise(...)``

Those entrypoints report lane-specific derivative metadata such as
``hybrid_scalar_vector_vjp``, ``hybrid_scalar_vector_hvp``,
``hybrid_matrix_vector_vjp``, ``hybrid_matrix_vector_hvp``,
``hybrid_path_summary_vjp``, and ``hybrid_early_exercise_vjp``. They do not
change the global backend capability flag to "universal hybrid AD".

Path Summary And Early Exercise Hybrid Lanes
--------------------------------------------

The path-summary hybrid lane is intentionally narrow. Arithmetic-average Asian
options are admitted only through the bounded smooth path-summary policy. The
current executable lane differentiates one graph-owned ``FlatVol`` coordinate
and reports ``hybrid_path_summary_vjp`` metadata. Grid-vol path summaries,
non-arithmetic summaries, discontinuous event monitors, dynamic state, HVP, and
JVP fail closed. The admission layer can still identify a grid-vol
path-summary request: it records a planned ``grid_node_vols`` volatility
requirement and a planned smooth path-summary state policy, but no runtime
helper is attached until the coordinate policy and verification exist. The
coordinate policy is a discovery-only
``grid_vol_state_control_policy`` chart carrying the active node keys,
interpolation basis, locality policy, and selected-factor behavior.

The early-exercise hybrid lane is also intentionally narrow. Vanilla
American/Bermudan call/put contracts over one ``FlatVol`` coordinate can be
admitted under the hard exercise-projection smooth-interior policy. Grid-vol
early exercise, boundary ties, HVP, and JVP fail closed. Grid-vol
early-exercise admission records the requested node-vol parameterization and a
planned hard-exercise-projection control policy so downstream runtime code can
fail closed with the same state/control contract instead of silently widening
support. The same chart family records typed unsupported-dependency reasons
for missing surfaces, unsupported interpolation, unsupported selected factors,
event monitors, and exercise-boundary kinks.

These lanes exist because they are mathematically defensible and testable
within a bounded smooth region. They are not broad pathwise AD for arbitrary
stateful contracts.

Multi-Product Hybrid Fixtures
-----------------------------

The multi-product Hybrid AD surface composes lane-local results; it does not
differentiate one global multi-product program. Each product lane first
produces a ``HybridDerivativeResult`` with its own graph, sparse risk vector,
semantic admission metadata, derivative-method metadata, unsupported
dependencies, and diagnostics. The multi-product wrapper then records:

- one ``HybridADMultiProductLaneResult`` per lane
- the requested derivative method and product family for that lane
- the lane quantity used to scale value and risk
- the lane-local ``HybridDerivativeResult`` payload
- any lane-local semantic admission payload

For supported VJP lanes, aggregate risk is the sparse sum:

.. math::

   R_{\mathrm{agg}}[f]
   =
   \sum_{\ell \in L_{\mathrm{supported}}}
   q_\ell R_\ell[f]

where :math:`f` is a ``RiskFactorId`` and :math:`q_\ell` is the lane quantity.
This is ordinary linear aggregation of already-computed adjoints, not a new
cross-lane adjoint.

Unsupported lanes remain visible through ``unsupported_lane_diagnostics``. A
permissive request can use ``unsupported_lane_policy="collect_supported"`` to
return supported aggregate risk while listing unsupported JVP, event-monitor,
dynamic-state, missing-chart, or other fail-closed lanes. A strict request can
use ``unsupported_lane_policy="fail_closed"``; in that case the aggregate value
and aggregate risk are unavailable when any unsupported lane is present. In
both cases, lane-local metadata still records the requested backend operator,
the resolved derivative method, and whether a backend operator actually ran.

Runtime Metadata Mirrors The Mathematics
----------------------------------------

Every derivative result should say what actually happened. Trellis uses runtime
metadata to distinguish:

- analytical formulas
- ordinary autograd scalar gradients
- backend VJP
- scalar-objective HVP
- bounded portfolio AAD
- bounded hybrid graph-backed VJP/HVP
- finite-difference bump/reprice fallbacks
- fail-closed unsupported requests

For example, a bounded hybrid JVP request does not run a fake JVP. It returns
``resolved_derivative_method="unsupported_hybrid_jvp"`` with
``fallback_reason.code="hybrid_jvp_backend_unsupported"`` and includes the
backend ``support_matrix`` record showing ``jvp=False``. The payload carries
``requested_backend_operator="jvp"`` because that is what the caller asked for,
but it does not carry ``backend_operator="jvp"`` because no executable JVP
operator ran.

Verification Strategy
---------------------

The methodology is only useful if tests defend it. Trellis uses several kinds
of checks:

Finite-difference parity
   Small fixtures compare VJP or HVP outputs against independent bump/reprice
   or central-difference checks. This is used heavily for bounded AAD and
   hybrid AD lanes.

Dense-Jacobian parity
   Small vector problems can compare sparse VJP aggregation against dense
   Jacobian construction to make sure coordinate ordering and weights are
   correct.

Fail-closed tests
   Unsupported JVP, discontinuous events, invalid correlation matrices,
   missing HVP directions, unsupported product shapes, and exercise-boundary
   ties are tested as first-class behavior.

Metadata tests
   Derivative-method metadata is tested so users and audit consumers can see
   whether a result came from AD, AAD, finite differences, or fail-closed
   unsupported behavior.

Benchmark-shape tests
   Local benchmark fixtures measure the bounded portfolio-AAD lanes against
   deterministic bump/reprice baselines. These benchmarks are evidence for the
   declared lanes, not a claim of industrial-scale universal AAD.

How To Read Trellis AD Results
------------------------------

When inspecting an AD or AAD result, read it in this order:

1. Check the resolved derivative method. It tells you what lane ran.
2. Check the factor coordinates. They tell you what the derivative is with
   respect to.
3. Check unsupported positions or dependencies. They tell you what was excluded
   from risk.
4. Check selected-factor diagnostics. They tell you whether requested factors
   were absent, unavailable, unsupported, or zero.
5. Check fallback metadata. It tells you whether the result used true AD/AAD,
   finite differences, or fail-closed unsupported behavior.

This order mirrors the implementation. Trellis does not ask a quant to infer
the derivative basis from route-local naming or hidden market-object identity.

What Trellis Does Not Claim
---------------------------

The current implementation does not claim:

- universal portfolio AAD
- universal hybrid AD
- supported JVP for pricing primitives
- pathwise AD through discontinuous barrier/event monitors
- stable adjoints at exercise-boundary ties
- correlation-surface AD
- correlation-matrix projection or PSD-boundary derivatives
- grid-vol early-exercise AD
- broad dynamic-state hybrid AD
- industrial-scale mixed-exotic book risk

Those are future implementation problems, not documentation omissions. The
current docs should make the boundary legible so future widening happens
through checked tickets, not accidental implication.

Implementation Map
------------------

.. list-table::
   :header-rows: 1

   * - Mathematical object
     - Trellis implementation
     - Notes
   * - Coordinate :math:`\theta_j`
     - ``RiskFactorId`` plus coordinate payload
     - Stable identity for reporting and aggregation
   * - Coordinate discovery
     - ``RiskFactorRegistry``
     - Discovers executable and discovery-only factors
   * - Sparse gradient
     - ``SparseRiskVector``
     - Omits zero entries and keys values by ``RiskFactorId``
   * - Book result
     - ``PortfolioAADResult``
     - Value, sparse risk, coordinates, unsupported positions, metadata
   * - Book request
     - ``PortfolioAADRequest``
     - Factor selection and unsupported-position policy
   * - Bucket map :math:`A`
     - ``RiskAggregationMap``
     - Maps low-level factors to reporting buckets
   * - Graph-owned hybrid coordinates
     - ``HybridFactorGraph`` and ``MarketObjectCoordinateChart``
     - Makes chart transforms and dependencies explicit
   * - Product-shape admission
     - ``admit_portfolio_aad_lane(...)`` and ``admit_hybrid_ad_lane(...)``
     - Classifies supported, planned, and unsupported lanes
   * - Backend truth
     - ``get_backend_capabilities()`` and ``operator_support(...)``
     - Reports executable operator support, including fail-closed JVP
   * - Runtime derivative classification
     - derivative-method metadata in result payloads
     - Distinguishes AD, AAD, HVP, finite-difference, and unsupported lanes
   * - Multi-product hybrid fixture
     - ``HybridADMultiProductResult``
     - Linear sparse-risk aggregation over lane-local ``HybridDerivativeResult``
       values plus structured unsupported-lane diagnostics

Practical Rule For New Lanes
----------------------------

A new AD or AAD lane should be added only when all of these are true:

1. The mathematical derivative is meaningful on the declared domain.
2. The differentiable coordinates are named with stable ``RiskFactorId``
   payloads.
3. The pricing path is trace-safe over those coordinates.
4. The backend operator needed by the lane is checked and executable.
5. Unsupported shapes are rejected before runtime derivatives execute.
6. Independent finite-difference or dense-Jacobian verification exists for a
   small representative fixture.
7. Runtime metadata states the derivative method that actually ran.
8. Documentation states the support boundary and the fail-closed cases.

This is the methodology behind the current Trellis implementation: use AD and
AAD where they make pricing and risk better, but make every derivative claim
auditable, coordinate-owned, and support-contract-correct.
