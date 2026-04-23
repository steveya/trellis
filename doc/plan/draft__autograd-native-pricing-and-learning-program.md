# Autograd-Native Pricing And Learning Program

## Status

Draft execution mirror for the autograd end-state umbrella `QUA-957`.

Status mirror last synced: `2026-04-22`

## Linked Context

- `QUA-957` Autograd platform: public-contract and self-learning closure
- `QUA-958` Autograd contract: trace-safe payoff and public value boundary
- `QUA-959` Autograd market objects: trace-safe curves, forwards, and vol surfaces
- `QUA-960` Risk runtime: capability-driven autograd, analytic, and bump selection
- `QUA-961` Monte Carlo: pathwise gradients for reduced-state and event-aware payoffs
- `QUA-962` Calibration derivatives: systemic Jacobian support across bootstrap, SABR, and Heston
- `QUA-963` Autograd verification: gradient cohort, support-contract audit, and limitation cleanup
- `QUA-964` Self-learning autograd: prompts, cookbooks, and docs align to the public contract
- `QUA-965` Autograd substrate: backend capability surface and future AAD hooks
- `ARCHITECTURE.md`
- `LIMITATIONS.md`
- `docs/quant/differentiable_pricing.rst`
- `docs/quant/extending_trellis.rst`
- `docs/quant/analytical_route_cookbook.rst`

## Linear Ticket Mirror

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for the autograd-native program.
- Do not mark any row `Done` here before the matching Linear ticket is actually
  closed.
- Keep this file synchronized with `LIMITATIONS.md`, the differentiable-pricing
  docs, and the builder prompt surface as the support contract moves.

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-957` Autograd platform: public-contract and self-learning closure | Backlog |

### Delivery Slices

| Ticket | Status | Scope |
| --- | --- | --- |
| `QUA-958` | Backlog | Public payoff contract and explicit float-only serialization boundary |
| `QUA-959` | Done | Trace-safe `YieldCurve` / `CreditCurve` / `GridVolSurface` and date-aware helpers |
| `QUA-960` | Done | Capability-driven risk-method selection and sensitivity provenance |
| `QUA-961` | Done | Reduced-state and event-aware Monte Carlo pathwise gradients |
| `QUA-962` | Done | Systemic calibration Jacobians and derivative-method provenance |
| `QUA-964` | Done | Builder prompts, cookbooks, and docs aligned to the autograd-native contract |
| `QUA-963` | Backlog | Gradient regression cohort, support-contract audit, and limitation cleanup |
| `QUA-965` | Backlog | Backend capability surface and future AAD hooks |

## Purpose

This plan defines the end-state needed for Trellis to become autograd-native in
the computational layer rather than merely autograd-capable in isolated raw
kernels.

The immediate goal is not to force every numerical method into one traced path.
The goal is to make the public pricing stack behave coherently:

1. if a pricing route is mathematically smooth, the same public route should
   support price and derivatives without bespoke sidecar wrappers
2. if a route is not smooth, Trellis should say so explicitly and choose an
   analytical or bump fallback intentionally
3. the builder / self-learning layer should emit code that preserves that same
   contract automatically

This is the architectural prerequisite for the next self-learning phase. Until
the public contract is trace-safe, generated mathematical tools can use the
right backend and still collapse the gradient at the adapter boundary.

## External Baseline: FinancePy

The upstream FinancePy codebase was rechecked directly on `2026-04-21` at
commit `2e8528a14fb5415a70e5e92aff5272f9bf64c98b`.

The important takeaway is that FinancePy is **not** an autograd-first library.
It is primarily:

- analytical Greeks where closed forms exist
- bump/revalue and shock engines elsewhere
- Numba-accelerated numerical kernels
- finite-difference PDE infrastructure

That means the Trellis target should not be “match FinancePy’s autograd use.”
FinancePy mostly does not have one. Trellis should differentiate itself by
making the public pricing and learning contracts trace-safe where the
mathematics is smooth.

## Decision Summary

### 1. The public pricing map becomes the differentiable map

For supported smooth routes, the object we differentiate is the same object we
price through the public runtime:

.. math::

   V(\theta) = \operatorname{Price}(P, M(\theta))

where:

- :math:`P` is the product / payoff contract
- :math:`M(\theta)` is the market state or model parameterization
- :math:`\theta` is the chosen sensitivity parameter vector

The public contract should not collapse :math:`V(\theta)` to a Python `float`
before the caller has a chance to differentiate it.

### 2. Differentiability is capability-driven, not heuristic

Each route or measure should disclose whether it supports:

- analytical derivatives
- autograd derivatives
- bump / finite-difference fallback
- or no supported derivative path

The runtime should choose among those paths explicitly and report the method
used.

### 3. Public market objects are part of the differentiable surface

Curves and surfaces are not just data containers. They are part of the
computational graph:

- :math:`r(t; x)` for zero-rate curves parameterized by node vector :math:`x`
- :math:`\lambda(t; h)` for hazard curves parameterized by node vector :math:`h`
- :math:`\sigma(T, K; \Sigma)` for vol surfaces parameterized by node matrix
  :math:`\Sigma`

If these objects are not trace-safe, the pricing stack cannot be trace-safe.

### 4. Monte Carlo and calibration follow the same rule

Monte Carlo and calibration are not special exceptions. They should use the same
contract:

- smooth pathwise Monte Carlo stays inside the trace
- calibration objectives expose residual vectors and Jacobians where the model
  is smooth
- non-smooth cases fail or fall back explicitly

### 5. Prompts and docs must teach the same contract the runtime enforces

The current prompt and documentation stack still teaches the older
“raw-kernel-only AD, public adapter returns float” rule. That guidance was a
useful proving-step for the analytical-support program, but it is now the main
blocker to autograd-native generation.

## Mathematical End-State

### Smooth route contract

For a smooth pricing route with parameter vector :math:`\theta`, Trellis should
support:

- scalar gradient:

  .. math::

     \nabla_\theta V(\theta)

- Jacobian to vector or matrix outputs when the route naturally returns a vector
  of residuals or sensitivities

- second-order objects only where they are computationally defensible

This does **not** mean every route must expose Hessians. First derivatives are
the required baseline.

### Piecewise-linear market objects

For fixed query coordinates, piecewise-linear interpolation is affine in node
values inside the active cell. For example, if a query lies between two zero
rate nodes:

.. math::

   r(t; x) = (1-w)x_i + w x_{i+1}

then:

.. math::

   \frac{\partial r}{\partial x_i} = 1-w, \qquad
   \frac{\partial r}{\partial x_{i+1}} = w

The same logic applies to bilinear vol interpolation in node values. The
practical implication is:

- differentiability with respect to node values is the important target
- query-location differentiability is only piecewise valid away from knot
  boundaries
- Trellis should document that distinction explicitly rather than pretending the
  interpolation map is globally smooth

### Calibration objective structure

For a least-squares calibration:

.. math::

   r(\theta) = m(\theta) - q

.. math::

   L(\theta) = \tfrac{1}{2} r(\theta)^\top W r(\theta)

the useful derivative objects are:

.. math::

   J(\theta) = \frac{\partial r}{\partial \theta}

.. math::

   \nabla L(\theta) = J(\theta)^\top W r(\theta)

and the standard Gauss-Newton approximation:

.. math::

   H_{GN}(\theta) \approx J(\theta)^\top W J(\theta)

Trellis does not need a bespoke research optimizer to benefit from autograd
here. It needs consistent residual-vector and Jacobian exposure across the
shipped calibration workflows.

### Pathwise Monte Carlo contract

For deterministic shocks :math:`z_1, \dots, z_N`, the pathwise Monte Carlo
estimator is:

.. math::

   \hat{V}(\theta) =
   \frac{1}{N} \sum_{n=1}^{N}
   e^{-rT}\,\phi(X_T(\theta, z_n))

When the state evolution and payoff are smooth enough, the pathwise derivative
is:

.. math::

   \nabla_\theta \hat{V}(\theta) =
   \frac{1}{N} \sum_{n=1}^{N}
   e^{-rT}\,
   \nabla_x \phi(X_T(\theta, z_n))
   \frac{\partial X_T(\theta, z_n)}{\partial \theta}

The end-state does not require every Monte Carlo route to satisfy those
assumptions. It requires Trellis to know when the assumptions hold, preserve
the trace in those cases, and fail or fall back clearly otherwise.

### Non-smooth fallback policy

The target state is **not** “everything differentiates.” The target policy is:

- use analytical derivatives when the route already has them
- use autograd when the computational path is smooth and trace-safe
- use bump / finite difference when the route is intentionally non-smooth or the
  computational kernel is forward-only
- report the chosen method and the fallback reason

Digitals, hard barriers, discrete exercise policies, and branch-singular
comparators should remain explicit exceptions unless Trellis later introduces a
checked smoothing or adjoint policy for them.

## Current Gap Map

### A. Public contract scalarization

Main files:

- `trellis/core/payoff.py`
- `docs/quant/differentiable_pricing.rst`
- `docs/quant/extending_trellis.rst`
- `trellis/agent/prompts.py`

Current condition:

- `Payoff.evaluate(...)` is specified as `-> float`
- `ResolvedInputPayoff.evaluate(...)` wraps traced values in `float(...)`
- expiry handling also scalarizes
- docs and prompts still encode the public adapter as float-returning by design

Why this matters:

- autograd support lives on sidecar `*_raw` helpers rather than on the public
  computational contract
- self-learning is trained to break the trace at exactly the wrong boundary

### B. Market objects are not trace-safe

Main files:

- `trellis/curves/yield_curve.py`
- `trellis/curves/credit_curve.py`
- `trellis/curves/date_aware_flat_curve.py`
- `trellis/curves/forward_curve.py`
- `trellis/models/vol_surface.py`
- `trellis/curves/interpolation.py`

Current condition:

- `YieldCurve` and `CreditCurve` use `np.asarray(..., dtype=float)` in public
  constructors
- `GridVolSurface.black_vol(...)` ends in `float(...)`
- date-aware flat curves and forward helpers cast results back to Python floats

Current reproducible failures from the 2026-04-21 review:

- differentiating through `YieldCurve(...)` construction raised:
  `NotImplementedError: VJP of asarray wrt argnums (0,) not defined`
- differentiating through `GridVolSurface.black_vol(...)` raised:
  `TypeError: float() argument must be a string or a real number, not 'ArrayBox'`

Why this matters:

- runtime risk uses a private `_AutodiffDiscountCurve` shadow helper instead of
  the public `YieldCurve`
- realistic smile / surface sensitivity cannot stay on the traced path

### C. Runtime risk is fragmented and partially stale

Main files:

- `trellis/analytics/measures.py`
- `LIMITATIONS.md`
- `docs/quant/pricing_stack.rst`
- `docs/user_guide/pricing.rst`

Current condition:

- rate-risk autodiff uses a private discount-curve wrapper rather than the
  public curve object
- vega is truly traced only for `FlatVol`
- non-flat surfaces still fall back to bucket bumps or one representative
  scalar-vol proxy
- spot delta / gamma / theta exist, but the documented support contract is not
  consistently aligned

Important support-contract drift:

- `LIMITATIONS.md` still says delta / gamma / theta have no runtime
  implementation even though they are now implemented and documented elsewhere
- `LIMITATIONS.md` also still states the bootstrap Jacobian path too strongly in
  places that no longer match the checked code exactly

Why this matters:

- the runtime cannot honestly claim one derivative-selection policy
- users and future builders cannot tell whether a number came from analytics,
  autograd, or bump fallback

### D. Reduced-state Monte Carlo remains forward-only

Main files:

- `trellis/models/monte_carlo/engine.py`
- `trellis/models/monte_carlo/path_state.py`
- `trellis/models/monte_carlo/event_aware.py`
- `trellis/models/monte_carlo/event_state.py`

Current condition:

- `differentiable=True` is explicitly rejected for state-aware or
  reduced-storage payoff surfaces
- `StateAwarePayoff`, `terminal_value_payoff(...)`, barrier helpers, and
  event-aware replay functions coerce outputs through raw NumPy arrays with
  `dtype=float`

Current reproducible failure from the 2026-04-21 review:

- state-aware Monte Carlo under `differentiable=True` raised:
  `NotImplementedError: differentiable Monte Carlo currently requires a plain path payoff callable`

Why this matters:

- the reusable computational layer is still forward-only even where the demo
  full-path lane can differentiate

### E. Calibration derivative support is selective

Main files:

- `trellis/models/calibration/solve_request.py`
- `trellis/curves/bootstrap.py`
- `trellis/models/calibration/sabr_fit.py`
- `trellis/models/calibration/heston_fit.py`

Current condition:

- bootstrap exposes a Jacobian-oriented residual path
- SABR exposes a scalar-objective gradient
- Heston does not expose a Jacobian and therefore uses SciPy `2-point` on the
  `trf` path

Why this matters:

- solver behavior is inconsistent across supposedly similar calibration
  workflows
- derivative method is not yet a first-class part of solve provenance

### F. Backend abstraction is still thin

Main file:

- `trellis/core/differentiable.py`

Current condition:

- the backend exposes only `get_numpy`, `gradient`, `jacobian`, and `hessian`
- there is no capability surface for JVP / VJP / HVP or for future backend
  policy

Why this matters:

- the immediate public-contract program can still land, but later scaling work
  has no stable place to attach

### G. Self-learning guidance still encodes the older boundary

Main files:

- `trellis/agent/prompts.py`
- `docs/quant/differentiable_pricing.rst`
- `docs/quant/extending_trellis.rst`
- `docs/quant/analytical_route_cookbook.rst`

Current condition:

- prompts require `get_numpy()` but still ask the model to implement
  `evaluate(...) -> float`
- docs still say public adapters stay float-returning while raw kernels carry
  differentiation

Why this matters:

- the builder is backend-aware but not contract-aware
- new generated tools are likely to repeat the same scalarization pattern

### H. Verification and support-contract coverage are too thin

Main files:

- `tests/test_core/test_differentiable.py`
- `tests/test_curves/test_forward_curve.py`
- `tests/test_models/test_monte_carlo/test_mc.py`
- `tests/test_models/test_trees/test_trees.py`
- `tests/test_verification/test_greeks.py`

Current condition:

- some bounded autograd tests exist
- there is still little direct coverage for the public pricing boundary through
  `evaluate(...)`
- the forward-curve test already documents the constructor trace break instead
  of locking the desired public behavior

Why this matters:

- the support contract is still inferred from scattered tests and docs rather
  than defended explicitly

## Recommended Execution Order

The tickets were created to allow some parallelism, but the program still has a
natural implementation order:

1. `QUA-958` — reset the public value contract
2. `QUA-959` — make public market objects trace-safe
3. `QUA-964` — align prompts, docs, and cookbooks to the new contract
4. `QUA-960` — rebuild runtime risk on explicit derivative-method selection
5. `QUA-961` — widen Monte Carlo pathwise support beyond plain full-path callables
6. `QUA-962` — unify calibration derivative handling and provenance
7. `QUA-963` — close the loop with a real gradient cohort and support-contract cleanup
8. `QUA-965` — harden the backend boundary for future AAD scaling

## Phase Plan

### Phase 1 — Public Contract Reset (`QUA-958`)

Target outcome:

- the public payoff and deterministic runtime no longer force traced values into
  Python floats

Key decisions:

- keep float-only conversion only at explicit reporting / serialization
  boundaries
- let the public computational contract return a numeric scalar that may be a
  traced scalar in smooth AD workflows

Completion criteria:

- one public resolved-input payoff can be differentiated through
  `evaluate(...)` directly

### Phase 2 — Trace-Safe Market Objects (`QUA-959`)

Target outcome:

- public curves and surfaces can stay on the traced path

Key decisions:

- node-value differentiability is the required target
- query-location differentiability is only piecewise and should be documented
  that way

Completion criteria:

- yield, credit, and grid-vol node sensitivities work through the public
  objects without shadow wrappers

### Phase 3 — Generation And Documentation Realignment (`QUA-964`)

Target outcome:

- prompts, docs, and cookbook guidance teach the same contract that the runtime
  now enforces

Key decisions:

- replace the “float-returning public adapter” rule with the autograd-native
  public contract
- keep the raw-kernel pattern, but stop treating the public boundary itself as a
  forced scalarization point

Completion criteria:

- generated-route guidance preserves differentiability by default for supported
  smooth routes

### Phase 4 — Runtime Risk Method Selection (`QUA-960`)

Target outcome:

- runtime measures choose analytical, autograd, or bump paths explicitly and
  disclose which path was used

Key decisions:

- capability metadata belongs to the route / measure contract, not to ad hoc
  type branches
- fallback reason is part of the result provenance

Completion criteria:

- rate-risk and vega no longer depend on hidden shadow helpers or silent scalar
  proxies

### Phase 5 — Monte Carlo Pathwise Expansion (`QUA-961`)

Target outcome:

- deterministic pathwise gradients extend to supported reduced-state and
  event-aware Monte Carlo contracts

Key decisions:

- preserve traced values on the differentiable lane
- keep explicit rejections for non-smooth reducers or event semantics

Completion criteria:

- at least one reduced-state payoff and one smooth event-aware payoff admit
  pathwise differentiation with explicit shocks

### Phase 6 — Systemic Calibration Derivatives (`QUA-962`)

Target outcome:

- derivative support is consistent across smooth calibration workflows

Key decisions:

- solve provenance must record derivative method
- Heston should stop relying on implicit `2-point` fallback with no explicit
  surface-level disclosure

Completion criteria:

- bootstrap, SABR, and Heston all report derivative method and use the typed
  solve surface consistently

### Phase 7 — Verification And Support-Contract Cleanup (`QUA-963`)

Target outcome:

- the new differentiable contract is defensible from tests and docs alone

Key decisions:

- correct stale support-contract claims in `LIMITATIONS.md`
- lock a representative gradient cohort rather than one or two isolated proofs

Completion criteria:

- tests and support-contract docs agree on the shipped differentiable surface

### Phase 8 — Backend Extension Hooks (`QUA-965`)

Target outcome:

- the backend boundary is ready for future JVP / VJP / HVP / AAD work without
  changing the public pricing contract again

Completion criteria:

- `trellis/core/differentiable.py` describes backend capabilities explicitly and
  has focused coverage

## Acceptance Standard For The Whole Program

The autograd-native program should be considered complete when all of the
following are true:

1. A smooth public pricing route can be differentiated directly through the
   public computational contract.
2. Public curve and surface objects support node-value sensitivities without
   private shadow wrappers.
3. Runtime measures disclose whether their derivatives are analytical,
   autograd-based, or bump-based.
4. Reduced-state and event-aware Monte Carlo have an explicit smooth pathwise
   contract.
5. Calibration workflows report derivative method consistently and avoid hidden
   finite-difference fallback on smooth routes.
6. Prompts, cookbooks, and docs teach the same contract the deterministic
   runtime implements.
7. `LIMITATIONS.md` and the checked tests describe the support boundary
   accurately.

## Residual Risks After This Program

Even after `QUA-957` closes, Trellis will still have important follow-on work:

- large-book reverse-mode AAD throughput
- broader industrial curve / surface / cube programs
- smoothing or custom adjoint policy for discontinuous products
- performance work on forward-only compiled kernels versus differentiable lanes

Those are real next steps, but they are downstream of the contract-level work
in this plan. The first job is to make the public pricing and learning surfaces
mathematically honest about where autograd does and does not apply.
