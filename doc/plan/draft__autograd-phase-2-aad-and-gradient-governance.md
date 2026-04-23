# Autograd Phase 2 AAD And Gradient Governance

## Status

Draft execution mirror for the post-`QUA-957` autograd Phase 2 program.

Status mirror last synced: `2026-04-23`

## Linked Context

- `QUA-966` Autograd Phase 2: portfolio AAD and gradient governance
- `QUA-967` Autograd backend: JVP VJP and HVP operator implementation
- `QUA-968` Portfolio AAD: book-level reverse-mode sensitivity substrate
- `QUA-969` Discontinuous Greeks: smoothing and custom-adjoint policy
- `QUA-970` Gradient matrix: product-family autograd regression cohort
- `QUA-971` Runtime derivatives: expanded method selection and reporting
- `QUA-957` Autograd platform: public-contract and self-learning closure
- `QUA-946` Calibration sleeve: Trellis-native industrial hardening program
- `docs/quant/differentiable_pricing.rst`
- `LIMITATIONS.md`

## Linear Ticket Mirror

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for the Phase 2 autograd program.
- Do not mark a row `Done` here before the matching Linear ticket is actually
  closed.
- Keep this mirror aligned with `docs/quant/differentiable_pricing.rst` and
  `LIMITATIONS.md` whenever the support contract changes.
- Treat `QUA-946` as the adjacent calibration industrialization program rather
  than duplicating curve/surface/cube plant work here.
- For cross-program pickup order, use the combined implementation queue in
  `doc/plan/draft__calibration-sleeve-industrial-hardening-program.md`.

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-966` Autograd Phase 2: portfolio AAD and gradient governance | Done |

### Ordered Queue

| Queue ID | Linear | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- | --- |
| `AD2.1` | `QUA-967` | Done | JVP, VJP, HVP operator implementation or checked backend decision | `QUA-957`, `QUA-965` |
| `AD2.2` | `QUA-968` | Done | book-level reverse-mode / portfolio AAD substrate | `QUA-967` |
| `AD2.3` | `QUA-969` | Done | smoothing and custom-adjoint policy for discontinuous products | `QUA-957` |
| `AD2.4` | `QUA-970` | Done | product-family gradient matrix and support-contract cohort expansion | `QUA-957`; consume `QUA-967` / `QUA-969` outcomes as they land |
| `AD2.5` | `QUA-971` | Done | runtime derivative-method taxonomy and reporting integration | `QUA-967`, `QUA-970` |

## Purpose

`QUA-957` closed the contract-level autograd gap: supported smooth pricing
routes can now preserve traced values through the public computational surface.
Phase 2 is the next layer. It should make derivative computation scalable,
governed, and broad enough for self-learning and book-level workflows without
claiming universal differentiability.

The main distinction is:

- Phase 1 made the pricing map trace-safe where the mathematics is smooth.
- Phase 2 decides how Trellis computes, reports, and validates derivatives when
  scale, backend operator choice, non-smooth products, and product-family
  coverage become first-order concerns.

## End State

The desired Phase 2 end state is:

1. Backend operators are explicit and truthful.
   `get_backend_capabilities()` should report real support for `jvp`, `vjp`,
   and `hessian_vector_product` only when those functions compute checked
   values.
2. Portfolio sensitivities have a throughput-oriented path.
   A bounded supported book should be able to compute a risk vector without
   central-bumping every trade and every risk factor.
3. Discontinuous derivatives are governed.
   Barriers, digitals, and exercise/event logic should either expose a checked
   smoothing/custom-adjoint/alternative estimator policy or fail/fall back with
   explicit metadata.
4. The derivative support matrix is test-backed.
   Product-family coverage should be visible from a checked gradient matrix,
   not inferred from scattered examples.
5. Runtime derivative reporting uses one taxonomy.
   Analytical, autograd, AAD, JVP/VJP/HVP-backed, smoothed/custom-adjoint,
   finite-difference, and unsupported lanes should be reported consistently.

## Mathematical Direction

### Directional Operators

The current backend supports scalar gradients, dense Jacobians, dense Hessians,
VJP, and scalar-objective HVP. JVP remains fail-closed because stock
`autograd` lacks the required forward-mode coverage for pricing primitives such
as `norm.cdf`. Phase 2 should use directional operators where they are
mathematically and computationally useful:

.. math::

   \operatorname{JVP}_f(x, v) = J_f(x) v

.. math::

   \operatorname{VJP}_f(x, w) = J_f(x)^\top w

.. math::

   H_f(x) v

These operators are the natural bridge from route-level AD to book-level and
calibration workflows because they avoid materializing dense derivative objects
when only directional sensitivities are needed.

### Portfolio AAD

For a book with trade values :math:`V_i(\theta)` and weights or notionals
:math:`q_i`, the book value is:

.. math::

   B(\theta) = \sum_i q_i V_i(\theta)

The desired first derivative is:

.. math::

   \nabla_\theta B(\theta) = \sum_i q_i \nabla_\theta V_i(\theta)

The Phase 2 question is not only whether this derivative exists. It is how to
compute it without scaling linearly through repeated bump/reprice loops across
every trade and risk factor.

`QUA-968` landed the first bounded answer: supported bond books on a shared
public `YieldCurve` can compute reverse-mode curve risk through
`trellis.book.portfolio_aad_curve_risk(...)`, with unsupported positions
reported explicitly. This is not universal portfolio AAD; broader books,
non-smooth routes, and richer risk vectors remain follow-on work.

### Discontinuities

For discontinuous payoffs, the pathwise derivative may not exist in the
ordinary sense:

.. math::

   \phi(x) = \mathbf{1}_{x > K}

or may be dominated by boundary events. Trellis should not silently pretend
that ordinary autograd solves this. A route must choose one of:

- no derivative support
- explicit finite-difference fallback
- governed smoothing with documented smoothing parameter
- custom adjoint or alternative estimator with a reference test
- analytical derivative where one exists

`QUA-969` landed the first bounded governed policy. Monte Carlo barrier
monitors plus barrier/exercise event replay now expose fail-closed
discontinuous derivative metadata, declare finite-difference bump/reprice as
the fallback method, and report metadata from the executed pricing branch
rather than from unused storage-policy requests. This is a policy and reporting
slice, not automatic discontinuous Greeks or universal smoothing/custom-adjoint
support.

## Relationship To Calibration Industrialization

`QUA-946` owns the broader industrial curve, surface, and cube calibration
plants. This Phase 2 autograd program should not duplicate that backlog.
Instead:

- `QUA-946` builds stronger calibrated market objects and fixtures.
- `QUA-966` defines how derivative operators, runtime reporting, and portfolio
  sensitivity workflows consume those objects.

## Acceptance Standard

The Phase 2 program should be considered complete when:

1. backend operator capabilities are checked by tests and accurately reported
2. one bounded portfolio AAD path exists with provenance and benchmark evidence
3. discontinuous derivative policy is implemented for at least one bounded
   product family
4. a product-family gradient matrix guards the public support contract
5. runtime derivative metadata uses one documented taxonomy across new and
   existing derivative lanes

## Residual Risks After Phase 2

Even after this program, Trellis may still need:

- a higher-performance backend for very large portfolios
- GPU or distributed derivative execution
- broader route-specific custom adjoints
- industrial market-data vendor integration for large calibrated surfaces
- wider self-learning benchmarks that synthesize new differentiable product
  families from scratch
