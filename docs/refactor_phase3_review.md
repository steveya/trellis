# Phase 3 Review

## Scope Decision

Phase 3 stays narrow.

The only justified expansion candidate after Tranche 2 is `qmc`:

- usable substrate already exists in [`trellis.models.monte_carlo.variance_reduction`](../trellis/models/monte_carlo/variance_reduction.py) via `sobol_normals`
- Brownian-bridge construction already exists in [`trellis.models.monte_carlo.brownian_bridge`](../trellis/models/monte_carlo/brownian_bridge.py)
- task inventory already references QMC-style work in [`TASKS.yaml`](../TASKS.yaml)
- the Tranche 1 audit already classified QMC as `WRAP + REFACTOR`, not greenfield implementation

The remaining method families from the original prompt are still intentionally deferred:

- `mlmc`
- `bsde`
- `quadrature` as a dedicated family
- `quantization`
- `surrogate`

Those families do not yet have a coherent substrate in the repository, so adding them now would violate the anti-hallucination and minimal-churn rules.

## Review Findings

### Current role of touched modules

| Module | Previous role | New role | Reason for change |
| --- | --- | --- | --- |
| `trellis.models.monte_carlo.variance_reduction` | variance-reduction helpers, including Sobol normals | keeps low-level MC implementation helpers | preserve implementation; avoid moving stable code |
| `trellis.models.monte_carlo.brownian_bridge` | Brownian-bridge path constructor under MC package | keeps low-level path-construction implementation | preserve implementation; avoid churn |
| `trellis.models` | public hub for main family packages | add `qmc` as canonical accelerator family export | target taxonomy already treats QMC as separate from plain MC |
| `trellis.core.capabilities` | method capability inventory without QMC family | include QMC as an explicit accelerator family | make public capability surface match real code |
| `trellis.agent/knowledge/canonical/*` | canonical method metadata for existing families only | add truthful QMC cookbook/requirements/API references | launched agents need the same family map as the library |

### Constraints

- QMC should be documented as an accelerator family layered on Monte Carlo internals, not as a universal standalone pricing engine.
- Existing low-level imports must keep working.
- No new stochastic theory should be invented; only existing Sobol and Brownian-bridge support should be wrapped.

## Phase 3 Subphases

### 3A. QMC Family Surface

Add a dedicated `trellis.models.qmc` package that re-exports the existing Sobol and Brownian-bridge helpers as the canonical QMC surface while preserving the existing Monte Carlo implementation modules.

Acceptance criteria:

- `trellis.models.qmc` exists and exports `sobol_normals` and `brownian_bridge`
- `trellis.models` exports the `qmc` family package
- capability inventory includes `qmc`

### 3B. QMC Knowledge and Docs

Add QMC to the canonical method metadata and public docs without overstating support.

Acceptance criteria:

- canonical knowledge contains a `qmc` cookbook and method requirements
- codegen guardrails and method normalization recognize `qmc`
- API/docs explain that `trellis.models.qmc` is canonical and old low-level imports remain valid

## Validation Plan

Targeted tests first:

- public package surface
- capability inventory
- knowledge retrieval
- codegen guardrails

Then broader regressions:

- `tests/test_agent`
- `tests/test_models/test_generalized_methods.py`
- `tests/test_models/test_monte_carlo/test_mc.py`
- `tests/test_core/test_capabilities.py`
