# Tranche 1 Audit

Date: 2026-03-24

## Scope

This note records the baseline audit for the Trellis refactor/expansion prompt. It is intentionally conservative:

- audit what already exists before changing structure
- map current modules to the target pricing-method taxonomy
- classify the main surfaces as `KEEP`, `REFACTOR`, `WRAP`, `MERGE`, `DEPRECATE`, or `REMOVE`
- capture validation baselines and known blockers

This tranche does not change runtime behavior. It establishes the integration map for later tranches.

## Executive Summary

Trellis already contains a substantial pricing library and should not be rewritten from scratch. The highest-value work is structural normalization, not speculative capability expansion.

The repository already has meaningful coverage for:

- analytical pricing
- transform pricing
- finite-difference PDE pricing
- lattice / tree pricing
- Monte Carlo pricing
- exercise logic via LSM, tree backward induction, and PDE obstacle handling
- stochastic-process modeling
- calibration
- copula-based credit modeling
- structured cashflow modeling
- agent and knowledge-system guidance for method selection and code generation

The main architectural issue is not missing code. It is that multiple concerns are intermixed:

- numerical engine families
- stochastic model definitions
- calibration tools
- structured cashflow logic
- user-facing orchestration
- agent/knowledge metadata that hard-codes method names and module paths

The safest path is incremental:

1. normalize the existing families first
2. preserve compatibility with shims and re-exports
3. update the agent and knowledge system in lockstep with structural changes
4. add new method families only where there is already enough substrate to support them honestly

## Current Architecture

### Stable domain/core surfaces

- `trellis/core/`
  - `market_state.py` is the main market data container
  - `payoff.py` defines the payoff protocol
  - `types.py` defines shared protocols and result containers
- `trellis/conventions/`
  - calendars, schedules, day counts, rate indices
- `trellis/curves/`
  - yield, forward, credit, interpolation, bootstrap
- `trellis/data/`
  - market-data providers and resolver
- `trellis/instruments/`
  - reference instruments and payoff-like wrappers

### Orchestration and user-facing API

- `trellis/session.py`
  - immutable market snapshot
  - pricing facade
  - analytics facade
  - agent entry point
- `trellis/book.py`
  - portfolio container and aggregate results
- `trellis/engine/`
  - `payoff_pricer.py` is the general payoff-evaluation gateway
  - `pricer.py` is still bond-centric
  - `analytics.py` computes curve Greeks
- `trellis/__init__.py`
  - top-level convenience exports

### Numerical/modeling layer

- `trellis/models/analytical/`
- `trellis/models/transforms/`
- `trellis/models/pde/`
- `trellis/models/trees/`
- `trellis/models/monte_carlo/`
- `trellis/models/processes/`
- `trellis/models/calibration/`
- `trellis/models/copulas/`
- `trellis/models/cashflow_engine/`
- `trellis/models/black.py`
- `trellis/models/vol_surface.py`

### Agent and knowledge system

- `trellis/agent/`
  - planning, prompt construction, payoff generation, validation
- `trellis/agent/knowledge/`
  - canonical feature taxonomy
  - instrument decompositions
  - method requirements
  - cookbook templates
  - import registry
  - lessons and retrieval logic

### Legacy and stale surfaces

- `README.md` still describes the repo as `rate-model`
- `setup.py` still packages `rate-model`, while `pyproject.toml` packages `trellis`
- `rate_model/` is still present as an old compatibility-era package
- `docs/api/models.rst` still references legacy PDE entry points instead of the canonical theta-method surface
- `trellis/models/__init__.py` and `trellis/core/__init__.py` are empty, so package-level public surfaces are inconsistent

## Target Taxonomy Mapping

| Target family | Current status | Current Trellis modules | Tranche 1 judgment | Notes |
|---|---|---|---|---|
| Transform / semi-analytic | Implemented | `trellis.models.transforms`, `trellis.models.analytical`, CF-capable process modules | `KEEP` + `REFACTOR` | FFT and COS exist; analytical formulas exist; no Laplace package |
| PDE / PIDE | Implemented in 1D PDE form | `trellis.models.pde` | `KEEP` + `REFACTOR` | Theta-method, operators, PSOR exist; no public PIDE / ADI / FE / FV family |
| Lattice / tree | Implemented | `trellis.models.trees` | `KEEP` + `REFACTOR` | Binomial, trinomial, short-rate lattice, backward induction exist |
| Monte Carlo | Implemented | `trellis.models.monte_carlo` | `KEEP` + `REFACTOR` | Simulation, schemes, LSM, and variance reduction exist but are not cleanly separated |
| QMC | Partial | `trellis.models.monte_carlo.variance_reduction.sobol_normals`, `brownian_bridge.py` | `WRAP` + `REFACTOR` | Functional pieces exist; no dedicated conceptual API/package |
| MLMC | Absent | none | `DEFER` | Do not scaffold without a real estimator hierarchy |
| Exercise / control | Partial | `trellis.models.monte_carlo.lsm`, `trellis.models.trees.backward_induction`, PDE exercise hooks in `theta_method.py` | `REFACTOR` | Existing functionality should be surfaced more explicitly |
| BSDE / FBSDE | Absent | none | `DEFER` | No honest substrate yet |
| Deterministic quadrature / sparse-grid / cubature | Internal-only / absent as a family | internal quadrature usage in `trellis.models.copulas.factor` | `DEFER` | No standalone public package today |
| Quantization | Absent | none | `DEFER` | No honest substrate yet |
| Surrogate / scientific-ML | Absent in package surface | none in `trellis/` proper | `DEFER` | Keep absent until there is real code, tests, and docs |

## Classification Matrix

The table below records the current role, intended role, and why each significant surface should or should not change.

| Surface | Label | Previous role | Planned role | Reason for change |
|---|---|---|---|---|
| `trellis/core/market_state.py` | `KEEP` | canonical immutable market snapshot | remain canonical market-state container | already aligns with the target domain layer |
| `trellis/core/payoff.py` | `KEEP` | payoff protocol plus deterministic-cashflow adapter | remain canonical payoff abstraction | already separates financial abstraction from numerical method |
| `trellis/core/types.py` | `REFACTOR` | mixed protocols and mutable result containers | remain shared types/protocols module with stricter value semantics | value types are not consistently frozen despite repo guidance |
| `trellis/conventions/` | `KEEP` | domain conventions and scheduling | remain domain conventions layer | correct placement already |
| `trellis/curves/` | `KEEP` | domain curve abstractions and interpolation/bootstrap support | remain market-data/domain layer | correct placement already |
| `trellis/data/` | `KEEP` | market-data fetch and resolution | remain infrastructure layer | architecturally separate from pricing already |
| `trellis/instruments/` | `KEEP` + `REFACTOR` | reference instruments and payoff implementations | remain reference contract/payoff layer with clearer public grouping | concept is right, but package docs and interfaces need standardization |
| `trellis/instruments/_agent/` | `DEPRECATE` | generated payoff modules | remain generated cache only, not canonical library surface | should stay clearly non-canonical |
| `trellis/models/analytical/` | `KEEP` | closed-form pricing formulas | remain analytical pricing family | correct conceptual family |
| `trellis/models/transforms/` | `KEEP` + `REFACTOR` | FFT/COS pricing | remain transform family with clearer compatibility docs and diagnostics | family exists but docs and method naming need normalization |
| `trellis/models/pde/operator.py` | `KEEP` | PDE operator abstractions | remain PDE operator layer | good separation of problem definition from solver |
| `trellis/models/pde/theta_method.py` | `KEEP` | canonical 1D theta-method solver | remain canonical PDE solver entry point | this is already the right canonical surface |
| `trellis/models/pde/crank_nicolson.py` | `DEPRECATE` | legacy PDE entry point | backward-compat wrapper only | theta-method already subsumes it |
| `trellis/models/pde/implicit_fd.py` | `DEPRECATE` | legacy PDE entry point | backward-compat wrapper only | theta-method already subsumes it |
| `trellis/models/pde/psor.py` | `WRAP` | standalone projected-SOR helper | remain low-level PDE utility under canonical theta-method docs | should not compete with the main solver surface |
| `trellis/models/trees/` | `KEEP` + `REFACTOR` | lattice and short-rate tree implementations | remain lattice/tree family with clearer model-vs-engine separation | core functionality exists; public organization needs cleanup |
| `trellis/models/trees/models.py` | `KEEP` | short-rate tree model specifications | remain model-spec layer for lattice builders | conceptually useful and should stay |
| `trellis/models/monte_carlo/engine.py` | `REFACTOR` | general simulation/pricing engine | remain MC path-simulation engine with narrower responsibilities | currently mixes path generation and pricing without clean estimator separation |
| `trellis/models/monte_carlo/discretization.py` | `KEEP` | simulation schemes | remain simulation-scheme layer | correct placement |
| `trellis/models/monte_carlo/schemes.py` | `KEEP` + `REFACTOR` | object-oriented simulation schemes | remain scheme interface layer | useful but should be documented relative to engine/discretization |
| `trellis/models/monte_carlo/lsm.py` | `WRAP` + `REFACTOR` | LSM optimal-stopping implementation | become explicit exercise/control-method surface | functionality exists but is conceptually buried under generic MC |
| `trellis/models/monte_carlo/variance_reduction.py` | `REFACTOR` | variance reduction plus Sobol helper | remain MC/QMC helper layer with clearer separation | QMC support is only partial and should be surfaced honestly |
| `trellis/models/monte_carlo/brownian_bridge.py` | `WRAP` | Brownian-bridge construction | become explicit QMC/path-construction helper | useful but not exposed as a dedicated family today |
| `trellis/models/processes/` | `KEEP` | stochastic process definitions | remain model layer distinct from numerical engines | this is the right conceptual placement |
| `trellis/models/calibration/` | `KEEP` | calibration primitives | remain calibration layer | good separation already |
| `trellis/models/copulas/` | `KEEP` + `REFACTOR` | dependence/credit-model components | remain model components, not general pricing-engine peers | concept is valid but package docs should state its narrower role |
| `trellis/models/cashflow_engine/` | `KEEP` + `REFACTOR` | waterfall, amortization, prepayment logic | remain structured cashflow modeling layer | useful but should not be confused with a general numerical pricing family |
| `trellis/models/black.py` | `KEEP` | Black76 formulas | remain analytical utility surface | stable and well-scoped |
| `trellis/models/vol_surface.py` | `KEEP` + `REFACTOR` | volatility-surface protocol and flat implementation | remain market/model interface surface | placement is good, but diagnostics and richer implementations may grow here |
| `trellis/engine/payoff_pricer.py` | `KEEP` + `WRAP` | generic payoff evaluation gateway | become the canonical orchestration surface for payoff evaluation | already aligns better with the target architecture than `pricer.py` |
| `trellis/engine/pricer.py` | `REFACTOR` | bond-oriented pricing orchestrator | narrow to fixed-income instrument orchestration or move behind clearer facade | currently too specific to be presented as the general pricing engine |
| `trellis/engine/analytics.py` | `REFACTOR` | curve-Greek calculator | merge into a clearer risk layer | this belongs conceptually with risk/analytics rather than engine dispatch |
| `trellis/analytics/` | `REFACTOR` | composable analytics measures and OAS tools | become explicit risk/analytics layer | good concept, but should own risk semantics more clearly |
| `trellis/session.py` | `KEEP` + `REFACTOR` | market snapshot, pricing facade, analytics facade, agent entry point | remain the user-facing facade with clearer delegation boundaries | high-value surface, but responsibilities are broad |
| `trellis/book.py` | `KEEP` | portfolio container and aggregation | remain user-facing portfolio layer | already useful and well contained |
| `trellis/pipeline.py` | `KEEP` + `REFACTOR` | scenario/pipeline orchestration | remain user workflow layer | should be aligned with the normalized session/analytics surface |
| `trellis/__init__.py` | `REFACTOR` | top-level convenience export surface | become the canonical stable public API | currently broad but not yet standardized |
| `trellis/models/__init__.py` | `REFACTOR` | empty package file | become curated family-level export surface or remain intentionally minimal with docs | current emptiness makes the package surface inconsistent |
| `trellis/core/__init__.py` | `REFACTOR` | empty package file | become curated domain-level export surface or remain intentionally minimal with docs | current emptiness makes the package surface inconsistent |
| `trellis/agent/quant.py` | `MERGE` | static pricing-plan source | merge authority into canonical decomposition/requirements knowledge | currently duplicates canonical YAML concepts |
| `trellis/agent/cookbooks.py` | `MERGE` | Python cookbook registry | consolidate with canonical `cookbooks.yaml` as the source of truth | current duplication risks drift |
| `trellis/agent/knowledge/import_registry.py` | `KEEP` | authoritative import surface for generation | remain authoritative import-registry layer | this is a core anti-hallucination control |
| `trellis/agent/knowledge/canonical/features.yaml` | `KEEP` + `REFACTOR` | canonical feature taxonomy | remain canonical feature ontology | should evolve carefully to reflect taxonomy changes |
| `trellis/agent/knowledge/canonical/decompositions.yaml` | `KEEP` + `REFACTOR` | canonical instrument-to-method map | remain canonical decomposition map | should absorb duplicated static planning logic |
| `trellis/agent/knowledge/canonical/method_requirements.yaml` | `KEEP` + `REFACTOR` | per-method modeling constraints | remain canonical modeling-constraint layer | method names and scope need normalization with refactor decisions |
| `trellis/agent/knowledge/canonical/api_map.yaml` | `KEEP` + `REFACTOR` | hot-start API reference | remain compact prompt-facing API map | must track public-surface changes closely |
| `README.md` | `REFACTOR` | stale repo identity | describe Trellis accurately | currently misstates the project |
| `setup.py` | `DEPRECATE` | legacy package metadata for `rate-model` | retain only if needed for compatibility; otherwise phase out in favor of `pyproject.toml` | currently conflicts with the active package identity |
| `rate_model/` | `DEPRECATE` | legacy pre-Trellis package surface | keep only as compatibility shim until migration path is explicit | no active in-repo references were found |

## Key Audit Findings

### 1. Trellis already has a workable domain/engine split

The repo already separates domain abstractions from numerical methods better than the prompt assumes:

- domain-like surfaces live in `core`, `conventions`, `curves`, and much of `instruments`
- numerical methods live primarily in `models`
- orchestration lives in `session`, `engine`, `analytics`, and `pipeline`

The refactor should preserve that shape instead of forcing a full rewrite into a new `pricing/` root.

### 2. `models/` is overloaded, but not wrong

`trellis/models/` currently contains at least four distinct concepts:

- numerical engines
- stochastic models/processes
- calibration
- structured cashflow modeling

This should be clarified with package docs and export conventions before any major file moves.

### 3. Exercise/control functionality exists, but is conceptually buried

Exercise logic is already present in three places:

- `trellis.models.monte_carlo.lsm`
- `trellis.models.trees.backward_induction` and lattice helpers
- `trellis.models.pde.theta_method` via obstacle/projected solves

This is not a missing capability problem. It is a discoverability and API-shape problem.

### 4. QMC is partial, not absent

There is no dedicated `qmc` package, but the repo already has usable building blocks:

- Sobol normal generation
- Brownian bridge

The right next step is to wrap and document this honestly as partial QMC support, not to invent a fake comprehensive subsystem.

### 5. The agent/knowledge system is a hard dependency for any refactor

Method names and module paths are embedded in:

- `trellis/agent/quant.py`
- `trellis/agent/cookbooks.py`
- `trellis/agent/prompts.py`
- canonical knowledge YAML
- lesson metadata
- tests

Any structural refactor that ignores these surfaces will create prompt drift and import hallucinations.

### 6. There is visible documentation and packaging drift

The repo identity is currently inconsistent across:

- `README.md`
- `setup.py`
- `pyproject.toml`
- parts of the docs tree

This is low-risk to functionality but high-risk to future maintenance and agent correctness.

## Baseline Validation

### Targeted baseline tests

All of the following passed before structural edits:

- `tests/test_core` → `25 passed`
- `tests/test_models/test_generalized_methods.py` → `44 passed`
- `tests/test_agent/test_knowledge_store.py` → `37 passed`
- `tests/test_v2_api.py` → `41 passed`

### Broad non-integration baseline

Two broader runs exposed the same external dependency issue:

- `pytest tests/ -x -q -m "not integration"` failed in `tests/test_crossval/test_xv_bonds.py`
- `pytest tests/ -x -q -m "not integration" --ignore=tests/test_crossval` failed in `tests/test_tasks/test_t01_zcb_option.py`

Both failures were caused by `financepy` import paths that trigger a `numba` caching error during import, not by a Trellis assertion failure:

- `RuntimeError: cannot cache function 'date_index': no locator available`

This is an environment/external-library blocker that should be recorded separately from Trellis behavioral regressions.

### Warning baseline

The broader runs also emitted pre-existing numerical warnings in:

- `trellis/models/pde/thomas.py`
- `trellis/models/transforms/cos_method.py`

These did not immediately fail the suite, but they are useful candidates for later diagnostics/guardrails work.

## Tranche 2 Priority Order

Tranche 2 should focus on normalization of existing families, not new method-family expansion.

### Priority 1: stabilize the public surface

- standardize package exports in `trellis/__init__.py`, `trellis/models/__init__.py`, and `trellis/core/__init__.py`
- update stale docs to point to canonical surfaces
- keep backward-compat wrappers where imports already exist

### Priority 2: reduce duplicated agent authority

- make canonical YAML the source of truth for method decomposition and cookbook guidance
- shrink duplication between `trellis/agent/quant.py` and `trellis/agent/knowledge/canonical/decompositions.yaml`
- shrink duplication between `trellis/agent/cookbooks.py` and `trellis/agent/knowledge/canonical/cookbooks.yaml`

### Priority 3: clarify family boundaries without sweeping file churn

- Monte Carlo vs exercise/control
- PDE canonical solver vs legacy wrappers
- tree engines vs tree-model specifications
- copulas and cashflow-engine modules as modeling components rather than generic pricing-family peers

### Priority 4: fix stale metadata and compatibility surfaces

- update `README.md`
- decide whether `setup.py` remains as a compatibility shim or is retired
- decide whether `rate_model/` remains as a compatibility shim or moves to formal deprecation

## Explicit Non-Goals For Tranche 2

Do not do the following yet:

- add BSDE/FBSDE scaffolding
- add MLMC scaffolding
- add quantization scaffolding
- add surrogate / PINN / deep-method scaffolding
- perform a full package-root rewrite into a new `pricing/` namespace

These would increase surface area faster than the current architecture can support honestly.
