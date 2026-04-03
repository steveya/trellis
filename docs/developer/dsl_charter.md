# Semantic DSL Charter

## Purpose

This document defines the scope, vocabulary, and layered architecture of Trellis's
internal semantic DSL — the contract language through which embedded agents
decompose, route, build, and learn from derivative pricing requests.

The DSL is **not** a user-facing syntax. It is the shared data model and policy
layer that binds the agent pipeline (quant → planner → builder → critic →
validator) into a deterministic, auditable flow. When a request is novel, the
DSL also governs how the system classifies the gap, proposes the smallest
extension, and promotes what it learned.

## Design Principles

**Family-name-free synthesis.** Product names (`himalaya_option`,
`callable_bond`) are request language, never architecture keys. The DSL
decomposes every request into reusable semantic primitives — features, payoff
rules, schedules, state transitions — and routes from those primitives alone.

**Feature-based retrieval.** Instruments are molecules; features are atoms.
`callable_bond = {callable, fixed_coupons, mean_reversion}`. Retrieval is the
union of knowledge matching any constituent feature, expanded transitively via
`implies` chains.

**Contracts over conventions.** Every boundary between agents is a typed data
contract: `SemanticContract`, `PricingPlan`, `GenerationPlan`, `BuildResult`.
Free-text handoffs are not permitted at decision points.

**Bounded extension.** When the DSL encounters a concept it cannot express, it
must classify the gap before attempting synthesis. Extensions follow a gated
lifecycle — candidate → validated → promoted — and never mutate hot-tier
knowledge without passing validation.

## Layered Architecture

The DSL is organized in five layers. Each layer has a well-defined input
contract, output contract, and ownership boundary.

### Layer 1 — Request Normalization

**Purpose:** Transform free-text or term-sheet input into a deterministic
`ProductIR` — the canonical intermediate representation of "what the user
wants."

**Canonical nouns:**

| Noun | Type | Description |
|---|---|---|
| `instrument` | str | Natural-language product label (request-only, not a routing key) |
| `payoff_family` | enum | `vanilla`, `barrier`, `asian`, `lookback`, `basket`, `credit`, `rate`, `structured` |
| `exercise_style` | enum | `european`, `american`, `bermudan`, `issuer_call`, `holder_put` |
| `path_dependence` | enum | `none`, `barrier`, `asian`, `lookback`, `cliquet`, `autocall` |
| `schedule_dependence` | enum | `none`, `observation`, `fixing`, `accrual` |
| `state_dependence` | enum | `terminal_markov`, `path_dependent`, `schedule_dependent` |
| `model_family` | enum | `equity_diffusion`, `interest_rate`, `stochastic_volatility`, `jump_diffusion`, `credit_copula`, `local_vol`, `fx` |
| `underlier_structure` | enum | `single`, `basket`, `spread`, `cross_currency` |
| `constituents` | list | Underlier descriptions with weight, role, currency |
| `payoff_rule` | str | Symbolic payoff expression (e.g., `max(ranked[0] - K, 0)`) |
| `settlement_rule` | str | Cash/physical, currency, timing |
| `state_variables` | list | Named state slots (`remaining_assets`, `locked_returns`, etc.) |
| `event_transitions` | list | State mutation rules triggered by observation events |
| `selection_operator` | str | `best_of`, `worst_of`, `ranked`, `all` |
| `aggregation_rule` | str | How period returns combine (`sum`, `mean`, `compounded`) |

**Implemented in:** `knowledge/decompose.py` → `decompose_to_ir()`,
`build_product_ir()`. Static keyword extraction first; LLM fallback only when
static inference sets `unresolved_primitives`.

**Output contract:** `ProductIR` (frozen dataclass in `knowledge/schema.py`).

### Layer 2 — Deterministic Validation

**Purpose:** Verify that the `ProductIR` is internally consistent and that the
knowledge base has sufficient coverage to attempt synthesis.

**Checks performed:**

1. **Decomposition completeness** — all IR fields populated; features resolved
   against `features.yaml` taxonomy.
2. **Cookbook coverage** — at least one applicable cookbook template exists for
   the inferred `route_families`.
3. **Lesson density** — feature-matched lessons exist with sufficient promoted
   count. Confidence score is the weighted sum across five dimensions
   (decomposition 0.20, cookbook 0.25, lessons 0.30, contracts 0.15,
   requirements 0.10).
4. **Data contract coverage** — method/data pairs have explicit unit-conversion
   contracts.
5. **Method requirement satisfaction** — the IR's features are compatible with
   the selected method's structural constraints.

**Implemented in:** `knowledge/gap_check.py` → `gap_check()`. Returns a
confidence score (0.0–1.0) plus a list of gap warnings. Confidence < 0.5
triggers additional retries and injects warnings into the builder prompt.

**Output contract:** `GapReport` (confidence score, gap warnings, dimension
breakdown).

### Layer 3 — Compiler and Route Blueprinting

**Purpose:** Translate the validated `ProductIR` + `PricingPlan` into a
`GenerationPlan` — the deterministic specification of what code to generate,
which modules to import, and what constraints to enforce.

**Pipeline:**

1. `ProductIR` → `select_pricing_method_for_product_ir()` → `PricingPlan`
   (method, modules, market data, sensitivity support).
2. `PricingPlan` → `build_generation_plan()` → `GenerationPlan` (routes,
   imports, constraints).
3. `GenerationPlan` + knowledge payload → builder prompt → generated code.
4. Generated code → `sanitize_generated_source()` + `validate_generated_imports()`
   → execution.

**Route selection priority:** analytical > rate_tree > pde_solver > fft_pricing
> monte_carlo > qmc > copula > waterfall. The route is determined by
feature-method compatibility, not product name.

**Partially implemented in:** `quant.py`, `planner.py`,
`codegen_guardrails.py`, `platform_requests.py`,
`semantic_contract_compiler.py`. Known checked-in template routing now goes
through the semantic request compiler; the deprecated family-contract compiler
is retained only for compatibility tests and direct legacy utilities.

**Output contract:** `GenerationPlan` (frozen dataclass in
`codegen_guardrails.py`).

### Layer 4 — Route Guidance and Instruction Policy

**Purpose:** Manage versioned, precedence-aware route instructions that
accumulate across builds. Stale instructions must not override current plans.

**Key structures:**

| Structure | Description |
|---|---|
| `InstructionRecord` | Single route instruction with version, source, confidence, and expiry |
| `ResolvedInstructionSet` | Precedence-resolved set for a given route family |
| `ToolContract` | Platform/agent tool API contract (call signature, pre/post conditions) |

**Policy rules:**

- Instructions are append-only with monotonic version numbers.
- Conflict resolution: higher-confidence instruction wins; ties broken by
  recency.
- Expired or deprecated instructions are archived, never deleted.
- Adapter lifecycle: `fresh` → `stale` → `deprecated` → `archived`.

**Partially implemented in:** `knowledge/schema.py` (dataclasses exist),
`knowledge/promotion.py` (adapter lifecycle tracking). Not yet enforced as
runtime policy in the executor loop.

**Output contract:** `ResolvedInstructionSet` per route family.

### Layer 5 — Novelty Detection and Learning Loop

**Purpose:** When the system encounters a request it cannot fully express in
existing DSL terms, this layer classifies the gap, proposes the smallest
extension, records the trace, and promotes validated lessons.

**Three-phase flow:**

1. **Diagnose** — `gap_check()` identifies which knowledge dimension is
   deficient. If `unresolved_primitives` is non-empty in the `ProductIR`,
   the gap is a missing concept. If confidence is low but primitives are
   resolved, the gap is lesson sparsity.
2. **Propose** — After a successful build despite gaps, `reflect()` captures
   new lessons tagged with the feature set from decomposition. If the gap was
   a missing feature or adapter, the reflection proposes an extension to the
   canonical taxonomy.
3. **Promote** — Captured lessons follow the gated lifecycle: candidate →
   validated (confidence ≥ 0.6) → promoted (confidence ≥ 0.8) → periodically
   distilled into principles. Promotion never mutates hot-tier knowledge
   without passing `LessonContractReport` validation.

**Implemented in:** `knowledge/autonomous.py` (build_with_knowledge),
`knowledge/gap_check.py`, `knowledge/reflect.py`, `knowledge/promotion.py`.

**Output contract:** `BuildResult` with `reflection` metadata, plus
side-effects to lesson store and canonical YAML.

## Canonical vs. Provisional Concepts

| Category | Canonical (stable, in features.yaml) | Provisional (inferred, not yet in taxonomy) |
|---|---|---|
| Features | 220+ features across cashflow, exercise, path-dependent, model, numerical, infrastructure categories | Features inferred by LLM decomposition that don't match existing taxonomy entries |
| Decompositions | 30+ instrument → feature-set mappings in `decompositions.yaml` | Learned decompositions saved by `reflect()` after successful novel builds |
| Principles | 8 distilled rules in `principles.yaml` (hot tier, always injected) | Candidate principles pending distillation from lesson accumulation |
| Cookbooks | 7 method templates in `cookbooks.yaml` | Cookbook candidates enriched from successful code by `reflect()` |
| Failure signatures | Pattern-matched error categories in `failure_signatures.yaml` | New patterns observed but not yet promoted to canonical |

**Promotion boundary:** Provisional knowledge lives in warm/cold tier and is
feature-tagged for retrieval. It becomes canonical only after passing
validation gates and accumulating sufficient confidence across multiple builds.

## What Belongs Where

| Artifact | Layer | Owner |
|---|---|---|
| `ProductIR` fields and enums | Contract schema (Layer 1) | `knowledge/schema.py`, `knowledge/decompose.py` |
| Feature taxonomy | Contract schema (Layer 1) | `canonical/features.yaml` |
| Gap confidence scoring | Compiler output (Layer 2) | `knowledge/gap_check.py` |
| `PricingPlan`, `GenerationPlan` | Compiler output (Layer 3) | `quant.py`, `codegen_guardrails.py` |
| `SemanticContract`, `FamilyContract` | Compiler output (Layer 3) | `semantic_contracts.py`, `family_contracts.py` |
| `InstructionRecord`, route policy | Route guidance (Layer 4) | `knowledge/schema.py`, `knowledge/promotion.py` |
| Lesson lifecycle, gap classification | Learning loop (Layer 5) | `knowledge/autonomous.py`, `knowledge/reflect.py` |
| Data contracts, method requirements | Cross-cutting (Layers 2–3) | `data_contract.py`, `canonical/` YAML |

## Agent Ownership

| Agent | DSL Responsibility |
|---|---|
| **Quant** | Consumes `ProductIR` → produces `PricingPlan`. Owns method selection logic and route priority. |
| **Planner** | Consumes `PricingPlan` → produces `BuildPlan` with `SpecSchema`. Owns field-level spec design. |
| **Builder** | Consumes `GenerationPlan` + knowledge payload → produces generated code. Owns code synthesis. |
| **Critic** | Consumes generated code + semantic signals → produces validation report. Owns correctness checks. |
| **Validator** | Consumes execution results → produces pass/fail with diagnostics. Owns numerical verification. |
| **Arbiter** | Owns gap classification decision: retry, extend, or escalate. |
| **Autonomous wrapper** | Owns the three-phase loop (gap_check → build → reflect) and lesson promotion triggers. |

## Resolved Work (QUA-409 epic)

The following tickets from the QUA-409 DSL Enforcement and Expressiveness epic
were completed in March 2026:

- **QUA-410 — Hard build gate** — `evaluate_pre_flight_gate()` and
  `evaluate_pre_generation_gate()` wired into `autonomous.py` and `executor.py`.
  Gap confidence < 0.4 now blocks the build with a structured clarification
  request. `BuildGateDecision` (proceed / narrow_route / clarify / block)
  replaces the silent warning path.

- **QUA-411 — Measure protocol** — `DslMeasure(str, Enum)` added to
  `trellis/core/types.py` with 13 first-class measures (PRICE, DV01, DURATION,
  VEGA, DELTA, GAMMA, THETA, RHO, OAS, Z_SPREAD, SCENARIO_PNL, CONVEXITY,
  KEY_RATE_DURATIONS). `normalize_dsl_measure()` with alias resolution threaded
  through `compile_semantic_contract()` and the sensitivity support layer.

- **QUA-412 — Event state machine** — `EventMachine`, `EventState`,
  `EventTransition`, `EventGuard`, `EventAction` frozen dataclasses implemented
  in `trellis/agent/event_machine.py`. BFS reachability validation in
  `validate_event_machine()`. `emit_event_machine_skeleton()` injects typed
  state-machine scaffolding into builder prompts for exotic pricing tasks.
  Factory functions `autocallable_event_machine()` and `tarf_event_machine()`
  validate the design.

- **QUA-413 — Composition algebra scaffold** — `PayoffComponent`,
  `CompositeSemanticContract`, and `CalibrationContract` implemented in
  `trellis/agent/composition_algebra.py` as a proof-of-concept. The module is
  intentionally not wired into the production pipeline; full compiler
  integration is deferred to a future epic (QUA-439).

## Open Tickets and Remaining Work

### Taxonomy and Governance (highest priority)

- **QUA-387** — Concept taxonomy and extension policy: formalize which concepts
  are canonical vs. provisional, define the promotion gate for new features.
- **QUA-388** — Lesson payload schema validation: strengthen
  `LessonContractReport` to enforce semantic consistency with feature ontology.
- **QUA-379** — Agent escalation and role ownership: define which agent owns
  gap classification, proposal, and promotion decisions.

### Compiler Integration

- **QUA-286** — Validator rules and draft fixtures: deterministic validation
  suite for Layer 2.
- **QUA-287** — Compiler and request routing: keep the semantic request
  compiler authoritative for known templates, route binding, and bridge
  retirement.

### Route and Market Data Policy

- **QUA-351** — Correlation source policy: explicit, empirical, implied,
  synthetic provenance for correlation matrices.
- **QUA-353** — Market parameters provenance-aware sourcing layer.
- **QUA-373** — Basket adapter schedule-builder primitive alignment.
- **QUA-374** — Knowledge routing instruction lifecycle and conflict resolution.

### Novel Request Extension

- **QUA-375** — Novel request extension loop (diagnose → propose → remember).
- **QUA-376** — Gap classification for novel requests (completed scaffold).
- **QUA-377** — Missing primitive and input proposal (completed scaffold).
- **QUA-378** — Extension trace and lesson promotion (completed scaffold).

## Success Criteria

The DSL is complete when:

1. Any structurally recognizable derivative can be expressed as a `ProductIR`
   without adding product-name-specific branches.
2. The compiler can deterministically translate `ProductIR` → `GenerationPlan`
   for all supported route families.
3. Novel requests trigger the extension loop, and extensions that pass
   validation automatically enrich the canonical taxonomy.
4. No free-text handoff exists between agents at any decision point — every
   boundary is a typed contract.
5. Stale route instructions are automatically deprecated and never override
   current plans.
