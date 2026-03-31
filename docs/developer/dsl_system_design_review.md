# Trellis Semantic DSL — System Design Review

**Date:** 2026-03-30
**Scope:** Integrity verification, extensibility analysis, comparative DSL gap assessment
**Corpus:** QUA-373, QUA-374, QUA-375, QUA-379, QUA-387, QUA-388 + full codebase read

---

## 1. Architecture Summary (as-built)

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 5: Learning Loop                    │
│  gap_check → reflect → promote → distill                    │
│  (autonomous.py, promotion.py, reflect.py)                  │
├─────────────────────────────────────────────────────────────┤
│                Layer 4: Instruction Policy                   │
│  InstructionRecord → resolve → ResolvedInstructionSet       │
│  (instructions.py, precedence: hard > hint > note > depr)   │
├─────────────────────────────────────────────────────────────┤
│              Layer 3: Compiler / Blueprinting                │
│  SemanticContract → validate → compile →                    │
│     SemanticImplementationBlueprint                          │
│  (semantic_contract_compiler.py, route_registry.py)         │
├─────────────────────────────────────────────────────────────┤
│              Layer 2: Deterministic Validation                │
│  ProductIR → GapReport (confidence 0.0–1.0)                 │
│  (gap_check.py, semantic_contract_validation.py)            │
├─────────────────────────────────────────────────────────────┤
│              Layer 1: Request Normalization                   │
│  free-text / term-sheet → ProductIR                          │
│  (semantic_concepts.py → resolve → decompose.py)            │
└─────────────────────────────────────────────────────────────┘
```

**Key invariants the DSL claims:**
- Family-name-free synthesis (no product-specific code branches)
- Feature-based retrieval (instruments are molecules, features are atoms)
- Contracts over conventions (every boundary is a typed frozen dataclass)
- Bounded extension (gap classification before synthesis; gated promotion)
- Deterministic validation (all decision points are traceable checkpoints)

---

## 2. Integrity Verification

### 2.1 What holds up well

| Property | Evidence | Verdict |
|---|---|---|
| **Type safety** | All core types are frozen dataclasses with tuple fields; MappingProxyType for dicts | ✅ Strong |
| **Determinism** | Concept resolution uses numeric scoring with explicit tie-breaking; instruction resolution is precedence-ordered | ✅ Strong |
| **Traceability** | Resolved instructions persisted in analytical traces; extension proposals written to `semantic_extension/` dir | ✅ Strong |
| **Immutability** | Lessons are append-only; superseded records are archived, never deleted; promotion gates require validation | ✅ Strong |
| **Import hygiene** | Route registry validates all primitives against live import registry at load time via `is_valid_import()` | ✅ Strong |

### 2.2 Integrity gaps

#### Gap I-1: DSL is advisory, not enforcing — **RESOLVED by QUA-410**

~~The most consequential finding. The DSL is a **guidance layer** consumed by LLM prompts, not a **constraint layer** that gates the build pipeline.~~

**Resolution (QUA-410, March 2026):** A hard build-gate checkpoint is now
enforced between Layer 2 (validation) and Layer 3 (compilation).
`evaluate_pre_flight_gate()` runs after `gap_check()` in `autonomous.py`;
`evaluate_pre_generation_gate()` runs inside `executor.py` before the builder
prompt is assembled. Both return a `BuildGateDecision` that can
`proceed / narrow_route / clarify / block`. Gap confidence < 0.4 blocks the
build and emits a structured clarification request instead of wasting an LLM
round-trip.

| Constraint | Defined in | Consumed by | Enforcement |
|---|---|---|---|
| Route selection | `route_registry.py` | `build_generation_plan()` | ✅ Assembly-first |
| Instruction precedence | `instructions.py` | Prompt rendering | ⚠️ Soft (LLM sees it) |
| Market data requirements | `MarketDataAccessSpec` | Post-generation review | ❌ Not yet gating |
| Spec schema selection | `family_contracts.py` | `planner.py` | ❌ Bypassed (see Gap I-2) |
| Gap report severity | `gap_check.py` | `build_gate.py` | ✅ Now blocks (QUA-410) |
| Instruction conflicts | `ResolvedInstructionSet` | Rendered for LLM | ❌ LLM decides |
| Contract validation | `semantic_contract_validation.py` | `lite_review.py` | ⚠️ Post-hoc only |

#### Gap I-2: Spec selection ignores DSL contracts — **PARTIALLY RESOLVED by QUA-410**

`planner.py._select_specialized_spec()` uses regex and keyword matching against description text. `STATIC_SPECS` is a hardcoded instrument → schema map. Neither consults `SemanticContract.blueprint.spec_schema_hints` or `FamilyContract.blueprint.spec_schema_hints`.

**Partial resolution (QUA-410):** The build gate now exposes the compiled
blueprint (including `spec_schema_hint`) to the executor before the planner
runs. Full wiring of `spec_schema_hints` into `planner.py` remains outstanding
(see Gap I-2 in open tickets above).

**Remaining impact:** The planner can still select a spec schema that contradicts the contract. The `spec_schema_hint` is computed in `SemanticImplementationBlueprint` and logged but not yet consumed by `planner.py`.

#### Gap I-3: Knowledge monkey-patching short-circuits DSL retrieval

`autonomous.py` pre-computes knowledge text and monkey-patches `executor._retrieve_knowledge`. This means the dynamic DSL-based retrieval (concept → feature → lesson lookup) is baked into a static string before the builder ever sees it. If the DSL contracts change mid-session (e.g., a lesson is promoted during a retry), the patched text is stale.

#### Gap I-4: No runtime contract enforcement

Once generated code passes `lite_review`, it executes with no DSL guards. If the code accesses a MarketState field that the contract declared as optional (not required), and that field is absent at runtime, the failure is a raw Python exception — not a contract violation with structured diagnostics.

#### Gap I-5: Concept resolution scoring is fragile

`resolve_semantic_concept()` uses additive numeric scoring (alias=100, phrase=30, cue=10) with a hardcoded threshold. There is no learning signal: the weights are static. For novel requests with partial overlap across multiple concepts, the confidence formula `min(1.0, top_score / 120.0)` can produce misleadingly high scores from a single cue-phrase hit.

---

## 3. Extensibility Analysis

### 3.1 Extension axes and their readiness

| Extension axis | Mechanism | Readiness | Bottleneck |
|---|---|---|---|
| **New product family** | Add concept to `_SEMANTIC_CONCEPT_REGISTRY` + contract factory + routes.yaml entry | 🟡 Medium | Manual; no scaffolding tool |
| **New market input** | Add `SemanticMarketInputSpec` to contract + capability to MarketState | 🟡 Medium | Capability registry is implicit |
| **New pricing method** | Add route to routes.yaml + primitives module | 🟢 Good | Route registry auto-discovers |
| **New exercise style** | Add conditional primitives to exercise routes + update concept `allowed_attributes` | 🟡 Medium | Must touch multiple files |
| **New validation rule** | Add check function to `semantic_contract_validation.py` | 🟢 Good | Linear addition |
| **New instruction source** | Emit `InstructionRecord` from any subsystem | 🟢 Good | Schema is open |
| **New agent role** | Add entry to `_SEMANTIC_ROLE_MATRIX` | 🟢 Good | Deterministic matrix |
| **Multi-curve / correlation** | Requires new `MarketInputSpec` entries + connector plumbing | 🔴 Hard | Connector layer not DSL-aware |
| **Path-dependent exotics** | Requires state-machine DSL for event transitions | 🔴 Hard | `event_transitions` field exists but is unused |
| **Basket of baskets** | Requires recursive `constituents` + nested schedule | 🔴 Hard | Flat constituent model |

### 3.2 Extension bottlenecks

**Bottleneck E-1: No concept scaffolding tool.** Adding a new product concept requires editing `_SEMANTIC_CONCEPT_REGISTRY` (hardcoded tuple), writing a `make_*_contract()` factory, adding routes to YAML, and updating decompositions. There is no `trellis add-concept` CLI or codegen for the boilerplate.

**Bottleneck E-2: Event-state machine is a placeholder.** `ProductSemantics.event_transitions` and `SemanticProductSemantics.event_transitions` are declared as `tuple[str, ...]` — flat string labels. For real exotic pricing (autocallables, TARFs, accumulators), you need a typed state machine with transitions, guards, and terminal conditions. The DSL has the field but not the formalism.

**Bottleneck E-3: No composition algebra.** The DSL can express single-product contracts but has no way to express structured products as a composition of sub-contracts (e.g., a callable range accrual = range_accrual ⊕ callable ⊕ fixed_coupon). The `constituents` field is a flat tuple of strings, not a recursive contract tree.

**Bottleneck E-4: Connector layer is outside the DSL boundary.** `MarketDataAccessSpec` declares `market_state.discount`, `market_state.vol_surface`, etc., but the actual connector resolution (how to populate MarketState from raw market data) is outside the DSL. This means the DSL can validate *what* data is needed but cannot validate *how* it arrives or *whether the provenance is acceptable*.

---

## 4. Comparative DSL Gap Analysis

### 4.1 Comparison targets

I compare against four DSL families relevant to Trellis's problem space:

| DSL / System | Domain | Key idea |
|---|---|---|
| **QuantLib** (C++ template DSL) | Derivative pricing | Instrument ↔ PricingEngine ↔ TermStructure triad; compile-time type safety |
| **Strata / OpenGamma** (Java) | Trade lifecycle | Typed product model with measure-driven pricing via `CalculationRunner` |
| **Halide** (image processing DSL) | Schedule-separated computation | Algorithm definition separated from schedule optimization |
| **LangGraph / DSPy** (LLM agent DSLs) | Agent orchestration | Graph-based agent state machines with typed state, conditional edges, and tool binding |

### 4.2 Gap matrix

| Capability | Trellis DSL | QuantLib | Strata | Halide | LangGraph/DSPy | Gap severity |
|---|---|---|---|---|---|---|
| **Typed product IR** | ✅ ProductIR (frozen DC) | ✅ Instrument hierarchy | ✅ Product model | N/A | N/A | None |
| **Method dispatch** | ✅ Route registry | ✅ PricingEngine registry | ✅ CalculationRunner | ✅ Schedule | N/A | None |
| **Market data binding** | ⚠️ Declared but not enforced | ✅ TermStructure handles | ✅ MarketData container | N/A | N/A | **Medium** |
| **Composition algebra** | ⚠️ Scaffold in `composition_algebra.py` (QUA-413, not wired) | ✅ CompositeInstrument | ✅ ResolvedTrade legs | ✅ Func composition | ✅ Graph edges | **Medium** |
| **State machine** | ✅ `EventMachine` + `validate_event_machine()` + skeleton emitter (QUA-412) | ⚠️ Ad-hoc per engine | ✅ Lifecycle states | N/A | ✅ Typed StateGraph | **Resolved** |
| **Measure protocol** | ✅ `DslMeasure` enum + `normalize_dsl_measure()` threaded through compiler (QUA-411) | ✅ NPV/Greeks/etc. | ✅ Measure enum | N/A | N/A | **Resolved** |
| **Schedule algebra** | ⚠️ generate_schedule only | ✅ Schedule class | ✅ PeriodicSchedule | ✅ Split/fuse/tile | N/A | **Medium** |
| **Calibration contract** | ⚠️ Concept exists, unused | ✅ CalibrationHelper | ✅ Calibrator | N/A | N/A | **Medium** |
| **Error typing** | ⚠️ String codes | ✅ Typed exceptions | ✅ FailureItems | N/A | ✅ NodeInterrupt | **Low** |
| **Retry / fallback policy** | ❌ LLM retry loop | N/A | N/A | N/A | ✅ Conditional edges | **High** |
| **Observability contract** | ⚠️ Trace files | ✅ Observer pattern | ✅ Explain | ✅ Trace annotations | ✅ LangSmith | **Medium** |
| **Separation of algorithm from schedule** | ❌ Merged in codegen | N/A | N/A | ✅ Core design | N/A | **Medium** |

### 4.3 Highest-impact gaps (in context of Trellis's embedded agent)

#### Gap C-1: No composition algebra (vs. QuantLib, Strata, LangGraph)

**What's missing:** Trellis cannot express "callable range accrual" as `compose(range_accrual, callable, fixed_coupon)` where each sub-contract carries its own market data requirements, method constraints, and validation rules that are unioned at the composition boundary.

**Why it matters for the agent:** When the LLM sees a structured product, it currently has to invent the entire evaluate() function from scratch. A composition algebra would let the agent assemble proven sub-contracts and only generate the glue code.

**Minimum viable version:** Add a `CompositeSemanticContract` that holds a DAG of `SemanticContract` nodes with typed edges (sequential, conditional, parallel). Each node inherits its parent's market data requirements. The compiler flattens to a single `GenerationPlan` but emits sub-contract boundaries as comments/hooks.

#### Gap C-2: No measure protocol — **RESOLVED by QUA-411**

~~**What's missing:** The DSL has no first-class notion of "what are we computing?"~~

**Resolution (QUA-411, March 2026):** `DslMeasure(str, Enum)` implemented in
`trellis/core/types.py` with 13 canonical measures. `normalize_dsl_measure()`
with alias resolution is threaded through `compile_semantic_contract()`.
`SemanticImplementationBlueprint` now carries `requested_measures: tuple[DslMeasure, ...]`
and `measure_support_warnings: tuple[str, ...]` — when a requested measure is
not natively supported by the selected method, the blueprint warns that a
bump-and-reprice fallback will be attempted. The sensitivity support layer
(`sensitivity_support.py`) ranks candidate methods by measure coverage when
`requested_measures` is provided.

#### Gap C-3: No typed state machine for event-driven exotics — **RESOLVED by QUA-412**

~~**What's missing:** The DSL has `event_transitions: tuple[str, ...]` which is a flat list of labels.~~

**Resolution (QUA-412, March 2026):** `EventMachine`, `EventState`,
`EventTransition`, `EventGuard`, and `EventAction` are fully typed frozen
dataclasses in `trellis/agent/event_machine.py`. `validate_event_machine()`
runs BFS reachability to detect orphaned states and unreachable terminal
conditions. `emit_event_machine_skeleton()` generates a Python state-enum +
transition stub that is injected into the builder prompt for exotic pricing
tasks — the agent fills in guard expressions and payoff actions, not the entire
state-machine scaffolding. Factory functions `autocallable_event_machine()` and
`tarf_event_machine()` serve as canonical reference implementations and test
fixtures.

#### Gap C-4: No retry/fallback policy in the DSL (vs. LangGraph)

**What's missing:** When a build fails, the retry strategy is implicit in `executor.py`'s loop (retry with extra context, widen method, etc.). The DSL has no way to express "if Monte Carlo fails, try PDE; if PDE fails, try analytical with approximation." The route registry has ranked candidates, but the fallback chain is not a DSL artifact.

**Why it matters for the agent:** The retry loop currently re-enters the full pipeline with extra_context appended to the prompt. It doesn't narrow the route search or change the contract. A DSL-level fallback policy would let the compiler pre-compute an ordered fallback chain that skips failed routes deterministically.

**Minimum viable version:** Add `fallback_chain: tuple[str, ...]` to `SemanticMethodContract` — an ordered list of method IDs to try. The executor consults this chain instead of re-running the full quant → planner → builder cycle on failure.

#### Gap C-5: No calibration contract (vs. QuantLib, Strata)

**What's missing:** `calibration_target` exists as a concept in the registry but has no contract structure. There is no DSL expression for "calibrate Hull-White to these swaption vols before pricing this callable bond." The calibration step is embedded in the generated code, not a separate contract.

**Why it matters for the agent:** Calibration is the #1 source of numerical failures. When the agent generates calibration code inline, it often produces unstable optimizers or mismatched vol conventions. A calibration contract would let the DSL declare the calibration target, the instruments to fit, the optimization method, and the acceptance criteria — then route to a proven calibration primitive.

---

## 5. Test Coverage Assessment

### 5.1 What's well-tested

- Concept resolution (15+ tests including deprecated wrappers, confidence bands, tie-breaking)
- Contract validation (ranked observation basket, quanto, callable bond, swaption)
- Route registry loading and primitive validation (17 routes, 8 engine families)
- Instruction lifecycle (precedence, supersession, conflict detection)
- Lesson supersedes (word overlap scoring, category filtering, backfill)
- Semantic signal extraction (AST-based MC method detection, exercise primitives)

### 5.2 What's undertested

| Area | Current coverage | Missing |
|---|---|---|
| **Negative paths** | Sparse | Invalid inputs, malformed YAML, missing required fields |
| **Cross-layer integration** | None | Contract → compiler → route → codegen → validation end-to-end |
| **Concurrent access** | None | Parallel trace recording, lesson promotion races |
| **Circular dependencies** | None | Supersedes cycles, concept cycles, route dependency loops |
| **Scale** | None | Large instruction sets (100+), many matching routes (10+) |
| **Fallback behavior** | None | What happens when no route matches, no concept resolves |
| **Contract ↔ planner** | None | Spec schema hints actually consumed by planner |
| **Extension trace replay** | 1 test | Replay of promoted extension becoming canonical |

---

## 6. Recommendations (prioritized)

### Tier 1 — Integrity fixes (should do before next feature work)

1. **Hard-gate between Layer 2 and Layer 3.** If gap confidence < threshold or unresolved conflicts exist, block the build and emit a structured clarification or route narrowing request. This is the single highest-leverage change: it prevents wasted LLM calls on doomed builds.

2. **Wire `spec_schema_hints` into `planner.py`.** The data exists in contracts. The planner ignores it. One function call eliminates the spec-selection bypass.

3. **Replace knowledge monkey-patching with dynamic retrieval.** Make `autonomous.py` pass a retrieval callback that queries the DSL at prompt-render time, not at build-start time. This ensures promoted lessons during retries are visible.

### Tier 2 — Extensibility foundations (next epic)

4. **Measure protocol.** Add `Measure` enum, thread through compiler, validate in semantic validator. Small surface area, high payoff for Greek/risk tasks.

5. **Event state machine types.** Replace `event_transitions: tuple[str, ...]` with `EventMachine` containing typed states, transitions, guards, and actions. Gate exotic pricing tasks behind state-machine completeness.

6. **Fallback chain in MethodContract.** Let the DSL pre-declare the ordered fallback. The executor consults the chain instead of re-running the full pipeline.

### Tier 3 — Composition and calibration (future epics)

7. **Composition algebra.** `CompositeSemanticContract` with a DAG of sub-contracts. Start with the simplest case: callable bond = bond + call schedule.

8. **Calibration contract.** Typed declaration of calibration target, fitting instruments, optimizer, and acceptance criteria. Route to proven calibration primitives.

9. **Concept scaffolding CLI.** `trellis add-concept <name>` that generates the registry entry, contract factory stub, route YAML entry, and test skeleton.

### Tier 4 — Learning and governance (ongoing)

10. **Concept resolution learning.** Replace static scoring weights with a fitted model (like `route_learning.py` already does for routes). Use successful builds as positive signal.

11. **Runtime contract enforcement.** Wrap MarketState access with a contract-aware proxy that raises typed `ContractViolation` instead of raw `AttributeError` when optional fields are accessed without fallback.

12. **Cross-layer integration tests.** End-to-end test: natural-language request → concept resolution → contract → compiler → route → codegen prompt → semantic validation. At least one per product family.

---

## 7. Trade-off Analysis

| Decision | Current choice | Alternative | Trade-off |
|---|---|---|---|
| **Advisory vs. enforcing DSL** | Advisory (LLM sees guidance) | Enforcing (build blocked on violation) | Flexibility vs. wasted LLM calls. Current choice prioritizes shipping velocity over correctness guarantees. |
| **Hardcoded concept registry** | Tuple in Python source | YAML file like routes | Code is type-checked at import; YAML is editable without touching Python. Current choice is correct for < 30 concepts. |
| **Flat constituents** | `tuple[str, ...]` | Recursive contract DAG | Simplicity vs. structured-product expressiveness. Current choice is correct until autocallables/TARFs are in scope. |
| **Post-hoc validation** | Validate after codegen | Validate before codegen | Late validation catches more (real code) but costs an LLM call. Pre-validation is cheaper but operates on abstractions. **Both are needed.** |
| **Linear route ranker** | Ridge regression on 7 features | Neural/tree model | Interpretability vs. accuracy. Linear is correct for 17 routes and 8 training cases. Revisit at 50+ routes. |
| **Append-only lessons** | Never mutate, only supersede | Editable with changelog | Auditability vs. convenience. Current choice is correct for an autonomous learning system. |

---

## 8. What I'd revisit as the system grows

- **At 50+ concepts:** Move registry to YAML with a loader that emits frozen dataclasses, like routes.yaml already does.
- **At 100+ routes:** The linear ranker will plateau. Add a tree-based ranker with cross-validated feature selection.
- **At 5+ structured products:** The flat-constituent model breaks. Prioritize composition algebra.
- **At production deployment:** Runtime contract enforcement becomes mandatory. The proxy-based MarketState wrapper should be non-negotiable.
- **At multi-team usage:** The concept scaffolding CLI becomes essential to prevent registry conflicts.
