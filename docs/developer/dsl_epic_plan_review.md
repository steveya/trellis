# QUA-409 Epic: DSL Enforcement and Expressiveness — Plan Review

**Date:** 2026-03-31
**Reviewer scope:** All 6 tickets in the epic (4 done, 2 backlog)
**Method:** Verify coherence across plans, check as-built code against stated plans, identify gaps and suggest changes for unimplemented tickets

---

## 1. Epic Coherence Check

### Dependency graph (verified)

```
QUA-410 (build gate) ✅
    ↓
QUA-420 (credit concepts) ✅ ── depends on QUA-410 gate infrastructure

QUA-411 (measure protocol) ✅ ── independent of QUA-410
QUA-412 (event machine) ✅ ── independent of QUA-410/411

QUA-438 (calibration contract) ── depends on QUA-411 ✅, QUA-412 ✅
    ↓
QUA-439 (composition algebra) ── depends on QUA-438
```

**Verdict: dependency chain is sound.** No circular dependencies. The implemented tickets (410, 411, 412, 420) form a clean foundation for 438 and 439.

### Naming and ID consistency

**Issue found:** QUA-439's description references `QUA-431` as the calibration contract dependency, but the actual ticket is `QUA-438`. This is a stale reference from the planning phase.

**Recommendation:** Update QUA-439's description to reference `QUA-438` instead of `QUA-431`.

---

## 2. Completed Tickets — As-Built vs. Plan

### QUA-410: Build Gate ✅

| Plan item | As-built | Match |
|---|---|---|
| `BuildGateDecision` frozen dataclass | In `knowledge/schema.py` | ✅ |
| `BuildGateThresholds` frozen dataclass | In `knowledge/schema.py` | ✅ |
| Pre-flight gate in `autonomous.py` | Blocks on confidence < 0.3 | ✅ |
| Pre-generation gate in `executor.py` | Checks blockers + instruction conflicts | ✅ |
| `gap_report` threaded to executor | New param on `build_payoff()` | ✅ |
| Spec-schema hint wiring | Planner consults `compiled_request` blueprint | ✅ |
| Gate decision in traces | `build_meta["build_gate_decision"]` | ✅ |

**No deviations from plan.**

### QUA-411: Measure Protocol ✅

| Plan item | As-built | Match |
|---|---|---|
| `DslMeasure(str, Enum)` | In `core/types.py`, 13 members | ✅ |
| `normalize_dsl_measure()` | 8 aliases, case-insensitive | ✅ |
| `sensitivity_support.py` returns DslMeasure | Updated `normalize_requested_measures()` | ✅ |
| `requested_measures` on blueprint | On `SemanticImplementationBlueprint` | ✅ |
| `measure_support_warnings` on blueprint | Added alongside | ✅ Bonus |
| Bridge `dsl_measure_to_runtime()` | In `analytics/measures.py` | ✅ |
| ~~Prompt changes~~ | Correctly removed from scope | ✅ |
| ~~required_outputs on GenerationPlan~~ | Correctly removed from scope | ✅ |

**Plan was revised mid-session to remove unnecessary prompt/validator changes. Revision was correct.**

**One minor issue found:** The `measure_support_warnings` comparison uses `.value` to extract the string from a `DslMeasure`, but `DslMeasure` already IS a `str` via inheritance. The comparison `m_val = m.value if hasattr(m, "value") else str(m)` is defensive but could be simplified to `str(m.value)` or just `m` since `m == "dv01"` already works. Not a bug, just unnecessary defensiveness.

### QUA-412: Event State Machine ✅

| Plan item | As-built | Match |
|---|---|---|
| 5 core frozen dataclasses | `EventState`, `EventGuard`, `EventAction`, `EventTransition`, `EventMachine` | ✅ |
| `validate_event_machine()` with BFS | 8 validation checks, returns error tuple | ✅ |
| `compile_event_machine_to_timeline()` | Maps to `PathEventTimeline` via `event_step_indices()` | ✅ |
| `emit_event_machine_skeleton()` | State enum + guard/action stubs + dispatch loop | ✅ |
| `autocallable_event_machine()` factory | 3 states, 2 transitions | ✅ |
| `tarf_event_machine()` factory | 3 states, 2 transitions | ✅ |
| `event_transitions_to_machine()` migration | Recognizes basket pattern + linear fallback | ✅ |
| Wire into `SemanticProductSemantics` | `event_machine: object | None` field | ✅ |
| Wire into `ProductIR` | `event_machine: object | None` field | ✅ |
| Wire into compiler | Propagated + `event_machine_skeleton` on blueprint | ✅ |
| Wire into validation | `_validate_event_machine()` check | ✅ |

**No deviations from plan.**

### QUA-420: Credit Concept Ambiguity ✅

| Plan item | As-built | Match |
|---|---|---|
| `credit_default_swap` concept | In `semantic_concepts.py` with 5 aliases, 8 cue phrases | ✅ |
| `nth_to_default` concept | In `semantic_concepts.py` with 6 aliases, 8 cue phrases | ✅ |
| Ambiguous → requires_clarification wiring | 5 lines in `classify_semantic_gap()` | ✅ |
| Batch path fix | `task_runtime.py` uses `compile_build_request` for all instruments | ✅ Bonus |

**Bonus: the batch path fix was not in the original QUA-420 scope but was identified during review and implemented. This closes the gap that would have required QUA-414.**

---

## 3. Unimplemented Tickets — Gap Analysis and Suggestions

### QUA-438: Calibration Contract

**What's good:**
- Clean type hierarchy: `CalibrationTarget` → `FittingInstrument` → `CalibrationMethod` → `CalibrationContract`
- Factory functions mapping to existing proven primitives (good reuse)
- Validation checking import registry (consistent with route registry pattern)

**Suggestions:**

**S1: Add `CalibrationResult` type, not just `CalibrationContract`.**
The contract declares *what* to calibrate. But the compiler also needs to know what the calibration *produced* — so the generated code can consume it. The existing `RatesCalibrationResult` has `calibrated_vol`, `repriced_price`, `residual`, `provenance`. A `CalibrationResult` type should be declared alongside the contract so the output binding is typed, not just a string path.

```python
@dataclass(frozen=True)
class CalibrationResult:
    target: CalibrationTarget
    calibrated_parameters: dict[str, float]  # e.g. {"mean_reversion": 0.03, "sigma": 0.01}
    residual: float
    provenance: dict[str, object]
    accepted: bool  # passes acceptance_criteria?
```

**S2: `output_binding` should be structured, not a string.**
`output_binding: str = "market_state.vol_surface"` is fragile. Consider:

```python
@dataclass(frozen=True)
class OutputBinding:
    target_path: str           # "market_state.vol_surface"
    parameter_names: tuple[str, ...]  # what the calibration produces
    consumption_pattern: str   # "replace_field", "inject_parameter", "build_lattice"
```

This makes the compiler's job deterministic: it knows exactly how to wire calibrated parameters into the pricing step.

**S3: Wire calibration into the build gate (QUA-410).**
The plan says "wire into compiler" but doesn't mention the build gate. If a task needs calibration (detected via concept decomposition) but no `CalibrationContract` exists, the build gate should emit `decision="narrow_route"` with a diagnostic. This is a one-line addition to `evaluate_pre_generation_gate()`.

**S4: Add `calibration` field to `SemanticContract` at the right level.**
The plan says "add to SemanticContract (the top-level contract, not product semantics)." This is correct — calibration is a build-phase concern, not a product-shape concern. But verify that `make_callable_bond_contract()` can actually set this field. Currently the factory returns a `SemanticContract` constructed in `semantic_contracts.py`. The `SemanticContract` dataclass needs to be updated to accept the new field.

**S5: Consider multi-step calibration.**
HW calibration is single-step (fit mean reversion + sigma to swaptions). But some products need chained calibration: first calibrate the discount curve (bootstrap), then calibrate HW to swaptions using that curve, then price. The `CalibrationContract` as designed is single-step. Add `depends_on: str = ""` (another `CalibrationContract` ID) to support chaining. This avoids needing the full composition algebra (QUA-439) for simple calibration chains.

### QUA-439: Composition Algebra

**What's good:**
- DAG-based design with typed edges (consistent with `EventMachine`'s graph validation)
- Market data unioning and method intersection are well-defined
- Factory functions for callable bond and callable range accrual ground the design

**Suggestions:**

**S6: Fix the stale dependency reference.**
QUA-439 references `QUA-431` as the calibration dependency. Should be `QUA-438`.
*(Fixed in QUA-409 epic closeout.)*

**S7: `SubContractRef.contract: object` should be typed.**
Using `object` avoids circular imports, but this is the composition algebra — the whole point is typed composition. Consider:

```python
from __future__ import annotations
from typing import Union

SubContractPayload = Union["SemanticContract", "CalibrationContract", "CompositeSemanticContract"]

@dataclass(frozen=True)
class SubContractRef:
    contract_id: str
    contract: SubContractPayload
    proven: bool = False
    primitive_ref: str = ""
```

With `from __future__ import annotations`, the forward reference resolves at runtime. This gives type checkers real types to validate.

**S8: Compiler integration needs more specificity.**
The plan says "extend `compile_semantic_contract()` to handle `CompositeSemanticContract`." But `compile_semantic_contract()` currently expects a `SemanticContract` — it calls `validate_semantic_contract(spec)`, accesses `contract.product`, `contract.market_data`, etc. A `CompositeSemanticContract` has a DAG of sub-contracts, not a single product.

Two options:
- **Option A:** Add a separate `compile_composite_contract()` function that does the topological sort and calls `compile_semantic_contract()` for each sub-contract. Returns a `CompositeBlueprint` (new type) with ordered steps.
- **Option B:** Make `compile_semantic_contract()` detect the input type and dispatch.

**Recommendation:** Option A. Keep `compile_semantic_contract()` focused on single contracts. A composite compiler is a higher-level orchestrator that calls the single-contract compiler for each sub-contract. This is cleaner and avoids the single function becoming a dispatcher.

**S9: Edge types need runtime semantics.**
`edge_type: str = "sequential"` tells you the order, but not what happens at the boundary. What data flows from one sub-contract to the next? Consider:

```python
@dataclass(frozen=True)
class ContractEdge:
    from_contract: str
    to_contract: str
    edge_type: str
    data_flow: tuple[str, ...]  # what flows: ("calibrated_lattice", "discount_curve")
    condition: str = ""
```

Without `data_flow`, the compiler doesn't know what the calibration step produces that the pricing step consumes. This is the glue that makes composition useful.

**S10: Start with the simplest composite — don't over-design.**
The callable bond is the canonical first case: `HW_calibration → bond_cashflows → backward_induction`. This is a linear chain (no branching, no conditionals). The plan includes conditional edges, parallel edges, and full DAG support. Consider implementing only `sequential` and `calibrate_then_price` edges first. Add `conditional` and `parallel` when there's a real product that needs them.

---

## 4. Cross-Cutting Concerns

### C1: Prompt rendering gap

QUA-438 and QUA-439 both list prompt rendering as "follow-on work." But the entire value of these contracts is that the agent sees structured context instead of ad-hoc discovery. The prompt wiring should be scoped into the implementation tickets, not deferred. At minimum:
- Calibration contract → prompt section: "Calibration is handled by `{proven_primitive}`. Consume `{output_binding}` — do not generate calibration code."
- Composition → prompt section: "Sub-contract `{contract_id}` is proven. Call `{primitive_ref}` for this leg. Generate only: `{unproven sub-contracts}`."

### C2: Gap check integration

Both tickets defer gap_check integration to follow-on work. But the build gate (QUA-410) already has the infrastructure. Add a simple check: if `ProductIR.state_dependence == "path_dependent"` and no `EventMachine` is declared, the gap_check confidence should be penalized. Similarly, if the decomposition suggests calibration is needed (rate-tree method + no CalibrationContract), penalize confidence.

### C3: Test strategy

The completed tickets have 82 unit tests but no integration test that exercises the full path: request → concept → contract → gate → compile → codegen → validate. This is listed in the system design review (`dsl_system_design_review.md`, Section 5.2) as "Cross-layer integration: None." QUA-438 and QUA-439 should each include at least one end-to-end test with a mock LLM call.

---

## 5. Summary of Recommended Changes

| # | Ticket | Change | Priority |
|---|---|---|---|
| S1 | QUA-438 | Add `CalibrationResult` typed output | High — compiler needs it |
| S2 | QUA-438 | Structured `OutputBinding` instead of string | Medium — improves compiler determinism |
| S3 | QUA-438 | Wire calibration need into build gate | Low — one-line addition |
| S4 | QUA-438 | Verify `SemanticContract` accepts `calibration` field | High — blocking |
| S5 | QUA-438 | Add `depends_on` for chained calibration | Medium — needed for multi-curve |
| S6 | QUA-439 | Fix `QUA-431` → `QUA-438` reference | Trivial |
| S7 | QUA-439 | Type `SubContractRef.contract` properly | Medium |
| S8 | QUA-439 | Separate `compile_composite_contract()` function | High — architecture clarity |
| S9 | QUA-439 | Add `data_flow` to `ContractEdge` | High — makes composition useful |
| S10 | QUA-439 | Start with linear chains only | Medium — reduces risk |
| C1 | Both | Scope prompt rendering into tickets, not follow-on | High |
| C2 | Both | Wire into gap_check confidence scoring | Medium |
| C3 | Both | Add at least one end-to-end integration test | Medium |
