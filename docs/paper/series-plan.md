# Trellis Paper Series Plan

## Recommendation

Write a three-part series now, with a clean fallback to two papers later.

The core novelty is not "LLMs price derivatives." The core novelty is that
Trellis makes agent-pricing possible by compiling ambiguous requests into
typed semantic objects, narrowing them through admissibility and lowering, and
then executing only deterministic numerical machinery in the pricing hot path.

That gives you a paper program with three distinct contributions:

1. a compiler/systems contribution
2. a mathematical and computational-methods contribution
3. a knowledge-governance and learning-systems contribution

## Proposed series thesis

Trellis is best presented as a governed semantic compiler for quantitative
finance, not as a generic code-generation agent. The system works because it
combines:

- contract algebra and typed product semantics
- DSL-style lowering onto bounded numerical families
- reusable deterministic mathematical engines
- a governed knowledge layer for retrieval, reflection, promotion, and audit

The series should repeatedly state one architectural rule:

> the agent handles ambiguity, compilation, and maintenance; deterministic
> numerical engines handle valuation.

## Recommended paper split

### Part I

**Working title:** `Trellis I: A Request-to-Outcome Compiler for Agent-Assisted Derivatives Pricing`

**Main question:** How can a natural-language or structured pricing request be
turned into a defensible deterministic pricing outcome?

**Primary thesis:** Agent-pricing is viable when the system compiles requests
through typed semantic contracts, valuation context, route admissibility, and
family lowering before any pricing call is executed.

**Core sections:**

1. Problem statement: why naive prompt-to-code pricing is unsafe.
2. Unified front door:
   `ask`, `Session`, `Pipeline`, structured product specs, task runner.
3. Canonical request layer:
   `PlatformRequest`, `CompiledPlatformRequest`, `ExecutionPlan`.
4. Semantic compilation:
   `SemanticContract`, semantic validation, `ValuationContext`,
   `RequiredDataSpec`, `MarketBindingSpec`, `ProductIR`.
5. Admissibility and lowering:
   route registry, build gate, `EventProgramIR`, `ControlProgramIR`,
   family-specific IR emission.
6. Validation contract and traceability:
   deterministic checks, compiled validation, canonical traces.
7. Case studies:
   one direct deterministic path and one guarded build path.

**Innovation angle:** This is a pricing compiler paper with an LLM at the
edges, not a free-form agent paper.

**Primary repo sources:**

- `ARCHITECTURE.md`
- `docs/quant/pricing_stack.rst`
- `docs/developer/overview.rst`
- `docs/agent/workflow_diagrams.md`
- `docs/developer/implementation_journey_prompt_to_price.md`
- `docs/design_validation_contract_loop.md`
- `trellis/agent/platform_requests.py`
- `trellis/agent/semantic_contracts.py`
- `trellis/agent/family_lowering_ir.py`

### Part II

**Working title:** `Trellis II: Mathematical and Computational Lanes for Semantic Derivatives Pricing`

**Main question:** What mathematical and computational abstractions let one
semantic request compile onto multiple pricing families without collapsing into
product-specific formula sprawl?

**Primary thesis:** Trellis works because it narrows rich financial semantics
onto bounded numerical families rather than forcing one universal solver IR.

**Core sections:**

1. Why bounded family IRs beat a monolithic universal pricing IR.
2. Contract algebra and analytical support:
   contracts as syntax, valuation as semantics, sound rewrites, reusable
   kernels.
3. Event/control programs as the bridge from product semantics to numerics.
4. Family lanes:
   analytical, exercise lattice, PDE, event-aware Monte Carlo, transforms,
   basket credit / copula.
5. Calibration and market-binding surfaces as governed numerical context.
6. Case studies:
   transform family hardening, event-aware Monte Carlo, callable/short-rate
   routes, basket credit.
7. Honest support boundary:
   use `LIMITATIONS.md` to mark what remains incomplete.

**Innovation angle:** The novelty is the composition of semantic contracts,
family IRs, and deterministic kernels across multiple pricing regimes.

**Primary repo sources:**

- `docs/design_analytical_support_contract_algebra.md`
- `docs/quant/pricing_stack.rst`
- `doc/plan/done__event-aware-monte-carlo-lane.md`
- `doc/plan/done__transform-family-ir-and-admissibility-hardening.md`
- `doc/plan/done__semantic-contract-registry-and-short-rate-claim-generalization.md`
- `LIMITATIONS.md`
- `trellis/agent/family_lowering_ir.py`
- `trellis/agent/lane_obligations.py`
- `trellis/models/`

### Part III

**Working title:** `Trellis III: Governed Knowledge, Reflection, and the Path to a Learning Pricing Agent`

**Main question:** How should a pricing agent improve over time without
letting free-form learning corrupt a numerical library?

**Primary thesis:** Trellis already has a governed memory and reflection
architecture; the real contribution is disciplined retrieval, promotion, and
auditability, with full autonomous learning presented as an explicit next
phase rather than a completed claim.

**Core sections:**

1. Why a pricing agent needs governed memory rather than raw conversational
   history.
2. Knowledge assets:
   principles, decompositions, cookbooks, contracts, requirements, lessons,
   import registry, traces.
3. Retrieval and prompt grounding:
   `KnowledgeStore`, prompt formatting, anti-hallucination controls.
4. Reflection and promotion:
   post-build reflection, lesson capture, cookbook candidates,
   promotion candidates, lesson-to-test seam.
5. Platform traces, remediation, and reliability measurement.
6. What is implemented now versus what is not yet true.
7. Roadmap to the real learning loop:
   durable improvement, automatic regression materialization, scorer training,
   validator promotion, and evidence standards.

**Innovation angle:** The contribution is governance for agent improvement in
quant software, not merely another memory feature.

**Primary repo sources:**

- `docs/platform_loop_workstream.md`
- `doc/plan/active__backlog-burn-down-execution.md`
- `doc/plan/draft__lesson-store-refactor.md`
- `trellis/agent/knowledge/store.py`
- `trellis/agent/knowledge/retrieval.py`
- `trellis/agent/knowledge/reflect.py`
- `trellis/agent/knowledge/promotion.py`
- `trellis/agent/knowledge/autonomous.py`
- `trellis/agent/task_runtime.py`

## Cross-paper narrative

Each paper should defend one layer, while reusing a shared vocabulary.

### Shared vocabulary

- `PlatformRequest`
- `SemanticContract`
- `ValuationContext`
- `ProductIR`
- `EventProgramIR`
- `ControlProgramIR`
- family IR
- admissibility
- helper-backed numerical route
- knowledge trace
- promotion candidate

### Cross-paper dependency

- Part I defines the compiler and runtime vocabulary.
- Part II explains why the compiler can target multiple mathematical lanes.
- Part III explains how the system remembers, critiques, and eventually
  improves those lanes without weakening deterministic guarantees.

## Two-paper fallback

If you want only two papers, keep Part I unchanged and merge Parts II and III
into:

`Trellis II: Mathematical Substrate, Governed Knowledge, and the Path to Learning`

That merged version should still separate:

- shipped numerical/compiler architecture
- shipped knowledge governance
- future autonomous learning loop

## What not to overclaim

- Do not claim that Trellis already has a fully effective autonomous learning
  loop. The repo shows retrieval, reflection, promotion, traces, and some
  closed-loop seams, but your own note is right: the real execution of the
  learning loop is not yet the main mature contribution.
- Do not claim universal solver coverage. The system is intentionally built on
  bounded family IRs and helper-backed routes.
- Do not claim desk-complete risk, calibration, or xVA support; anchor those
  limits in `LIMITATIONS.md`.

## Artifact backlog

Store all figures, tables, and exported evidence under `docs/paper/artifacts/`.

### Figures to prepare

1. Unified request-to-outcome pipeline.
2. Semantic compile and lowering stack.
3. Family IR map: analytical / lattice / PDE / MC / transforms / credit.
4. Knowledge lifecycle: retrieve -> build -> reflect -> promote -> test.
5. Validation contract loop and trace surfaces.

### Tables to prepare

1. Front-door to canonical-request mapping.
2. Semantic object inventory and responsibility split.
3. Family IRs and supported numerical obligations.
4. Current proven families versus open limitations.
5. Knowledge asset types and lifecycle states.

### Case studies to prepare

1. Transform lane hardening (`T39`, `T40`).
2. Event-aware Monte Carlo and swaption comparison recovery (`T73`).
3. Short-rate helper extraction under callable/bond and Bermudan flows.
4. One lesson-to-test or promotion-candidate example for Part III.

## Recommended writing order

1. Finish Part I outline into prose first.
2. Draft Part II while the compiler vocabulary is still fresh.
3. Draft Part III last, after choosing exactly how candidly to frame the
   learning loop as present capability versus roadmap.

## Near-term execution plan

### Week 1

- lock the series thesis and paper titles
- export the first two core architecture figures
- write Part I introduction and system overview

### Week 2

- draft Part II sections on contract algebra and family IRs
- collect one numerical case study per lane you want to feature

### Week 3

- draft Part III with explicit "current state" and "future loop" split
- add the cross-paper introduction/conclusion language for a coherent series
