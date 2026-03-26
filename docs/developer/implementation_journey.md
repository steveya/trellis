# Trellis Implementation Journey

These notes are the higher-level narrative of how Trellis evolved from a
promising pricing library with an LLM sidecar into a more coherent request
compiler, knowledge-backed build system, and auditable task-learning platform.

They are intentionally different from the smaller `refactor_*.md` notes and
the workstream plans:

- the `refactor_*.md` files capture one tranche at a time
- the `*workstream.md` files capture forward-looking plans
- this set captures the reasoning arc, inflection points, and the parts of the
  system that changed the shape of the platform

## Reading Order

1. [Prompt to Price](./implementation_journey_prompt_to_price.md)
   The operational path from request surface to execution, validation, and
   pricing.
2. [Knowledge System](./implementation_journey_knowledge_system.md)
   How isolated agent memory became a shared substrate.
3. [Learning and Feedback Loop](./implementation_journey_learning_feedback_loop.md)
   How tasks, traces, issues, and reruns became the main development engine.
4. [Token Efficiency](./implementation_journey_token_efficiency.md)
   How cost and token pressure forced a more deterministic, staged design.

## The Main Arc

Trellis did not become one system in a single refactor. It evolved in layers.

### 1. The build core became stronger before the front doors were unified

The first meaningful shift was in the build core. `executor.py` stopped being
just a thin LLM wrapper and started doing more structured work:

- `ProductIR` decomposition
- route and primitive planning
- blocker classification
- import validation
- semantic validation

That made the center of the system stronger even while the top-level entry
surfaces were still mixed.

### 2. The front doors were pulled into a canonical request/compiler layer

Once the core was robust enough, the next step was to stop treating
`ask(...)`, `Session`, `Pipeline`, structured product definitions, and task
runs as unrelated interfaces. The platform request/compiler layer made them
share one internal shape and one trace surface.

### 3. Market data became a real subsystem instead of loose runtime fields

The market-data workstream introduced `MarketSnapshot`, named components,
resolver APIs, and simulated provider support. That was necessary because a
pricing platform cannot become auditable or replayable if its market context is
still implicit.

### 4. The task runtime became the platform's proving ground

`TASKS.yaml`, reruns, shared-memory comparison reports, stress tasks, issue
creation, and UI audit trails turned the task loop into more than benchmarking.
It became the mechanism by which Trellis learns what the platform is missing.

### 5. Token pressure forced architectural discipline

Once realistic task batches started consuming meaningful tokens, it became
clear that prompt engineering alone was not enough. Trellis needed:

- stage-aware model hierarchies
- token telemetry
- token budgets
- compact-first prompt surfaces
- deterministic fast paths
- failure-type-aware retries

That work ended up improving the architecture, not just reducing cost.

## How To Use These Notes

Use this set when one of these questions comes up:

- Why does the platform look the way it does now?
- Which parts were deliberate and which are still transitional?
- What did we learn from repeated task reruns?
- Where should the next autonomous-library-development milestone start?

Use the workstream notes when you need the live action plan:

- [Autonomous library development workstream](../autonomous_library_development_workstream.md)
- [Platform loop workstream](../platform_loop_workstream.md)
- [Market data workstream](../market_data_workstream.md)

## Current Position

The platform is now strong enough to support a real next milestone:

- repeated reruns are auditable
- the agents share more memory than before
- market-data plumbing is no longer purely ad hoc
- the UI and task-run store can explain why a task passed or failed
- token usage is visible enough to optimize intentionally

But it is not yet fully autonomous. The missing-pricing-infrastructure loop is
still supervised by the human "Library Developer" role, even though the system
can already classify blockers, preserve evidence, and create tracked work.

That is the bridge these notes are meant to document.
