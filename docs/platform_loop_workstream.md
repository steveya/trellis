# Platform Loop Workstream

This note tracks the unification of the full Trellis platform loop:

`request -> semantic compile -> execution/build -> result -> trace -> learning`

## Goal

Trellis should stop treating prompt builds, direct pricing, structured product
compilation, and book execution as unrelated entrypoints.

Every front door should compile into the same internal shape:

- canonical `PlatformRequest`
- optional canonical `ComparisonSpec` for multi-method tasks
- optional `ProductIR`
- route / primitive selection
- `ExecutionPlan`
- trace artifact
- knowledge/remediation feedback

## Completed Platform Phases

### P1: Canonical Request Model

Implemented in:

- [platform_requests.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/platform_requests.py)

Adds:

- `PlatformRequest`
- `ExecutionPlan`
- `CompiledPlatformRequest`
- `ComparisonSpec`
- `ComparisonMethodPlan`

### P2: Unified Request Compiler

The request compiler now handles:

- ask / term-sheet requests
- direct `Session` instrument requests
- direct book / `Pipeline` requests
- structured user-defined products
- free-form build requests
- comparison/multi-method build requests

### P3: Unified Execution Planning

Requests now compile into explicit actions like:

- `price_existing_payoff`
- `price_existing_instrument`
- `compute_greeks`
- `analyze_existing_instrument`
- `price_book`
- `build_then_price`
- `compile_only`
- `compare_methods`
- `block`

### P4: Platform Traces

Implemented in:

- [platform_traces.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/platform_traces.py)

This records a canonical trace across front doors, not only the build/task
path.

### P5: Learning / Remediation Integration

The remediation layer now has access to platform trace summaries through:

- [remediate.py](/Users/steveyang/Projects/steveya/trellis/scripts/remediate.py)

### P6: Direct Book Integration

Implemented in:

- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
- [pipeline.py](/Users/steveyang/Projects/steveya/trellis/trellis/pipeline.py)

Direct library usage can now emit canonical platform requests and traces.

### P7: Platform Evaluation Gate

Covered by:

- [test_platform_loop.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py)
- existing ask / session / pipeline / user-defined / quant / semantic tests

## Structured Task / Learning Loop Hardening

The platform loop now also covers the remaining task-runner and reflection
seams that were still operating in a more legacy style.

Implemented in:

- [task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py)
- [autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
- [config.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py)
- [reflect.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/reflect.py)

This adds:

- active use of `TASKS.yaml` `construct`, `cross_validate`, and `new_component`
  metadata in the task runner
- end-to-end `preferred_method` routing from task runner into the knowledge-aware
  build path and canonical build compiler
- comparison-task fan-out for multi-method benchmark tasks, including concrete
  `cross_validate.internal` targets and optional analytical reference targets
- runtime comparison pricing for built methods, so benchmark tasks now fail if
  the numeric cross-validation fails even when the code generation itself
  succeeds
- task-level `market:` selection and `market_assertions:` so comparison/stress
  tasks can resolve named mock-market components and fail early on market
  selection mismatches
- explicit guardrails for empty text / empty JSON / invalid JSON model responses
- bounded OpenAI timeout/retry handling around text and JSON generation paths
- deterministic cookbook-candidate capture when a build succeeds but LLM
  reflection does not yield a reusable cookbook extract
- deterministic closed-loop coverage showing that promoted lessons can be
  captured and then retrieved on the next pass
- artifact propagation from build/reflection into task results, including:
  - executor/platform trace request ids and trace paths
  - knowledge trace paths
  - cookbook-candidate paths
  - knowledge-gap log paths
- enough structured artifact metadata for UI surfaces to render both the live
  run summary and the raw YAML-backed documents behind it

## Current Architecture

The platform loop is now:

1. front door creates a canonical request
2. compiler derives semantics, route, execution action, and comparison intent
3. execution/build runs through existing library or agent machinery
4. the result is traced
5. remediation and learning can inspect a shared trace surface

The shared trace surface is now visible through task/build results as well as
the filesystem. Task results can carry artifact references for:

- platform traces
- knowledge traces
- cookbook candidates
- lesson `source_trace` links

The UI surface in `trellis-ui` now uses those references directly:

- Task Monitor can render trace and cookbook-candidate references for a run
- blocked task rows can expand and show blocker/new-primitive workflow detail
- lesson IDs and trace tags can jump the user into the matching Knowledge view
- Knowledge polls while active so newly captured lessons/traces appear without a
  full app reload

## Canonical Capability Vocabulary

The platform loop now treats market-data capability names as canonical in the
core/runtime/compiler layers. The target vocabulary is:

- `discount_curve`
- `forward_curve`
- `black_vol_surface`
- `local_vol_surface`
- `credit_curve`
- `fx_rates`
- `spot`
- `state_space`
- `jump_parameters`
- `model_parameters`

Legacy names are still normalized at ingestion boundaries when reading older
tasks, lessons, or ad hoc requirements, but they are no longer the canonical
names emitted by the core platform surfaces.

## Remaining Work

The loop is now unified structurally, but not fully optimized yet.

Remaining upgrades are:

- a dedicated stress-task tranche for mock-connector and comparison-task
  coverage, with manifest-backed deterministic preflight grading before adding
  task-level market specs
- richer request-driven market-data resolution
- better cookbook refinement for weak existing cookbooks, not only missing ones
- full use of compiled-request semantics inside the deeper model validator
- eventual reactive dataflow integration for execution and invalidation
