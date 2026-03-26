# Platform Loop Review: P1-P7

## Review

Before this tranche, Trellis had a strong middle layer:

- `ProductIR`
- route and primitive planning
- semantic validation
- blocker taxonomy
- structured user-defined product compilation

But the full platform loop was still split.

Different front doors still behaved differently:

- [ask.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/ask.py)
- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
- [pipeline.py](/Users/steveyang/Projects/steveya/trellis/trellis/pipeline.py)
- [user_defined_products.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/user_defined_products.py)
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)

The main gap was not the build core. It was the lack of one canonical request
and execution layer across those entrypoints.

## Plan

This tranche implemented the platform-loop phases as one additive pass:

1. canonical request model
2. unified request compiler
3. explicit execution plans
4. platform traces
5. remediation visibility
6. direct book integration
7. deterministic platform-loop evaluation

## Tests First

The red tests were centered in:

- [test_platform_loop.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py)

They required:

- ask / term-sheet requests compile to the correct direct-vs-build action
- blocked structured products compile to `block`
- direct book requests compile to `price_book`
- direct session greek requests compile to `compute_greeks`
- platform traces round-trip through disk

## Implementation

Core new modules:

- [platform_requests.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/platform_requests.py)
- [platform_traces.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/platform_traces.py)

Key integrations:

- [ask.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/ask.py)
  - ask requests now compile through the canonical request/compiler layer
- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
  - direct price / greeks / analytics requests can emit canonical platform requests
  - direct execution now records platform traces best-effort
- [pipeline.py](/Users/steveyang/Projects/steveya/trellis/trellis/pipeline.py)
  - direct book execution can compile a request and emit a platform trace
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
  - free-form build path now consumes the canonical compiled build request
- [quant.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/quant.py)
  - adds `select_pricing_method_for_product_ir(...)`
- [model_validator.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/model_validator.py)
  - adds `validate_model_for_request(...)`
- [remediate.py](/Users/steveyang/Projects/steveya/trellis/scripts/remediate.py)
  - platform trace summaries are now visible to remediation analysis

## Validation

Focused front-door/compiler slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_user_defined_products.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_session.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_pipeline.py
```

Result:

- `74 passed`

Broader deterministic regression gate:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_user_defined_products.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_ir_retrieval.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_semantic_validation.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_evals.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_scoring.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_session.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_pipeline.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_market_snapshot.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_resolver.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_engine/test_pricer.py
```

Result:

- `122 passed`

The full `tests/test_agent` sweep was not used as the tranche gate because the
older slow build-validation path still includes non-deterministic/slow behavior
that is not specific to this change.

## Outcome

The whole platform loop is now structurally unified:

- prompt-driven requests
- direct session requests
- direct pipeline/book requests
- structured user-defined products
- free-form build requests

all pass through the same canonical request/compiler vocabulary.

This does not mean every downstream subsystem is fully optimized yet, but it
does mean Trellis now has one coherent request-to-trace backbone instead of a
set of disconnected front doors.
