# Phase 11 Review: User-Defined Product Workflow

## Review

By the end of Phase 10, Trellis could:

- decompose free-form product descriptions into `ProductIR`
- rank and select deterministic routes
- surface structured blocker reports
- produce concrete new-primitive workflows for unsupported products

What it still lacked was a clean user-facing compile path for **structured**
user-defined derivatives. There was no way to hand Trellis a typed product spec
and ask:

- what `ProductIR` does this correspond to?
- what route would Trellis use?
- is it supported by current machinery?
- if not, what blocker/workflow would be produced?

## Plan

Phase 11 should introduce a separate structured product-compile path, rather
than overloading the existing free-form `build_payoff(...)` entrypoint.

The workflow should be:

1. parse a YAML/dict structured product spec
2. convert it into a deterministic `ProductIR`
3. derive a `PricingPlan`
4. reuse `GenerationPlan`, blocker taxonomy, and new-primitive workflow
5. retrieve the same knowledge context the builder would see

## Tests First

Red tests were added for:

- parsing a YAML user-defined product spec
- compiling a supported callable fixed-income product into an existing lattice
  exercise route
- compiling a blocked stochastic-volatility composite into a blocker report and
  new-primitive workflow

## Implementation

New module:

- [/Users/steveyang/Projects/steveya/trellis/trellis/agent/user_defined_products.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/user_defined_products.py)

It introduces:

- `UserProductSpec`
- `UserDefinedProductCompilation`
- `parse_user_product_spec(...)`
- `compile_user_defined_product(...)`

To avoid duplicating decomposition logic, Phase 11 also adds:

- `build_product_ir(...)` in
  [/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py)

That helper reuses the same normalization and inference rules already used by
the text-based decomposition path, but starts from explicit semantic fields.

The compile step then reuses the existing stack:

- `ProductIR`
- `PricingPlan`
- `GenerationPlan`
- `BlockerReport`
- `NewPrimitiveWorkflow`
- knowledge retrieval / prompt formatting

So this is not a second planning system. It is a structured entrypoint into the
same planning system.

## Validation

Focused Phase 11 slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_user_defined_products.py
```

Broader regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_user_defined_products.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_new_primitive_workflow.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_blocker_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_learning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_scoring.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_ir_retrieval.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_evals.py
```

Result:

- focused slice: `3 passed`
- broader regression: green

## Outcome

Phase 11 gives Trellis the first clean user-defined derivative workflow:

- supported structured products compile to existing machinery
- unsupported structured products compile to blocker/workflow output

That is the first end-to-end second-half proving ground. It demonstrates that
the current architecture can now accept **typed product semantics from the
user**, not only heuristically inferred semantics from a prompt.
