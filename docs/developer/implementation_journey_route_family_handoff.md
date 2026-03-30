# Design Note: Official Route Family Planner Rule

This note is the contract for future agents that touch pricing-task routing.
The failures were mostly contract drift, not bad math, so the durable fix is
to keep the planner, prompts, validator, and API map aligned on exact route
families.

## Planner Rule

Use exact route-family labels when the product semantics are known:

- `equity_tree` means American/Bermudan equity tree routing
- `rate_lattice` means callable/puttable/Bermudan rate routing

Those labels are not interchangeable. They exist so the planner can keep the
equity tree branch and the rate-lattice branch separate even though both live
under the broader tree family.

## Enforcement Surface

The rule is enforced in code, not just described here:

- [trellis/agent/knowledge/decompose.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py) computes `ProductIR.route_families`
- [trellis/agent/codegen_guardrails.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py) scores routes using exact route families
- [trellis/agent/semantic_validation.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/semantic_validation.py) maps exact imports to exact families
- [trellis/agent/knowledge/canonical/api_map.yaml](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/canonical/api_map.yaml) records the authoritative submodule paths
- [trellis/agent/prompts.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/prompts.py) shows the exact route family and import paths to generated code

The tests that lock this down are:

- [tests/test_agent/test_decomposition_ir.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_decomposition_ir.py)
- [tests/test_agent/test_route_scoring.py](/Users/steveyang/Projects/steveyang/trellis/tests/test_agent/test_route_scoring.py)
- [tests/test_agent/test_api_map.py](/Users/steveyang/Projects/steveyang/trellis/tests/test_agent/test_api_map.py)
- [tests/test_agent/test_semantic_validation.py](/Users/steveyang/Projects/steveyang/trellis/tests/test_agent/test_semantic_validation.py)

## Fallback Policy

The planner is intentionally conservative when it cannot prove the exact
family:

- exact submodule imports resolve to the exact family
- umbrella `trellis.models.trees` imports may fall back to coarse validation
- unknown tree symbols are treated as ambiguous rather than guessed

That fallback is only a bridge for legacy or partially migrated code. The
preferred contract is always the exact submodule path.

## What Was Fixed

The supporting surfaces were updated to agree with the rule:

- the compact API map points at exact submodules instead of the umbrella tree package
- planner scoring and route ranking use the exact route families
- prompt text shows the exact route family when it is known
- semantic validation recognizes the exact tree submodules
- thin adapters under `trellis.instruments._agent/*` are treated as wrappers
  around shared implementations, not as full numerical kernels

## Files To Inspect First

- [trellis/agent/knowledge/decompose.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py)
- [trellis/agent/codegen_guardrails.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py)
- [trellis/agent/semantic_validation.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/semantic_validation.py)
- [trellis/agent/knowledge/canonical/api_map.yaml](/Users/steveyang/Projects/steveyang/trellis/trellis/agent/knowledge/canonical/api_map.yaml)
- [trellis/agent/prompts.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/prompts.py)

## What The Next Agent Should Check

1. Search for any remaining generic `trellis.models.trees` assumptions in prompt text,
   cookbook guidance, or route selection code.
2. Verify that any new thin adapters are only exempted from primitive validation
   when they truly delegate into a shared implementation.
3. Keep the exact route-family split stable. Do not collapse `equity_tree` and
   `rate_lattice` back into a single tree bucket.
4. Continue the knowledge-manifest / provenance work so route guidance, API maps,
   and lesson indexes are treated as compiled artifacts with freshness tracking.

## Current Confidence

This is a stable fix, not a one-off patch. The behavior is enforced by code
and seam tests, including the generated rate-tree artifacts.
