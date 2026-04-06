# Policy vs Live Knowledge

This note defines the split used by the Trellis agent and knowledge system.

For an end-to-end picture of how these layers interact with planning, code
generation, and semantic validation, see
[`workflow_diagrams.md`](./workflow_diagrams.md).

## Policy

Policy is the stable guidance that should remain canonical across runs and
should change only through deliberate review.

In Trellis, policy lives primarily in:

- `trellis/agent/knowledge/canonical/principles.yaml`
- `trellis/agent/knowledge/canonical/features.yaml`
- `trellis/agent/knowledge/canonical/decompositions.yaml`
- `trellis/agent/knowledge/canonical/cookbooks.yaml`
- `trellis/agent/knowledge/canonical/method_requirements.yaml`
- `trellis/agent/knowledge/canonical/data_contracts.yaml`
- `trellis/agent/knowledge/canonical/api_map.yaml`

Examples of policy:

- method-family naming and compatibility aliases
- package placement rules
- cookbook templates
- modeling requirements
- anti-hallucination rules
- canonical instrument decompositions

## Live knowledge

Live knowledge is repo state or accumulated operating memory that changes
frequently and should not become the authoritative policy source.

In Trellis, live knowledge includes:

- `trellis/agent/knowledge/lessons/`
- `trellis/agent/knowledge/traces/`
- `task_results_*.json`
- generated import-registry output
- current repo tree and current exported symbols

Examples of live knowledge:

- recent failures
- newly captured lessons
- transient traces
- current module graph
- active task outcomes

## Rule

When policy and live knowledge disagree:

1. inspect the canonical policy files first
2. use live knowledge as evidence, not as the authority
3. update canonical policy deliberately instead of encoding behavior in an
   ad hoc Python table or stale prompt fragment

## Compatibility

Some Python modules still expose thin compatibility surfaces for older callers,
but policy should still come from canonical YAML rather than hand-maintained
Python copies.

Current examples:

- `trellis.agent.quant` loads plans from canonical decompositions
- builder and review flows read cookbook guidance from canonical cookbook YAML
- diagnostic helpers and retry feedback draw related guidance from canonical
  lesson and failure-signature records

## Import and Inspection Guardrails

Tranche 2B adds a second layer on top of canonical policy: live repo-backed
import validation.

The intent is narrow:

- prompts may describe preferred modules and patterns
- canonical YAML still defines method families, cookbook structure, and policy
- but generated `trellis.*` imports must now resolve against the live import
  registry before code is written to disk

In practice this means:

- the agent should inspect modules, exports, tests, and lessons before editing
- the builder receives an explicit approved-module set derived from the
  selected method family plus inspected references
- unknown imports are rejected
- wildcard imports are rejected
- existing but unapproved Trellis modules are rejected until the generation
  plan is widened deliberately

This keeps policy and live knowledge in their proper roles:

- policy says what kind of module placement and method structure is correct
- live knowledge says which modules and symbols actually exist today
