Audit And Observability
=======================

Trellis keeps an appendable audit trail for compiled platform requests and can
optionally mirror notable lifecycle events into GitHub or Linear issues.

Platform Traces
---------------

``trellis.agent.platform_traces`` persists YAML trace files under
``trellis/agent/knowledge/traces/platform``. Each trace captures:

- request metadata such as entry point, request type, action, and measures
- semantic request metadata such as the compiled semantic contract, selected DSL route, family IR summary, helper bindings, and structured lowering errors when present
- routing outcomes such as selected method, blocker codes, and whether a build was required
- compact knowledge summaries from the shared retrieval payload
- append-only lifecycle events with timestamps and free-form details
- optional external issue references when GitHub or Linear sync is enabled

Use ``load_platform_traces()`` to read traces back into structured
``PlatformTrace`` objects and ``summarize_platform_traces()`` for coarse action counts.

For migrated semantic families, the trace payload is now deep enough to replay
the route-selection decision:

- ``semantic_contract`` describes the validated typed contract boundary
- ``semantic_blueprint`` includes the selected DSL route and route family
- ``dsl_family_ir`` records the checked family-level lowering summary
- ``dsl_target_bindings`` shows the concrete helper and market-binding targets
- ``dsl_lowering_errors`` preserves structured rejection codes without parsing free text
- ``validation_contract`` records the deterministic check set, normalized
  comparison relations, and residual-risk summary that governed validation
- validation-stage events record the resolved review-policy reason so replay can
  show whether critic/model-validator escalation came from compiled contract
  state or only from the fallback route heuristic

Task-Run Telemetry
------------------

``trellis.agent.task_run_store`` now projects selected generated skills and
route-health observations into the persisted task-run record.

Use:

- ``load_latest_skill_telemetry_rollup()`` for artifact-level outcome counters
- ``load_latest_route_health_rollup()`` for route and route-family counters
- ``load_latest_skill_ranking_inputs()`` for the stable retained ranking metrics
- ``load_latest_route_ranking_inputs()`` for the route-level ranking metrics

These rollups are rebuilt from canonical run records rather than from ad hoc
trace scraping. Diagnosis dossiers also surface the same telemetry so a human
can move from batch failure to selected-skill attribution without opening raw
trace files first.

Issue Sync
----------

Two best-effort issue-sync adapters are built into the repo:

- ``trellis.agent.github_tracker`` creates issues and lifecycle comments through the GitHub REST API
- ``trellis.agent.linear_tracker`` creates issues and comments through the Linear GraphQL API

Both use the same trace-driven context and the shared formatting helpers in
``trellis.agent.request_issue_format``. They are intentionally non-blocking:
pricing or build execution should not depend on an external tracker being available.

Validation Outputs
------------------

There are two complementary validation surfaces:

- the compiled validation-contract path, which selects deterministic checks,
  relation-aware comparison semantics, and replayable harness requirements
- the critic and arbiter path, documented in :doc:`../agent/critic_agent`, which focuses on generated code correctness and invariant checks
- the model-validator path in ``trellis.agent.model_validator``, which emits structured
  ``ValidationReport`` findings for conceptual, calibration, sensitivity, benchmark, and limitation concerns

That split is useful operationally: code may compile and satisfy invariants while
still failing model-risk review.

The critic/arbiter path is now bounded by the compiled validation contract: the
critic can only emit allowed check families, and the arbiter only dispatches
those allowed check ids during the standard path. Reviewer prompts now also
receive the compact compiled route contract, so prompt replay can distinguish a
real route/helper drift from a generic code-quality complaint.

For thorough-mode model validation, the same contract now supplies the residual
conceptual-risk context so the LLM stage can focus on approximation and
limitation review instead of repeating deterministic checks.

Operational Guidance
--------------------

For developer-facing observability:

1. keep traces enabled for entry points that matter
2. treat GitHub/Linear sync as an outward mirror of the trace, not the source of truth
3. inspect knowledge summaries and blocker codes before diagnosing builder failures
4. keep high-severity validation findings visible in downstream tooling

Related Reading
---------------

- :doc:`hosting_and_configuration`
- :doc:`task_and_eval_loops`
- :doc:`../agent/architecture`
