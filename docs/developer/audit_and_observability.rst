Audit And Observability
=======================

Trellis keeps an appendable audit trail for compiled platform requests and can
optionally mirror notable lifecycle events into GitHub or Linear issues.

Platform Traces
---------------

``trellis.agent.platform_traces`` persists YAML trace files under
``trellis/agent/knowledge/traces/platform``. Each trace captures:

- request metadata such as entry point, request type, action, and measures
- routing outcomes such as selected method, blocker codes, and whether a build was required
- compact knowledge summaries from the shared retrieval payload
- append-only lifecycle events with timestamps and free-form details
- optional external issue references when GitHub or Linear sync is enabled

Use ``load_platform_traces()`` to read traces back into structured
``PlatformTrace`` objects and ``summarize_platform_traces()`` for coarse action counts.

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

- the critic and arbiter path, documented in :doc:`../agent/critic_agent`, which focuses on generated code correctness and invariant checks
- the model-validator path in ``trellis.agent.model_validator``, which emits structured
  ``ValidationReport`` findings for conceptual, calibration, sensitivity, benchmark, and limitation concerns

That split is useful operationally: code may compile and satisfy invariants while
still failing model-risk review.

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

