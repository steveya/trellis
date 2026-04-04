Audit And Observability
=======================

Trellis keeps an appendable audit trail for compiled platform requests and can
optionally mirror notable lifecycle events into GitHub or Linear issues.

Governed Model Registry
-----------------------

`trellis.platform.models` is now the governed lifecycle boundary for executable
model identity. It persists:

- `ModelRecord` for the typed match basis and model-level lifecycle summary
- `ModelVersionRecord` for contract, methodology, validation, lineage, and artifact metadata
- explicit lifecycle-transition history for `draft`, `validated`, `approved`, and `deprecated`

This registry is intentionally separate from the research-side audit trail in
`trellis.agent.model_audit`. A successful build audit can support a draft or
validated candidate, but it does not imply governed approval.

Validation evidence is now explicitly version-scoped:

- a newly persisted model version does not inherit validation state from its parent
- promotion reads the latest deterministic validation result for that exact version
- metadata-only revisions still receive their own version-scoped code sidecar so
  later audit review can inspect the implementation associated with that
  specific version id

The registry now also owns execution-time lifecycle gating:

- sandbox default bundles may execute `draft`, `validated`, or `approved` versions
- research default bundles may execute `validated` or `approved` versions
- production default bundles may execute `approved` versions only
- `deprecated` versions are rejected by the default governed gate

That split matters because lifecycle state is now an execution decision, not
just a field that later reporting surfaces happen to show.

Canonical Run Ledger
--------------------

`trellis.platform.runs` is now the local-first source of truth for governed run
history. It persists:

- `RunRecord` for request identity, runtime context, provider bindings, snapshot id, selected model/engine, warnings, and result summary
- `ArtifactReference` entries that link traces, model-audit files, task-run records, and diagnosis packets back to the canonical run record

This changes the ownership boundary:

- platform traces remain useful append-only artifacts for request replay
- task-run records remain useful task-runtime views
- model-audit files remain useful research/build artifacts
- the run ledger is the canonical place that says which of those artifacts belong to a governed run

Canonical Audit Bundle
----------------------

``trellis.platform.audits`` now builds the governed review package that sits on
top of ``RunRecord`` and its linked artifacts.

The canonical ``RunAuditBundle`` sections are:

- ``run`` for run id, request id, action, status, run mode, session id, and policy id
- ``inputs`` for request identity, parsed semantic contract summaries, provider bindings, snapshot id, and valuation timestamp
- ``execution`` for selected model and engine identity, route provenance, validation summary, and policy outcome
- ``outputs`` for the canonical result summary
- ``diagnostics`` for warnings, blocker codes, trace status and events, and failure context for blocked or failed runs
- ``artifacts`` for linked artifact refs plus normalized summaries of attached trace, model-audit, task-run, and diagnosis artifacts

Operationally, that means later library and MCP surfaces can ask one builder
for audit state instead of re-synthesizing provenance from raw YAML traces and
task-run files themselves. Blocked runs are first-class here: a policy-denied
or otherwise blocked run still produces the same audit-bundle shape with
structured blocker and failure context.

The first MCP read tools now consume these canonical surfaces directly:

- ``trellis.run.get`` returns the stored ``RunRecord`` projection for one run id
- ``trellis.run.get_audit`` returns the canonical ``RunAuditBundle`` for that
  same run id

The same governed stores now also back explicit review and replay flows:

- ``trellis.model.persist`` writes new model versions with explicit lineage and
  stable sidecar artifacts instead of editing governed versions in place
- ``trellis.model.versions.list`` and ``trellis.model.diff`` project version
  history and code/contract/methodology/validation diffs directly from that
  canonical model store
- ``trellis.snapshot.persist_run`` writes a reproducibility bundle snapshot
  containing the persisted run summary, selected model/engine, market snapshot,
  output payload, tolerances, random seed, and calendars, then links it back to
  the ``RunRecord`` as a canonical artifact reference

Those same canonical stores now sit behind stable MCP resources:

- ``trellis://runs/{run_id}`` and ``.../audit`` read the stored run and audit views
- ``trellis://runs/{run_id}/inputs`` and ``.../outputs`` expose the stable
  replay-oriented projections without rerunning the valuation
- ``trellis://models/{model_id}``, ``.../versions``, and version-sidecar URIs
  expose the governed model-review surface, including the exact code,
  validation report, and lineage associated with a requested version
- ``trellis://market-snapshots/{snapshot_id}`` reads either the original
  governed market snapshot or the later reproducibility bundle snapshot written
  for replay and review

Those tools do not reconstruct status from ad hoc trace scraping. If the run is
persisted in the ledger, the MCP layer reads the ledger and audit builder; if
the run id is unknown, it fails with a structured MCP error instead of a raw
filesystem exception.

Governed Provider Provenance
----------------------------

``trellis.platform.providers`` is now the governed market-data entry point for
the migration path. It provides:

- ``ProviderRecord`` and ``ProviderRegistry`` for stable market-data provider ids
- explicit bound-provider resolution instead of convenience ``source=...`` selection
- canonical ``provider_id`` and ``snapshot_id`` fields in ``MarketSnapshot`` provenance

Operationally, that means:

- governed traces and run records can refer to a stable provider id instead of a
  human-oriented source label alone
- governed snapshot provenance now has a durable handle even before the future
  snapshot resource layer exists
- silent mock fallback is no longer allowed on governed paths; a mock provider
  has to be explicitly bound and policy-permitted

Governed Policy Outcomes
------------------------

``trellis.platform.policies`` now provides the deterministic runtime guard
layer for governed execution. It adds:

- ``PolicyBundle`` for explicit sandbox, research, and production policy defaults
- ``PolicyEvaluation`` for a structured allowed/blocked result
- ``PolicyBlocker`` records for stable blocker codes, requirement names, and field-level details

This matters operationally because:

- policy failures no longer need to surface as ad hoc resolver or executor exceptions
- the run ledger can persist structured ``policy_outcome`` payloads directly
- later executor, MCP, and audit-bundle work can reuse the same blocker codes
  instead of inventing parallel policy reporting

Platform Traces
---------------

``trellis.agent.platform_traces`` persists trace summaries under
``trellis/agent/knowledge/traces/platform`` as ``<request_id>.yaml`` plus an
adjacent append-only ``<request_id>.events.ndjson`` lifecycle log. Together
they capture:

- request metadata such as entry point, request type, action, and measures
- semantic request metadata such as the compiled semantic contract, selected DSL route, family IR summary, helper bindings, and structured lowering errors when present
- routing outcomes such as selected method, blocker codes, and whether a build was required
- compact knowledge summaries from the shared retrieval payload
- append-only lifecycle events with timestamps and free-form details
- optional external issue references when GitHub or Linear sync is enabled

Use ``load_platform_traces()`` for the cheaper summary view,
``load_platform_trace_events()`` when full lifecycle history is needed, and
``summarize_platform_traces()`` for coarse action counts. Legacy single-file
YAML traces remain readable and are migrated to the split layout on the next
writer update.

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
