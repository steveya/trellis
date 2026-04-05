Task And Eval Loops
===================

The task and evaluation layer is how Trellis measures whether the agent and
knowledge system are actually improving over time.

Task Corpus
-----------

``TASKS.yaml`` is the canonical pricing-task manifest, while
``FRAMEWORK_TASKS.yaml`` holds non-priceable framework/meta tasks.
``trellis.agent.task_runtime`` turns pricing-task entries into offline-ready
execution contexts, market states, method plans, and benchmarkable generated
modules.

The runtime helpers cover:

- loading task ranges and statuses
- building default or task-specific mock market states
- reusing generated modules when available
- benchmarking cached task payoffs
- normalizing task descriptions into request/build inputs

The non-integration pytest surface is also grouped into explicit reviewable
strata:

- ``crossval`` for independent-library cross-checks
- ``verification`` for numerical or analytical reference tests
- ``global_workflow`` for user-facing workflow tests that span modules
- ``legacy_compat`` for deprecated or compatibility-only behavior

Operational Scripts
-------------------

The repo ships a small task-operations toolchain:

- ``scripts/run_tasks.py``: run a contiguous task block or all pending tasks
- ``scripts/rerun_ids.py``: re-run specific task ids
- ``scripts/benchmark_tasks.py``: benchmark cached generated payoffs without rebuilding
- ``scripts/remediate.py``: analyze failures, categorize knowledge gaps, and patch common knowledge issues
- ``scripts/evaluate_shared_memory.py``: compare two task-result tranches and render a shared-memory improvement report

Evals And Stress Tasks
----------------------

``trellis.agent.evals`` provides deterministic graders around the build loop:

- generation-plan grading
- generated-import and semantic validation grading
- task-result classification and summarization
- stress-task preflight checks against the manifest in ``tests/evals/stress_tasks.yaml``
- baseline-versus-candidate comparison reports for shared-memory analysis

This is the main developer-facing evidence loop for changes to routing,
knowledge retrieval, or generation guardrails.

Task-run artifacts now preserve analytical trace paths alongside the existing
platform traces. When a build loop emits an analytical route, the stored task
record keeps both the JSON trace and the rendered Markdown path, including
reused or cached analytical routes that short-circuit before code generation,
so replay, diagnosis, and benchmark tooling can inspect the same build history
without rerunning the build.

Every task run also writes a canonical diagnosis packet and Markdown dossier
under ``task_runs/diagnostics/``. The packet is the structured source of truth
for run diagnosis; the dossier is the human-facing view that opens first when
you need to understand a failure or confirm a success. The batch runner now
surfaces those packet paths directly on the task result so you do not have to
reconstruct them from traces or summary files.

The task record and diagnosis packet also preserve post-build checkpoint
metadata. That metadata records the transition from a validated method build
into reflection, token-usage attachment, decision-checkpoint emission, and
background consolidation so silent post-build stalls can be bisected without
adding ad hoc logging.

Review policy is now explicit about critic cost. Standard validation keeps the
deterministic gates authoritative and treats critic as a bounded residual-risk
reviewer for high-risk routes. In practice that means standard runs use an
``advisory`` critic mode with a single JSON attempt and no JSON-to-text
fallback chain. The critic now selects from a bounded menu of deterministic
``check_id`` values instead of emitting open-ended review code on the standard
path. Thorough validation keeps the broader reviewer path for routes that
still need deep conceptual review.

Route-specific invariant packs now carry more of the rejection load. For
single-name CDS routes, deterministic validation checks quote normalization
and hazard sensitivity before generic price-sanity heuristics, so spread-unit
mistakes are rejected as contract violations instead of being left to a slow
reviewer stage.

For earlier request/build phases, set ``TRELLIS_LLM_WAIT_LOG_PATH`` during a
rerun. That emits a JSONL timeline of bounded LLM waits, keyed by stage and
request metadata, and the configured path is echoed into the task result and
diagnosis dossier so the run record points back to the live wait log.

For comparison tasks, the top-level task result also aggregates nested method
failures into a single failure list before remediation runs. That keeps the
analysis loop from losing the actual timeout, import, or implementation-gap
message behind a method-level success/failure split.

Analytical traces also carry a structured instruction-lifecycle payload. The
``GenerationPlan`` now includes a resolved instruction set, the trace emits a
dedicated ``instruction_lifecycle`` step, and the persisted task-run summary
surfaces the effective, dropped, and conflicting instruction counts. That
means route guidance is visible as structured trace data instead of only
appearing as rendered prompt prose.

Semantic traces now carry the same kind of concept-level metadata. The
semantic contract summary includes the canonical concept registry entry,
including the semantic id, version, compatibility wrappers, and extension
policy. The registry distinguishes product contracts from supporting atoms and
market-input concepts, so gap and extension traces can tell whether the request
reused an existing concept, needed a new attribute on that concept, fell back
to a thin wrapper, or pointed at a genuinely new concept. Supporting atoms are
only surfaced when the request is actually about that semantic layer rather
than just a novel product request that mentions the atom as a secondary detail.

Semantic role ownership is recorded on top of that contract metadata. The
request metadata and platform traces now carry a stage-specific ownership
summary that makes the handoff explicit: gap classification, bounded quant
route assembly, payoff/model validation, and trace handoff are all owned by
different roles, and quant/model_validator stay bounded to assembly and
validation instead of inventing new semantic grammar. If no role can safely
own a stage, the metadata marks the extension as fail-closed instead of
silently continuing.

For analytical FX routes, the trace now also makes the basis assembly visible:
the FX vanilla route records the forward bridge, the Black76 basis claims, and
the terminal payoff assembly step so the knowledge system can distinguish a
thin adapter from an explicit analytical decomposition. In practice the trace
reads like a small basis-assembly proof: map spot and curves to a forward,
evaluate the reusable terminal basis claims, assemble the vanilla payoff, and
then record the final PV and validation checks.

The same provenance rule applies to multi-curve rates work: when a snapshot is
compiled into a runtime market state, Trellis keeps the selected curve names
alongside the curve objects and repeats that provenance in the runtime
contract, task result, and persisted task-run record so later debugging can
tell which named discount or forecast curve was actually chosen.

If the market snapshot was built from bootstrapped rate instruments, the
resolver keeps that assembly step visible too: named bootstrap buckets become
named curves in the snapshot, and the chosen curve name is preserved in the
runtime state and replay metadata for validation.

Suggested Validation Order
--------------------------

When changing platform behavior:

1. run the targeted unit tests for the affected area
2. run the relevant task block or rerun ids
3. inspect remediation analysis if failures cluster by pattern
4. compare candidate task results to the baseline when the change is meant to improve the platform

Useful commands:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py T13 T24
   /Users/steveyang/miniforge3/bin/python3 scripts/rerun_ids.py T54 T62
   /Users/steveyang/miniforge3/bin/python3 scripts/remediate.py --analyze-only
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "crossval and not integration"
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "verification and not integration"
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "global_workflow and not integration"

Related Reading
---------------

- :doc:`audit_and_observability`
- :doc:`../quant/knowledge_maintenance`
