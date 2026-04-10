Task Diagnostics
=================

This page explains the canonical per-task diagnosis packet and dossier that
Trellis now writes for every task run.

Why it exists
-------------

Before the packet, diagnosis was spread across:

- the raw batch JSON from ``scripts/run_tasks.py``
- the latest/history run records in ``task_runs/``
- platform and analytical traces
- remediation summaries
- learning artifacts and lessons

That made both human and agent diagnosis expensive. The packet collects the
important pieces into one structured artifact, and the dossier renders the
same data in a readable Markdown form.

What gets written
-----------------

For each task run, Trellis writes:

- a canonical JSON packet
- a human-readable Markdown dossier
- the existing run record in ``task_runs/history`` and ``task_runs/latest``

The packet and dossier live under:

- ``task_runs/diagnostics/history/<task_id>/<run_id>.json``
- ``task_runs/diagnostics/history/<task_id>/<run_id>.md``
- ``task_runs/diagnostics/latest/<task_id>.json``
- ``task_runs/diagnostics/latest/<task_id>.md``

The batch runner also surfaces those paths on the task result itself so a
caller can jump directly to the packet after a batch finishes.

Replay marker
-------------

Full-task canary replays now persist the same task-run and diagnosis artifacts
as live runs, but they also mark the execution surface explicitly:

- ``execution_mode`` is set to ``cassette_replay``
- ``llm_cassette`` records the cassette name, path, and replay policy

That metadata is stored on the top-level task result and the persisted task-run
record so diagnosis, remediation, and canary-summary tooling can tell live and
cassette-backed runs apart without needing a separate artifact schema.

For replay stability, record and replay full-task canary cassettes with
``PYTHONHASHSEED=0`` and keep them under ``cassettes/full_task/``.

How to read one
---------------

Start with the Markdown dossier. It is ordered for diagnosis:

1. Summary
2. Primary diagnosis
3. Comparison summary, if present
4. Method outcomes
5. Trace index
6. Learning summary
7. Skill telemetry
8. Evidence
9. Workflow and storage paths

If the dossier is still not enough, open the JSON packet next. The packet is
the canonical structured record and should contain the same evidence in a
machine-friendly shape.

Operational use
---------------

The packet is meant to shorten the feedback loop after a pricing batch:

- ``scripts/run_tasks.py`` writes the packet paths into the batch results
- ``scripts/remediate.py --analyze-only`` can be used alongside the packet to
  bucket failures
- future diagnosis work should refine the packet schema rather than creating
  another parallel report

Skill telemetry
---------------

Diagnosis packets now include a dedicated telemetry section sourced from the
persisted task-run record.

That section answers:

- which generated skills were selected for the run
- which audience surface consumed them
- what the normalized run outcome was
- whether the run required retries or ended in a degraded partial-success state
- which route or route family was involved, together with any recorded
  instruction-resolution counts

For maintenance tooling, the same data is also available as deterministic
rollups through ``trellis.agent.task_run_store``:

- ``load_latest_skill_telemetry_rollup()``
- ``load_latest_route_health_rollup()``

Canary batch telemetry
----------------------

Canary runs now persist one explicit batch record under
``task_runs/canary_batches/`` in addition to the per-task diagnosis artifacts.

That batch layer exists because the raw task-run store mixes ordinary ad hoc
task executions, replay-backed runs, and any historical one-off reruns. The
canary batch record gives tooling a trustworthy benchmark surface instead of
forcing it to infer canary lineage from task ids alone.

Each batch record captures:

- ``execution_mode`` so live and replay runs stay separate
- ``benchmark_eligible`` so replay-backed history can be excluded cleanly
- the batch scope (full curated set, subset, or single task)
- aggregate pass, elapsed-time, token, and attempt metrics
- per-canary links back to the underlying task-run and diagnosis artifacts

The deterministic loaders are:

- ``load_canary_batch_records()``
- ``load_canary_task_history()``

Use ``load_canary_task_history(..., benchmark_only=True)`` when you want a
live-only history surface for one canary without mixing in replay runs or
root-level pytest artifacts.

Connector-stress batch surfacing
--------------------------------

The standing connector-stress runner now treats the packet and dossier as the
primary per-task drill-down surface instead of a side artifact.

For ``scripts/run_stress_tranche.py`` this means the batch report should show,
for each task:

- outcome class (``compare_ready`` vs ``honest_block``)
- failure bucket
- observed blocker categories
- latest diagnosis dossier path
- latest diagnosis packet path
- any follow-on candidate derived from repeated failures

Operationally, the batch report is the front door and the per-task dossier is
the second click. If a stress rerun looks wrong, read the batch report first,
then open the linked dossier for the specific task instead of starting from the
raw batch JSON or trace directories.

Post-build checkpoints
----------------------

The diagnosis packet now carries a compact post-build summary for each run and,
when applicable, for each comparison method. This is meant to narrow the
silent gap between “the method validated” and “the task returned”.

The post-build phase markers currently cover:

- ``build_completed``
- ``reflection_started``
- ``reflection_completed``
- ``token_usage_attached``
- ``decision_checkpoint_emitted``
- ``consolidation_dispatched``

The dossier renders the latest post-build phase and a per-method checkpoint
summary so a human can quickly see whether the run failed before build
completion, during reflection, or after the build returned.

Runtime bisection flags
-----------------------

Three environment flags can temporarily narrow the post-build path during live
debugging:

- ``TRELLIS_SKIP_POST_BUILD_REFLECTION=1``
- ``TRELLIS_SKIP_POST_BUILD_CONSOLIDATION=1``
- ``TRELLIS_SKIP_TASK_DIAGNOSIS_PERSIST=1``

These are debugging controls, not normal operating modes. When enabled, the
task run record and diagnosis packet note the active flags so later diagnosis
can tell which parts of the runtime were intentionally bypassed.

LLM wait logs
-------------

Earlier-stage stalls can be diagnosed with ``TRELLIS_LLM_WAIT_LOG_PATH``:

- ``TRELLIS_LLM_WAIT_LOG_PATH=/tmp/t38_waits.jsonl``

When set, Trellis writes one JSON line per bounded LLM wait with:

- stage name
- request metadata such as ``task_id`` and ``comparison_target``
- timeout bound
- completion vs timeout outcome

The task dossier now records the configured wait-log path under ``Runtime
Controls`` so a human can jump directly from the packet to the live request
timeline during a rerun.

Reviewer latency
----------------

When a rerun looks quiet after a successful build, check the wait log before
assuming the task runtime is deadlocked. Standard validation now bounds the
critic path deliberately:

- critic mode ``advisory`` for high-risk standard routes
- one JSON attempt
- no JSON-to-text fallback chain

That means a slow critic should now show up as a single bounded wait and a
non-blocking ``critic_failed`` event, not as a long series of reviewer
retries. Thorough validation is still allowed to use the broader reviewer path
when deep conceptual review is the point of the run. On the standard path,
critic concerns should now carry deterministic ``check_id`` values plus
evidence and remediation, rather than critic-authored executable test code.

For credit-default-swap routes, the deterministic bundle now also emits more
specific failures before the generic sanity checks. If a run is mixing decimal
and basis-point spread quotes, or if the payoff ignores the credit curve, the
packet should now show a CDS-specific invariant failure rather than only a
large-PV heuristic.

For comparison-quality single-name CDS Monte Carlo runs, the typed spec now
also carries the path-count control explicitly. The planner specializes the
CDS Monte Carlo schema with ``n_paths``, and the ``T38`` canary pins a tighter
``2.0`` percent comparison tolerance around the resulting internal reference.
If the generated adapter hard-codes a smaller path count such as ``50000``
instead of flowing ``spec.n_paths``, expect a route-specific diagnosis packet
that fails on internal comparison spread long before reviewer escalation.

For analytical rate-style swaption routes, the deterministic bundle now also
checks helper consistency against the checked-in
``trellis.models.rate_style_swaption`` Black76 surface. If a generated adapter
compiles but misbinds annuity, forward swap rate, or notional, the packet
should now show a route-specific ``check_rate_style_swaption_helper_consistency``
failure with sampled scenario prices before the run escalates to critic or
model-validator review.

Eligible single-method routes now also emit a post-bundle
``reference_oracle_executed`` event before reviewer escalation. This is the
next checkpoint after the deterministic bundle for helper-backed exact or
bound-style routes:

- analytical swaptions against the checked-in Black76 helper
- analytical zero-coupon-bond options against the Jamshidian helper
- callable and puttable rate-tree bonds against the straight-bond bound helper

The packet should therefore show the oracle id, source, relation, tolerance,
sampled scenario prices, and maximum observed deviation when that checkpoint
runs. Comparison builds intentionally skip this oracle step because they
already carry explicit cross-method evidence.

Compiled primitive obligations
------------------------------

When the compiler resolves a request onto a checked helper-backed route, the
semantic-validator layer now treats that primitive plan as binding contract.
The diagnosis packet should therefore show a structured semantic-validator
failure when generated code ignores a required helper, even if the surrounding
module imports look plausible.

This is deliberate: prompt text may mention the helper, but the packet is now
the place to confirm whether the generated adapter actually satisfied the
compiled primitive obligation. For schedule-bearing routes, the same layer now
also surfaces raw string schedule fields before execution so timeline-typing
drift is visible in the packet rather than only as a later runtime failure.

Lane obligations
----------------

The generation boundary now also persists a compiler-emitted lane plan. This is
the constructive side of the semantic compiler:

- lane family
- timeline roles
- market requirements
- state and control obligations
- construction steps
- exact backend bindings, when they exist

When a build fails, this makes it easier to distinguish two cases:

- the agent ignored an exact backend binding that the compiler had already
  found
- the compiler intentionally emitted a constructive lane plan and the build
  failed while trying to synthesize that lane-level implementation

For tranche-2 synthesis proving, use
``scripts/run_knowledge_light_proving.py``. That runner forces a
compiler-first, knowledge-light prompt surface and still writes the standard
task diagnosis packet and dossier for each benchmark. In that mode the lane
card now also renders exact backend helper signatures when the compiler found
a safe checked binding. This matters for thin-adapter families such as CDS,
where the task may select the correct helper but still drift on keyword names
without the signature in view.

Runtime contract failures
-------------------------

Generated and checked-in payoffs are now evaluated through a contract-aware
``MarketState`` proxy inside ``trellis.engine.payoff_pricer.price_payoff``.
When a route indexes a missing named market binding or tries to use an
ambiguous scalar field that the runtime contract cannot supply honestly,
Trellis raises ``ContractViolation`` instead of surfacing a raw Python
``AttributeError`` or ``KeyError``.

In practice this means diagnosis packets and comparison-task failures should
now preserve the missing field or mapping key explicitly, for example a missing
``underlier_spots['GOOG']`` lookup, rather than collapsing into a generic
runtime failure.
