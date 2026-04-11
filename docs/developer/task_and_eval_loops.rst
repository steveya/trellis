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

The stale-test hygiene layer now sits alongside those strata rather than
replacing them. Use ``scripts/test_hygiene.py`` to inventory `skip`, `xfail`,
`importorskip`, and ``legacy_compat`` usage with approximate git-age buckets:

- ``quarantine`` for fresh markers
- ``stale`` for markers that have lingered for at least 30 days
- ``ancient`` for markers that are 90 days or older

Pytest collection also now rejects ancient ``xfail`` markers that do not carry
a linked ticket id such as ``QUA-123`` or ``CR-10``. That guard is
intentionally narrow: it keeps unticketed xfails visible without turning every
skip into a hard failure.

Operational Scripts
-------------------

The repo ships a small task-operations toolchain:

- ``scripts/run_tasks.py``: run a contiguous task block or all pending tasks
- ``scripts/rerun_ids.py``: re-run specific task ids
- ``scripts/benchmark_tasks.py``: benchmark cached generated payoffs without rebuilding
- ``scripts/run_task_learning_benchmark.py``: run repeated non-canary passes at a fixed revision and emit a learning scorecard
- ``scripts/remediate.py``: analyze failures, categorize knowledge gaps, and patch common knowledge issues
- ``scripts/evaluate_shared_memory.py``: compare two task-result tranches and render a shared-memory improvement report
- ``scripts/should_run_canary.py``: decide whether current local changes justify the focused core canary gate
- ``scripts/test_hygiene.py``: report stale skip/xfail/quarantine markers for local test-hygiene triage

The repo root ``Makefile`` now exposes the explicit gate entrypoints:

- ``make gate-pr`` for PR-ready validation
- ``make gate-canary`` for the focused live canary subset
- ``make gate-release`` for the broader replay/drift/freshness release gate

Use ``scripts/should_run_canary.py`` before paying for the live canary subset.
The helper reads local changed files from git status by default and recommends
the focused ``core`` canary subset when runtime, pricing, task-manifest, or
canary-runner surfaces move.

The release gate keeps its canary evidence deterministic by replaying the
committed full-task cassettes and then calling ``--check-drift``. That drift
step still depends on a latest available decision checkpoint for the replayed
task, so the runner will explicitly report when no checkpoint exists yet and
the drift probe was skipped.

For curated canaries specifically, ``scripts/run_canary.py`` now executes the
live task entry plus any richer canary-only fields from ``CANARY_TASKS.yaml``.
That lets sparse manifest rows keep their normal task-inventory shape while
the curated regression surface still carries the fuller descriptions or market
blocks needed to exercise canonical comparison cases honestly.

The same runner now also has a full-task replay mode for diagnosis-heavy
canaries:

- record with
  ``PYTHONHASHSEED=0 python3 scripts/record_cassettes.py --task T13``
- replay with
  ``PYTHONHASHSEED=0 python3 scripts/run_canary.py --task T13 --replay``
- full-task canary cassettes live under ``cassettes/full_task/``

The hash seed matters because the replay contract hashes the full prompt text.
Use ``PYTHONHASHSEED=0`` for both recording and replay so prompt surfaces that
still depend on iteration order stay stable across processes.
Run these commands with the repo-standard miniforge interpreter; if your shell
``python3`` resolves elsewhere, invoke the configured interpreter path from
``AGENTS.md`` directly.

Unlike the older tier-2 pipeline cassettes, the full-task canary path replays
the real ``run_task(...)`` surface. That means the replay still exercises the
runtime contract, comparison harness, task-run persistence, and diagnosis
packet/dossier writes, but it does not spend live model tokens.

Full-task cassette sessions also keep the knowledge store read-only during
record and replay. The build still performs its normal LLM reflection calls,
but it does not write new lessons, traces, cookbook candidates, or promotion
candidates back into the canonical knowledge store. This keeps the replay
prompt surface aligned with the state that the cassette was recorded against
instead of letting the recording run mutate later retrieval inputs.

Every canary batch now also writes a dedicated aggregate telemetry record under
``task_runs/canary_batches/``:

- ``task_runs/canary_batches/history/<batch_id>.json`` keeps the immutable
  per-batch history
- ``task_runs/canary_batches/latest/<scope>.json`` keeps the latest batch for a
  stable comparison scope such as ``live__full_curated__standard__default``

Those records sit on top of the existing per-task run history. They do not
replace ``task_runs/history/``; instead they provide the trustworthy canary
view that raw task runs could not guarantee on their own.

Each batch record carries:

- explicit lineage for ``live`` versus ``cassette_replay`` execution
- whether the batch is benchmark-eligible
- the selection scope (full curated set, subset, or single task)
- aggregate pass/fail, elapsed-time, token, and attempt metrics
- per-canary entries with links back to the underlying task-run and diagnosis
  artifacts

For maintenance tooling, the deterministic loaders live in
``trellis.agent.task_run_store``:

- ``load_canary_batch_records()``
- ``load_canary_task_history()``

Use those loaders, not ad hoc scans over ``task_runs/history/``, when you want
to compare canary latency, token, or attempt trends over time. The batch store
excludes ordinary ad hoc task runs by construction and can filter replay-backed
history out of the benchmark view. Root-level pytest runs are also marked as
synthetic so they do not become accidental benchmark baselines.

To compare against the ``2026-04-09`` full curated rerun baseline, treat that
date's documented ``14/14`` pass as the historical anchor and compare fresh
batch records that match the same live scope:

- ``execution_mode=live``
- ``batch_scope=full_curated``
- ``validation=standard``
- ``knowledge_profile=default``

Fresh partial-task reruns are still useful for diagnosis, but they should not
be described as replacements for the full curated baseline.

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

For the focused simple-derivative tranche, use
``scripts/run_knowledge_light_proving.py``. That harness keeps the
compiler-first, knowledge-light prompt surface but now emits a more explicit
reliability summary alongside the ordinary correctness outcomes:

- ``first_pass`` for task-level first-attempt success
- ``attempts_to_success`` for successful-task attempt counts
- ``retry_taxonomy`` for recovered successes bucketed by the triggering stage

The retry taxonomy is derived from the recorded platform trace events rather
than free-form closeout text, so a proving rerun can answer "what recovered?"
and "which stage missed first?" from the stored artifact alone.

For comparison tasks, ``attempts_to_success`` is normalized by the slowest
successful method leg, not by summing all nested attempts. Two methods that
both succeed on their first method-local attempt still count as a first-pass
task-level success.

The proving harness depends on the same ``ProductIR`` retrieval bridge as the
ordinary build loop. That bridge now carries market-data-derived retrieval
features plus a small set of IR-derived semantic text markers into lesson
selection. The intent is narrow: once a lesson is already broadly relevant by
method and feature overlap, the rerank can surface the lane-specific lesson
that actually matches the current analytical or helper-bound contract instead
of stopping at a more generic lesson from the same method family.

For broader short-term learning evidence, use
``scripts/run_task_learning_benchmark.py``. That runner selects a non-canary
cohort from ``TASKS.yaml`` and executes repeated passes at the same git
revision. The benchmark is asking a narrower question than a canary or stress
gate:

- do later passes fail less often?
- do first-pass success and attempts-to-success improve?
- do elapsed time and token usage fall?
- do those improvements line up with reused knowledge signals?

The runner defaults to ``fresh_build=True``. That is deliberate. Reusing
admitted adapters would swamp the signal and turn the benchmark into a warm
cache measurement instead of a knowledge-carry-forward measurement. Reuse mode
still exists, but only when you explicitly want warm-path operational data.

Each pass writes:

- a raw task-result file under ``task_runs/learning_benchmarks/raw/``
- a pass summary JSON beside that raw file
- a Markdown plus JSON scorecard under ``task_runs/learning_benchmarks/reports/``

The scorecard reports:

- success and failure counts
- task-level ``first_pass``
- ``attempts_to_success``
- ``retry_taxonomy``
- elapsed-time totals
- token-usage totals
- shared-knowledge reuse signals
- attribution buckets separating knowledge-assisted improvements from residual
  knowledge gaps, implementation gaps, and market/provider noise

Use it like this:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py --limit 10 --passes 2

If you want to inspect the selected cohort first:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py --list-tasks --limit 10

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

For cassette-backed canary replays, the task result and persisted task-run
record now carry explicit execution metadata:

- ``execution_mode`` is ``cassette_replay`` instead of ``live``
- ``llm_cassette`` records the cassette name, path, and replay policy

That keeps replay runs obvious to humans and downstream tooling without
changing the packet or task-result schema that remediation and canary summary
tools already consume.

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

Comparison-target planning is now stricter about method-family labels as well.
Task-level construct hints such as ``credit`` or ``volatility`` are treated as
domain labels, not as canonical pricing methods, so they no longer bleed into
comparison-target routing. That keeps helper-backed canaries such as ``T51`` on
the analytical CDS route instead of degrading into an unbound generic lane.

The same lower-layer authority rule now applies to instrument families. Once a
task, request, or compiled plan already knows the concrete family
(``zcb_option``, ``basket_option``, ``barrier_option``, and so on), lower
runtime layers must not fall back to a generic text heuristic such as
``european_option`` just because the description happens to contain words like
``European`` or ``option``. Planner specialization, cached-module schema
inference, and executor fallback inference now all follow the same rule:
explicit family beats generic heuristic unless the specialization is an
explicitly allowed refinement of that family.

The rate cap/floor stress lane now follows that same authority rule through a
dedicated semantic family rather than a classifier exception. Cap and floor
requests now draft into the ``rate_cap_floor_strip`` semantic contract, which
materializes the schedule-driven rate-option-strip shape, required market
inputs, and route surface before generic semantic-gap handling runs. That
keeps compare-ready tasks such as ``E22`` on the actual pricing route without
teaching the generic classifier any instrument-specific special cases.

Comparison pricing now also prefers market-aligned smoke fixtures when the
route family needs them. For the supported quanto slice, the comparison
harness derives an at-the-money fixture strike from the resolved runtime
market state instead of relying on a generic hard-coded strike. That keeps the
analytical-versus-Monte-Carlo parity check focused on route differences rather
than on a distorted fixture contract.

For helper-backed comparison routes, the build loop now also prefers
deterministic exact wrappers over open-ended adapter synthesis. Those wrappers
thread semantic comparison-regime bindings such as explicit Hull-White
parameters through the analytical, tree, and Monte Carlo helper calls, and
they can attach stable comparison sampling controls for Monte Carlo lanes. This
keeps comparison failures concentrated on real model/regime mismatches instead
of letting generated adapter drift reintroduce stale imports or route-local
numerical defaults.

The lower-layer cleanup tranches now apply that same pattern to FX and quanto
routes as well. ``trellis.models.fx_vanilla`` and
``trellis.models.quanto_option`` are the semantic-facing helper kits for the
supported analytical and Monte Carlo slices, and the checked-in adapters under
``trellis.instruments._agent`` are intentionally thin shells over those
helpers instead of separate implementations.

Fresh-build proving keeps a stricter boundary than ordinary supported-route
reuse. When a task or canary is run with ``fresh_build=True``, the executor
now skips deterministic exact-binding materialization so the run still has to
exercise live code generation. Ordinary supported-route runs can still reuse
exact helper wrappers for stability, but fresh-build canaries no longer get a
free pass from executor-side deterministic module synthesis.

The same rule now applies to tranche-style basket-credit comparison routes.
Curated copula canaries bind through the semantic-facing
``trellis.models.credit_basket_copula`` helper surface instead of asking the
builder to reconstruct tranche-loss projection from raw copula primitives. In
practice that means the build loop can keep tranche attachment/detachment,
representative credit-curve binding, and dependence-family selection on a
checked helper path while semantic validation treats the helper as the public
assembly contract rather than forcing direct calls to the lower-level loss
distribution primitives.

For the supported swaption tree slice, the comparison harness now has a
checked-in helper-backed route for single-exercise European rate-tree
comparators. That keeps canaries such as ``T65`` on the stable
``price_swaption_tree(...)`` surface instead of regenerating inline lattice
exercise code for the comparison target.

Transform proving now also distinguishes model families explicitly. The thin
vanilla transform helper surface is only used for ``equity_diffusion`` claims;
stochastic-volatility tasks such as the Heston smile canary stay on the raw
FFT/COS kernel path under the same route family. That keeps helper reuse
honest without widening a GBM-oriented helper into unsupported Heston
authority.

That route family now also has its own lowered contract boundary. Transform
tasks compile onto ``TransformPricingIR`` before admissibility, so the canaries
no longer have to rely on the broader upstream ``vanilla_option`` semantics
when the numerical lane only needs a terminal-only characteristic-function
contract. In practice this is what keeps transform canaries such as ``T39`` and
``T40`` from failing on irrelevant option-family tags like ``holder_max`` or
``recombining_safe``.

For the analytical benchmark side of those canaries, the runtime now also
materializes a deterministic exact wrapper around the checked Black76 kernels.
That wrapper binds ``year_fraction(...)``, discounting, and expiry-vol lookup
through the actual runtime market-state protocols instead of asking the build
loop to regenerate a tiny analytical adapter for every transform comparison.

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

That summary now also carries the registered semantic family key plus the
resolved method-surface summary when the contract family is registry-backed.
In practice this makes the trace payload show both the semantic identity and
the exact family/method surface that the lower compiler layers are expected to
honor, which is useful when debugging admissible-method questions without
diffing request-layer branch logic.

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

Replay summaries follow the same rule. For analytical traces, selected curve
names are read from either ``context.selected_curve_names`` or the nested
``context.runtime_contract.snapshot_reference.selected_curve_names`` payload,
so trace consumers do not need to re-resolve snapshot inputs to recover the
executed curve-role binding.

If the market snapshot was built from bootstrapped rate instruments, the
resolver keeps that assembly step visible too: named bootstrap buckets become
named curves in the snapshot, and the chosen curve name is preserved in the
runtime state and replay metadata for validation.

The same rule now applies to the seeded mock market path. When a task or
proving run uses ``source="mock"``, the runtime contract and market-context
provenance preserve the synthetic prior seed plus the nested
``prior_parameters.synthetic_generation_contract`` payload. That means replay
and proving tooling can recover not only which named curves or surfaces were
selected, but also which bounded synthetic rates, credit, and volatility
authority pack produced those runtime objects.

For market-parameter sourcing specifically, downstream tooling should read the
stable ``market_parameter_trace`` summary carried in ``market_context``,
``runtime_contract``, and ``runtime_contract.snapshot_reference``. That compact
surface is the replay/report contract for selected parameter packs: it records
the chosen parameter-set name, the source kind, the selected source reference,
the parameter keys, and the source-family-specific details that matter for
review. Bootstrap entries expose their originating curve buckets, empirical
entries expose estimator/sample metadata, calibration entries expose workflow
plus quote-family / quote-subject / quote-unit metadata, and seeded mock runs
collapse to a synthetic-prior trace with the governing contract version and
seed.

Lower-layer family identity now follows the same provenance rule. Task runtime
records both ``instrument_type`` and ``instrument_identity_source`` on the
request metadata, runtime contract, simulation identity payload, task result,
and persisted run record. That gives later executor, cached-selection, and
validation slices a stable signal for whether family identity came from an
explicit field or from a single ingress fallback, instead of forcing them to
re-parse raw title text again.

Executor-side validation and helper-binding paths now prefer that threaded
family identity over raw-description fallback whenever the request metadata,
runtime contract, or ``ProductIR`` already supplies the family. Description
heuristics remain only as the final fallback when no structural family signal
exists yet. The residual text-pattern table is intentionally narrower now:
generic ``bond`` and ``swap`` wording no longer produce a family on their own,
so ingress fallback only fires for phrases that still correspond to defended
task families rather than widening broad desk summaries into a pricing schema.

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
