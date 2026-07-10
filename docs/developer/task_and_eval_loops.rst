Task And Eval Loops
===================

The task and evaluation layer is how Trellis measures whether the agent and
knowledge system are actually improving over time.

Task Corpus
-----------

The pricing-task surface is split across:

- ``TASKS_BENCHMARK_FINANCEPY.yaml`` for FinancePy parity tasks
- ``TASKS_EXTENSION.yaml`` for Trellis-only nearby variants
- ``TASKS_NEGATIVE.yaml`` for clarification / honest-block tasks
- ``TASKS_PROOF_LEGACY.yaml`` for retained proof-only legacy tasks

``FRAMEWORK_TASKS.yaml`` still holds non-priceable framework/meta tasks.
``trellis.agent.task_runtime`` turns pricing-task entries into offline-ready
execution contexts, market states, method plans, and benchmarkable generated
modules.

The runtime helpers cover:

- loading task ranges and statuses
- building default or task-specific mock market states
- reusing generated modules when available
- benchmarking cached task payoffs
- normalizing task descriptions into request/build inputs

Agent Composition Guardrails
----------------------------

Task generation is intentionally assembly-first. Route and backend-binding
catalogs should expose reusable primitives and binding surfaces; generated
adapters still own product-specific payoff assembly unless a checked helper is
explicitly selected as the exact backend binding.

Current composition rules:

- identity inference treats product names such as ``autocallable``,
  ``autocall``, ``phoenix``, and ``snowball`` as stronger than barrier-trait
  words. A terminal protection or autocall barrier is a trait of an
  autocallable, not evidence that the product should be narrowed to
  ``barrier_option``.
- ordinary barrier-option requests add the ``single_barrier`` payoff trait
  unless they explicitly name lower-and-upper or double-barrier structure. The
  PDE lane prefers ``price_single_barrier_option_pde_result`` and the Monte
  Carlo lane prefers ``price_single_barrier_option_monte_carlo_result`` from
  ``trellis.models.single_barrier_option`` for zero-rebate single-barrier
  comparison targets.
- double-barrier requests add the ``double_barrier`` payoff trait. The PDE lane
  prefers ``trellis.models.double_barrier_option``'s
  ``price_double_barrier_option_pde_result`` and records the lower-level
  ``resolve_double_barrier_inputs``, ``Grid``, ``BlackScholesOperator``,
  ``theta_method_1d``, and ``terminal_double_barrier_payoff`` obligations. The
  Monte Carlo lane prefers ``price_double_barrier_option_monte_carlo_result``
  and records the ``GBM``, ``MonteCarloEngine``, and
  ``double_barrier_state_payoff`` path-monitor obligations.
- stochastic-volatility PDE requests with ``stochastic_vol`` traits can select
  the ``heston_adi_2d`` route. That route is still a ``pde_solver`` family
  route; the route id only supplies the Heston-specific ADI evidence signature
  and model-parameter binding contract.
- resolver primitives with role ``market_binding`` may satisfy deterministic
  market-access review checks. Generated code can call the resolver instead of
  duplicating every ``market_state`` lookup inline, as long as it uses the
  selected primitive surface.
- single-underlier autocallable MC proof routes select
  ``trellis.models.autocallable.price_autocallable_monte_carlo_result`` as the
  checked event contract. The pseudo target calls it with
  ``sampling="pseudo"``; the QMC target calls it with ``sampling="sobol"``.
  Sobol is therefore a QMC comparison-target obligation, not a base
  autocallable MC primitive.
- deterministic API guardrails reject known near-misses before runtime, such
  as ``GBM(spot=...)`` and importing ``SobolNormals``. Use ``GBM(mu=...,
  sigma=...)`` and pass the initial spot to simulation/path construction; use
  the function ``sobol_normals`` from ``trellis.models.monte_carlo`` or
  ``trellis.models.qmc``.

The non-integration pytest surface is also grouped into explicit reviewable
strata:

- ``crossval`` for independent-library cross-checks
- ``verification`` for numerical or analytical reference tests
- ``task_challenge`` for proof-style task challenge regressions under
  ``tests/test_tasks/``
- ``global_workflow`` for user-facing workflow tests that span modules
- ``legacy_compat`` for deprecated or compatibility-only behavior
- ``freshness`` for release-gate freshness contracts such as cassette age
  validation

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

- ``scripts/run_tasks.py``: run a contiguous task block or all pending pricing tasks, optionally filtered by corpus
- ``scripts/render_task_batch_report.py``: turn one ``task_results_*.json`` tranche into a portable Markdown/JSON scorecard for GitHub summaries or artifact upload
- ``scripts/rerun_ids.py``: re-run specific task ids
- ``scripts/benchmark_tasks.py``: benchmark cached generated payoffs without rebuilding
- ``scripts/run_task_learning_benchmark.py``: run repeated non-canary passes at a fixed revision and emit a learning scorecard
- ``scripts/run_financepy_benchmark.py``: run the FinancePy parity corpus with timestamped run-history persistence
- ``scripts/run_negative_benchmark.py``: run the clarification / honest-block corpus with timestamped run-history persistence
- ``scripts/run_benchmark_history_scorecard.py``: build a repeated-run scorecard from append-only FinancePy or negative benchmark history
- ``scripts/remediate.py``: analyze actionable failures from the latest canonical task-run surface by default, inspect root-level ``task_results_*.json`` tranches with ``--source tranches``, or bound the scan to explicit result files with ``--results`` / ``--task-id`` / ``--skip-platform-traces``
- ``scripts/evaluate_shared_memory.py``: compare two task-result tranches and render a shared-memory improvement report
- ``scripts/should_run_canary.py``: decide whether current local changes justify the focused core canary gate
- ``scripts/test_hygiene.py``: report stale skip/xfail/quarantine markers for local test-hygiene triage

``run_task(...)`` defaults to ``recovery_mode="strict"`` for production-like
callers.  Task-operation scripts default to ``assisted`` and expose
``--recovery-mode strict|assisted|remediation``.  In assisted/remediation mode,
a failed target may receive one bounded intra-run retry with an ephemeral
``KnowledgePatchCandidate`` overlay derived from failure/reflection evidence
and deterministic contract evidence.  The candidate can carry structured
callable-signature records, required primitive obligations, and comparison
contract metadata; the prompt overlay is only the rendered form of that bounded
evidence.  The runtime scores ``contract_completeness`` before retrying.  A
candidate with only prose guidance is persisted as a skipped recovery attempt
with explicit ``skip_reasons`` and does not call the builder again.  The overlay
is not canonical cookbook knowledge and cannot promote itself.  It is persisted
as ``recovery_attempts`` / ``intra_run_learning`` evidence so diagnostics can
distinguish recovered candidate-knowledge retries, skipped candidates, and
ordinary build retries.

Assisted retries also persist a retry-attribution record.  The runtime records
whether the candidate was retryable, whether structured contract evidence was
present, whether deterministic retry inputs changed, which fields changed, and
whether the retry changed the observed result.  A retry is counted as
``contract_evidence_consumed`` only when the candidate's structured obligations
are carried into changed build inputs; this prevents task scorecards from
treating a retry attempt itself as evidence that learning helped.

The overlay is not only prompt text.  ``build_with_knowledge(...)`` forwards
retry overlays into ``build_payoff(...)`` and ``compile_build_request(...)``.
The compiler consumes available required-primitive and callable-signature
obligations into ``GenerationPlan`` module, symbol, reusable-primitive, and
helper-ref fields before validation contracts and route-binding authority are
computed.  The compiled request records
``intra_run_learning_overlay_consumption`` with candidate ids, target ids,
obligation kinds, applied inputs, and unapplied obligations.

Sparse proof rows may also bind directly to deterministic task-runtime
targets when the reusable library primitive already exists and code generation
would only create adapter noise.  The counterparty CVA proof rows use this path
for ``T52`` and ``T54``: ``T52`` compares the bounded IRS exposure-CVA helper
against its flat-hazard expected-exposure approximation, while ``T54`` declares
``independent_cva`` as the reference target and checks ``correlated_cva`` with
a directional ``>=`` relation.  Directional comparisons should be represented
with ``cross_validate.reference_target`` plus per-target ``relations`` rather
than forced through the median/equality harness.

The repo root ``Makefile`` now exposes the explicit gate entrypoints:

- ``make gate-pr`` for PR-ready validation
- ``make gate-canary`` for the focused live canary subset
- ``make gate-release`` for the broader replay/drift/freshness release gate

``make gate-pr`` now skips the slow proof/reference layers
(``tests/test_crossval/``, ``tests/test_verification/``, ``tests/test_tasks/``,
and cassette freshness) and keeps those in ``make gate-release`` instead. The
intent is to keep ordinary merge validation centered on core correctness while
still preserving the broader numerical/reference evidence before releases.

GitHub Actions now runs that same PR surface as deterministic shards generated
by ``scripts/pr_gate_shard.py`` plus a separate tier-2 contract job, so PR wall
clock is no longer pinned to one serial pytest command.

For ad hoc monitoring, ``.github/workflows/task-batch-report.yml`` exposes a
manual ``workflow_dispatch`` runner for the pricing-task surface. It supports
three selection modes:

- ``ids`` for explicit task ids such as ``F001 P004 P006``
- ``range`` for one contiguous id block such as ``F001`` through ``F015``
- ``all`` for the full manifest surface after any status/corpus filter

The workflow also accepts optional corpus filters, ``pending`` versus ``all``
status selection, validation profile, reuse versus fresh-build controls, and
the knowledge-light profile. The GitHub Actions default model is
``openai/gpt-4.1`` because GitHub Models expects catalog model IDs such as
``openai/gpt-4.1`` rather than direct OpenAI API model names such as
``gpt-5.4-mini``. Each manual run emits:

- raw batch results JSON
- the deterministic summary JSON from ``summarize_task_results(...)``
- a portable Markdown plus JSON task-batch report
- a compact Markdown summary appended to the GitHub Actions run summary

The workflow is also explicitly gated on ``github.triggering_actor`` so only
the allowed GitHub login can dispatch or re-run it. That is a spend-control
guard for token-backed task batches; keep the allowed actor in the workflow
file aligned with the repository owner. The job itself now runs inside the
``paid-task-batch`` environment with ``deployment: false`` so the batch still
uses environment protection rules without creating a deployment record.

The hosted workflow is wired to GitHub Models, not the direct OpenAI API. It
uses the Actions ``GITHUB_TOKEN`` as ``OPENAI_API_KEY``, sets
``OPENAI_BASE_URL=https://models.github.ai/inference``, and grants only
``models: read`` plus ``contents: read`` permissions. Keep yourself as the
required reviewer for the ``paid-task-batch`` environment so the job is still
withheld until you approve it. Local runs can continue to use the direct
OpenAI API by leaving ``OPENAI_BASE_URL`` unset and providing a normal
``OPENAI_API_KEY``.

The uploaded report artifact is the right first monitoring surface because it
keeps the repo free of ad hoc benchmark noise while still leaving one durable
Markdown table per triggered run. If you later want Trellis GitHub Pages to
show the latest scorecard, publish that generated Markdown as a second step
instead of checking every manual run into ``docs/benchmarks/``.

Use ``scripts/should_run_canary.py`` before paying for the live canary subset.
The helper reads local changed files from git status by default and recommends
the focused ``core`` canary subset when runtime, pricing, task-manifest, or
canary-runner surfaces move.

The release gate keeps its canary evidence deterministic by replaying the
committed full-task cassettes and then calling ``--check-drift``. That drift
step still depends on a latest available decision checkpoint for the replayed
task, so the runner will explicitly report when no checkpoint exists yet and
the drift probe was skipped.

Decision checkpoints are now binding-first at the generation boundary. Fresh
artifacts emit a ``binding`` stage plus binding validation identity, while
legacy route-era checkpoint YAML is normalized on load so drift comparison can
still compare old and new runs without treating the stage rename itself as a
meaningful divergence.

For curated canaries specifically, ``scripts/run_canary.py`` now executes the
live task entry plus any richer canary-only fields from ``CANARY_TASKS.yaml``.
That lets sparse manifest rows keep their normal task-inventory shape while
the curated regression surface still carries the fuller descriptions or market
blocks needed to exercise canonical comparison cases honestly.

The same runner now also has full-task replay modes for diagnosis-heavy
canaries.  Cassette-backed replay is for canaries that still require recorded
LLM calls:

- record with
  ``PYTHONHASHSEED=0 python3 scripts/record_cassettes.py --task <task-id>``
- replay with
  ``PYTHONHASHSEED=0 python3 scripts/run_canary.py --task <task-id> --replay``
- full-task canary cassettes live under ``cassettes/full_task/``

The hash seed matters because the replay contract hashes the full prompt text.
Use ``PYTHONHASHSEED=0`` for both recording and replay so prompt surfaces that
still depend on iteration order stay stable across processes.
Run these commands with the repo-standard miniforge interpreter; if your shell
``python3`` resolves elsewhere, invoke the configured interpreter path from
``AGENTS.md`` directly.

Canaries whose manifest entry sets
``replay_mode: deterministic_exact_binding`` use a different zero-token replay
lane.  The runner executes the real ``run_task(...)`` surface under the
offline-local LLM guard and requires the task to complete through deterministic
exact bindings.  No cassette calls are consumed, so prompt-hash drift in old
recordings cannot block the replay.  If such a canary attempts a live LLM call,
the offline guard fails the run.

Unlike the older tier-2 pipeline cassettes, the full-task canary path replays
the real ``run_task(...)`` surface. That means the replay still exercises the
runtime contract, comparison harness, task-run persistence, and diagnosis
packet/dossier writes, but it does not spend live model tokens.

Full-task canary replay remains strict about prompt hashes, call ordering, and
unexpected additional LLM calls. The canary runner does tolerate skipped or
unconsumed ``critic`` and ``unscoped`` calls, because those are optional
post-build review or learning stages that can be skipped between deterministic
method targets or after the runtime path has completed. Skipped or unconsumed
planning, binding, or generation calls still make the replay stale and fail the
run.

Full-task cassette sessions also keep the knowledge store read-only during
record and replay. The build still performs its normal LLM reflection calls,
but it does not write new lessons, traces, cookbook candidates, or promotion
candidates back into the canonical knowledge store. This keeps the replay
prompt surface aligned with the state that the cassette was recorded against
instead of letting the recording run mutate later retrieval inputs.

Every FinancePy benchmark run now also writes append-only timestamped records
under ``task_runs/financepy_benchmarks/`` with explicit ``run_started_at`` and
``run_completed_at`` fields so repeated reruns can be compared across code,
knowledge, and campaign revisions. The runner also emits a repeated-run
scorecard from the matching append-only history slice.

Those benchmark runs now keep any per-task run/diagnosis packets under an
isolated local store at ``task_runs/financepy_benchmarks/task_run_records/``
instead of rewriting the repo-wide canonical ``task_runs/latest`` surface.

Those repeated-run scorecards now include an ``agent_cycle`` block derived from
the same product-facing cycle surface returned by task results. The aggregate
tracks cycle-report availability, pass/fail/incomplete counts, stage trigger
rates, model-validator execution/skips/failures, blocker-bucket totals, and
residual limitation/risk counts. Use that block to monitor the
quant/critic/arbiter/model-validator loop; raw trace prose is not the stable
monitoring API.

Negative-task benchmark runs do the same under
``task_runs/negative_benchmarks/`` so clarification and honest-block behavior
can be tracked across repeated library and knowledge updates, again with a
scorecard generated from append-only history. Their per-task run packets now
live under ``task_runs/negative_benchmarks/task_run_records/`` for the same
reason.

Every canary batch now also writes a dedicated aggregate telemetry record under
``task_runs/canary_batches/``:

- ``task_runs/canary_batches/history/<batch_id>.json`` keeps the immutable
  per-batch history
- ``task_runs/canary_batches/latest/<scope>.json`` keeps the latest batch for a
  stable comparison scope such as ``live__full_curated__standard__default``

Canary batch summaries now also persist the repo revision and normalized
knowledge revision so replay and live batches can be compared against the
library/knowledge state that produced them.

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
synthetic so they do not become accidental benchmark baselines, and synthetic
pytest batches no longer rewrite the stable ``task_runs/canary_batches/latest``
comparison files.

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
cohort from the active pricing-task manifests and executes repeated passes at the same git
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
- an isolated per-pass task-run ledger under
  ``task_runs/learning_benchmarks/task_run_records/``

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
- retry-learning attribution that separates genuine retry-learned recovery
  from first-pass deterministic reuse

Use it like this:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py --limit 10 --passes 2

If you want to inspect the selected cohort first:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py --list-tasks --limit 10

To prove the bounded intra-run retry path itself without live model calls, use
the seeded local fixture:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py \
     --seeded-retry-fixture --passes 1 --knowledge-light \
     --report-name seeded_retry_learning

That fixture bypasses manifest selection and uses a local fake builder. The
first attempt fails on a checked callable signature, then the assisted retry
must consume structured contract evidence and recover. Its scorecard should
show a retry-learned recovery, not only a first-pass deterministic route reuse.

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

Task-result rows also mirror the diagnosis packet's compact outcome fields:
``failure_bucket``, ``diagnosis_headline``, ``diagnosis_decision_stage``, and
``diagnosis_next_action``. The older ``task_diagnosis_*`` path/status fields
remain for compatibility, but remediation and batch summaries should prefer
the canonical aliases when they are present. ``scripts/remediate.py
--analyze-only`` uses those structured buckets before raw text heuristics, so
proof-task failures stay grouped as ``blocked``,
``comparator_build_failure``, or ``comparison_insufficient_results`` instead
of collapsing into a generic validation-failure bucket.

Sparse legacy proof rows can also carry a deterministic task-runtime contract
bridge before generation. The bridge is intentionally small and auditable:
``T25``, ``T26``, ``T31``, and ``T32`` default to a European SPX call
semantic contract for Monte Carlo numerical-method proof work. ``T14`` uses an
American SPX put contract for the PDE/tree/LSM comparison and resolves
``lsm_mc`` through the exact
``price_american_equity_option_lsm_monte_carlo(...)`` helper, so the task tests
route-helper binding rather than generated early-exercise branching. ``T15``
uses a bounded CEV European call contract for the CEVOperator PDE versus CEV
tree comparison and binds ``cev_pde`` / ``cev_tree`` to deterministic local
proof adapters. The PDE adapter must consume CEV parameters through
``CEVOperator``; it should not silently reuse a Black-vol vanilla PDE helper
just because the payoff is also European vanilla. Its validation bundle is
``*:cev_option`` and intentionally skips Black-vol-surface sensitivity checks,
because CEV proof routes use explicit CEV model parameters rather than a Black
vol surface as the live model driver. ``T27`` defaults to an American SPX put
contract for the LSM basis comparison and binds the comparison target ids
``polynomial``, ``laguerre``,
``hermite``, ``chebyshev``, and ``high_step_tree_2000`` to deterministic local
proof adapters that compose the public Longstaff-Schwartz basis primitives and
CRR tree reference. They are task-runner proof contracts, not promoted
cookbooks or public pricing APIs. ``T18`` does not get a synthetic rate payoff;
it is certified as an expected honest block because the legacy row names a
log-space PDE transform for rate instruments without specifying the rate
payoff, schedule, strike/coupon terms, or settlement rule. Those bridge
decisions are task-runner contracts, not general natural-language parser
behavior.

Stochastic-volatility task runs also carry a ``computational_problem`` block
when the task target is Heston, Bates, SLV/LSV, or a related path-dependent
control shape. The block is copied into the task result,
``runtime_contract``, comparison-target request metadata, and the diagnosis
packet. It is meant for task triage and remediation: it records the
computational bucket, model-parameter semantics, validation bundle, and any
missing primitive or unsupported class before raw build exceptions dominate
the failure story. Path-dependent Heston control targets also preserve a
``path_dependent_control_contract`` so expected honest blocks identify the
missing path-state, event-monitor, payoff-summary, control-policy, and
stochastic-vol coupling abstractions directly.
The developer-facing contract for those terms lives in
:doc:`stochastic_vol_computational_ir`.

Task results now distinguish fail-closed pricing success from expectation
success. ``outcome_class="honest_block"`` plus ``passed_expectation=true``
means the route did not produce a price, but it satisfied a negative or
honest-block task by stopping with a concrete blocker. Batch summaries and
``scripts/remediate.py --analyze-only`` exclude those rows from actionable
failure counts while preserving the underlying ``success=false`` pricing
semantics.

That packet/checkpoint surface also now treats backend binding identity as the
primary implementation provenance. Compatibility route aliases may still
appear, but only as secondary metadata so replay, canary drift, and learning
artifacts no longer rely on route-primary checkpoint joins.

For canary replays, the task result and persisted task-run record now carry
explicit execution metadata:

- ``execution_mode`` is ``cassette_replay`` instead of ``live``
- ``execution_mode`` is ``deterministic_replay`` for exact-binding replay
  canaries that do not consume cassette calls
- ``llm_cassette`` records the cassette name, path, replay policy, and
  ``used=false`` for deterministic replay

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
requests now draft into the canonical ``period_rate_option_strip`` semantic
contract. That semantic contract materializes the schedule-driven
rate-option-strip shape, required market inputs, and route surface before
generic semantic-gap handling runs. That keeps compare-ready tasks such as
``E22`` on the actual pricing route without teaching the generic classifier
any instrument-specific special cases.

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

The bounded P001 Bermudan rainbow proof uses the same compatibility-shell
discipline, but its target is execution IR rather than an exact helper module.
``trellis.instruments._agent.rainbowoption`` delegates to
``trellis.execution.price_bermudan_best_of_basket_from_compat_spec(...)`` for
both Monte Carlo and lattice task targets. Task validation treats explicit
vector vol fields on that generated spec as the volatility authority when it
runs vega checks, so the proof validates the actual shim inputs instead of a
market-state flat-vol field the shim does not consume.

Fresh-build proving keeps a stricter boundary than ordinary supported-route
reuse. When a task or canary is run with ``fresh_build=True``, the executor
now skips deterministic exact-binding materialization so the run still has to
exercise live code generation. Ordinary supported-route runs can still reuse
exact helper wrappers for stability, but fresh-build canaries no longer get a
free pass from executor-side deterministic module synthesis.

The same rule now applies to tranche-style basket-credit and typed
loss-distribution comparison routes.
Curated copula canaries bind through the semantic-facing
``trellis.models.credit_basket_copula`` helper surface instead of asking the
builder to reconstruct tranche-loss projection from raw copula primitives. In
practice that means the build loop can keep tranche attachment/detachment or
portfolio-loss horizon, representative credit-curve binding, and
dependence-family selection on a checked helper path while semantic validation
treats the helper as the public assembly contract rather than forcing direct
calls to the lower-level loss-distribution primitives.

For the supported swaption tree slice, the comparison harness now has a
checked-in helper-backed route for single-exercise European rate-tree
comparators. That keeps canaries such as ``T65`` on the stable
``price_swaption_tree(...)`` surface instead of regenerating inline lattice
exercise code for the comparison target.

Transform proving now also distinguishes model families explicitly. The thin
vanilla transform helper surface is only used for ``equity_diffusion`` claims;
stochastic-volatility tasks such as the Heston smile canary stay on a checked
Heston transform helper under the same route family. That helper resolves
underlier spot plus ``market_state.model_parameters`` into the existing
FFT/COS kernels and keeps Black volatility surfaces out of live Heston pricing
unless a calibration bridge explicitly owns the conversion. Unsupported
transform methods such as Heston Gauss-Laguerre produce a repair packet instead
of falling back to a vanilla Black-vol adapter. T114-style targets also carry a
``quadrature_transform_contract`` so the repair packet names the missing
Heston characteristic-function quadrature kernel, integration requirements,
diagnostics, and validation bundle explicitly.

Jump-diffusion transform targets follow that same separation. Merton comparison
targets such as ``merton_fft`` and ``merton_cos`` now bind to
``trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_transform``,
and ``merton_mc`` binds to the sibling terminal Monte Carlo helper. The
``ProductIR`` keeps the product shape as a European vanilla option, but adds
``model_family=jump_diffusion`` and ``jump_parameters`` as required runtime
evidence. Validation fixtures must include those jump parameters instead of
trying to reinterpret the task as an ordinary Black-vol vanilla route.

Variance Gamma, CGMY, and Kou targets use the same ProductIR discipline.
Sparse task labels such as ``vg_cos``, ``vg_mc``,
``madan_carr_chang_reference``, ``cgmy_cos``, ``cgmy_mc``,
``cgmy_reference_values``, ``kou_fft``, ``kou_mc``, and
``kou_reference_values`` remain European vanilla options, but their
``model_family`` narrows to ``variance_gamma``, ``cgmy``, or ``kou`` and
their market evidence narrows to explicit ``model_parameters``. The
deterministic exact-binding wrappers should call the checked
``trellis.models.levy_option`` helpers. Validation should not run generic
Black-vol vega checks for these routes unless a separate calibration problem
explicitly owns a Black-vol-to-Levy-parameter bridge.

Bates comparison targets keep the same product-shape discipline. A task such
as ``bates_fft`` versus ``bates_mc`` remains a European vanilla option in
``ProductIR`` and narrows by ``model_family=bates`` plus both
``model_parameters`` and ``jump_parameters``. The FFT target binds through
``trellis.models.bates_option.price_bates_option_transform`` and the MC target
binds through ``trellis.models.bates_option.price_bates_option_monte_carlo``.
Generated wrappers should call those helpers rather than composing Heston
paths and jump aggregation locally.

SABR comparison targets keep the same product-shape discipline. A task such as
``sabr_mc`` versus ``sabr_hagan_analytical`` remains a European vanilla
forward-style option in ``ProductIR`` and narrows by ``model_family=sabr`` plus
``model_parameters``. The Hagan target binds through
``trellis.models.sabr_option.price_sabr_forward_option_hagan`` and the MC
target binds through
``trellis.models.sabr_option.price_sabr_forward_option_monte_carlo``. The
generated wrapper must not inherit a ``black_vol_surface`` requirement from the
generic Black76 skeleton; SABR parameters are the runtime evidence unless a
separate calibration problem explicitly owns the Black-vol-to-SABR conversion.

The Monte Carlo lane now follows the same model-family separation for European
Heston vanilla options. ``euler_heston`` and ``heston_mc`` targets bind to
``trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo``
with ``scheme="euler"``, while ``qe_heston`` binds to the same helper with
``scheme="heston_qe"``. The helper consumes explicit Heston model parameters
and reports the ``heston:monte_carlo`` validation bundle, so the task runtime no
longer treats Andersen QE as a missing generated-adapter primitive. It still
does not recalibrate Heston parameters from a bumped Black vol surface unless a
separate calibration problem owns that conversion.

The Heston ADI PDE target uses the same model-parameter boundary. The
``heston_adi_pde`` comparison target should bind through
``trellis.models.pde.heston_adi.resolve_heston_adi_pde_inputs`` and price with
``price_heston_option_adi_pde_result``. Agents should not call
``resolve_heston_transform_inputs`` with payoff arguments such as ``strike``;
FFT/COS transform calls are optional diagnostics, not the ADI market-binding
contract.

Double-barrier PDE and Monte Carlo targets use the same helper-owned contract
style. When the compiled primitive plan selects
``price_double_barrier_option_pde_result`` or
``price_double_barrier_option_monte_carlo_result`` as the required route
helper, a generated adapter should stay thin and call that helper with
``market_state`` plus the original spec-like object. Semantic validation treats
the checked helper as owning its internal grid, operator, barrier monitor,
terminal payoff, and discounting obligations, but it still rejects adapters
that skip the helper or invent alternate raw inputs such as ``spot`` or barrier
keywords.

Single-barrier PDE and Monte Carlo targets follow the same helper-owned
contract for the ordinary zero-rebate barrier cohort. When the compiled
primitive plan selects ``price_single_barrier_option_pde_result`` or
``price_single_barrier_option_monte_carlo_result``, a generated adapter should
delegate directly to that helper with ``market_state`` plus the original spec.
The helper owns the absorbing barrier boundary, far vanilla boundary,
single ``BarrierMonitor``, notional convention, and deterministic discounting.
Rebate-bearing barriers should remain on the analytical Rubinstein route until
the PDE/MC rebate contract is implemented.

Digital, Asian, and fixed-lookback proof targets use exact helper wrappers only
for the retained comparison surfaces that have checked numerical contracts.
Cash-or-asset digital Crank-Nicolson and Rannacher targets bind to
``price_equity_digital_option_pde(...)``; arithmetic-Asian MC targets bind to
``price_arithmetic_asian_option_monte_carlo(...)``; Turnbull-Wakeman targets
are analytical comparison targets and bind to
``price_arithmetic_asian_option_analytical(...)`` even when the broader task is
a multi-method path-dependent option; and fixed-lookback MC targets bind to
``price_equity_fixed_lookback_option_monte_carlo(...)``. Sparse legacy task
text may use cross-validation target names to recover product identity, but the
runtime still fails closed if the resolved contract and exact helper disagree.

Capped/floored cliquet comparisons follow a bounded version of that contract.
The analytical target can use the checked capped/floored reset-return
quadrature path, and the Monte Carlo target should call
``price_equity_cliquet_option_monte_carlo(market_state, spec, ...)`` instead of
rebuilding reset-date GBM path generation inline. The generic volatility
monotonicity invariant is not enforced for ``cliquet_option`` because
local/global caps and floors can make the capped return value non-monotone in
Black volatility; volatility sensitivity remains part of the validation pack.

The ADI helper also owns its variance-grid domain. The grid upper bound is
based on the CIR variance-process dispersion, not a raw ``xi * sqrt(T)`` move;
otherwise high vol-of-vol fixtures place the initial variance too close to the
lower boundary and bias the finite-difference price high. T20-style ADI
comparisons should therefore use the checked helper instead of rebuilding the
variance grid in generated code.

Offline local-agent reruns can use ``--offline-local-agents`` on
``scripts/run_tasks.py`` or ``scripts/rerun_ids.py``. That mode sets the
post-build learning skips, disables LLM-backed critic/model-validator review
through deterministic review policy, and leaves the LLM override guard active
as a hard backstop. Batch summaries report expectation semantics separately
from pricing success: an expected honest block remains fail-closed with no
price, but prints as ``HONEST_BLOCK`` and counts toward
``passed_expectation`` rather than actionable failure.

The reference no-LLM closeout pack for the contract-backed task-learning work
is:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
     --task-id T20 --task-id T22 --task-id T105 --task-id T107 --task-id E27 \
     --status all --offline-local-agents --recovery-mode assisted \
     --validation standard \
     --output task_results_qua1143_offline_closeout_20260702.json

That run reported ``5/5`` passed expectations in ``95s``: ``T20``, ``T22``,
``T105``, and ``T107`` were ``compare_ready`` pricing successes; ``E27`` was an
``honest_block``; token usage was zero; and bounded remediation reported zero
failures.  The expected healthy shape is now first-pass deterministic reuse,
not repeated retry recovery.

That route family now also has its own lowered contract boundary. Transform
tasks compile onto ``TransformPricingIR`` before admissibility, so the canaries
no longer have to rely on the broader upstream ``vanilla_option`` semantics
when the numerical lane only needs a terminal-only characteristic-function
contract. In practice this is what keeps transform canaries such as ``T39`` and
``T40`` from failing on irrelevant option-family tags like ``holder_max`` or
``recombining_safe``.

For the analytical benchmark side of those canaries, the runtime now also
materializes a deterministic exact wrapper around the checked Black76 kernels.
The same rule applies to sparse transform and credit comparison targets:
``fft``/``cos`` GBM lanes delegate to
``price_vanilla_equity_option_transform(...)``, digital ``fft``/``cos`` lanes
delegate to ``price_equity_digital_option_transform(...)``, and
``credit_default_swap`` analytical lanes reuse the static ``CDSSpec`` and
``price_cds_analytical(...)`` binding.  These are task-runtime exact bindings,
not cookbook-authored generated adapters; under ``--offline-local-agents`` they
should complete without spec-design, code-generation, critic, or
model-validator LLM calls.
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
and an explicit family such as ``zcb_option`` no longer gets silently redrafted
into a generic ``vanilla_option`` semantic contract just because the request
text contains phrases like ``European call option``. That keeps exact helper
binding on the intended family-first path for the short-rate comparison cohort,
so ingress fallback only fires for phrases that still correspond to defended
task families rather than widening broad desk summaries into a pricing schema.

Option-family identity follows the same rule without requiring every combination
to become a new product family. Semantic contracts and ``ProductIR`` now carry
``derivative_family="option"``, ``payoff_family`` such as ``vanilla_option``,
the underlier asset class and identifiers, ``exercise_style``, and
``option_type`` when a call/put side is defined. Route and validation selection
must consume those axes before accepting an adapter, so an American equity put
can remain a vanilla-option payoff while still selecting early-exercise
artifacts, and an FX vanilla adapter cannot satisfy a quanto or equity option
only because the terminal payoff is numerically vanilla-shaped.

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
   /Users/steveyang/miniforge3/bin/python3 scripts/remediate.py --analyze-only --results task_results_rerun_failed_pack_20260623.json --skip-platform-traces
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "crossval and not integration"
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "verification and not integration"
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "task_challenge and not integration"
   /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "global_workflow and not integration"

Related Reading
---------------

- :doc:`audit_and_observability`
- :doc:`../quant/knowledge_maintenance`
