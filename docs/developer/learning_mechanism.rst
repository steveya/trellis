How Trellis Learns
==================

This document describes the current learning mechanism in operational terms.
It is the precise runtime path by which Trellis turns failures, successful
repairs, and build observations into reusable guidance.

Scope
-----

In Trellis, "learning" means:

- retrieving prior lessons, signatures, and policy before a build
- recording structured observations during and after a build
- distilling new candidate lessons from resolved failures
- validating and promoting those lessons into the canonical store
- using the promoted guidance on later builds and reruns

It does not mean autonomous code landing. The system can capture, rank,
promote, and reuse knowledge, but implementation and validation of new
foundational primitives still require an explicit code change.

Canonical Persistence Surfaces
------------------------------

The learning loop is canonical-only. The authoritative artifacts are:

- ``trellis/agent/knowledge/lessons/entries/*.yaml``: full lesson records
- ``trellis/agent/knowledge/lessons/index.yaml``: generated retrieval index
- ``trellis/agent/knowledge/canonical/failure_signatures.yaml``: fast failure-pattern index
- ``trellis/agent/knowledge/traces/``: audit and replay traces
- ``trellis/agent/knowledge/canonical/*.yaml``: policy assets such as
  decompositions, cookbooks, contracts, and requirements

There is no longer a parallel ``experience.yaml`` lesson file.

Runtime Learning Loop
---------------------

The active loop has five stages.

1. Retrieve guidance before generation

   ``KnowledgeStore.retrieve_for_task(...)`` builds the knowledge payload used
   by generation and review. Retrieval expands the feature set, scores lessons
   by feature overlap plus method/instrument/semantic bonuses, suppresses
   superseded lessons, and returns the top active matches together with the
   relevant cookbook, contracts, requirements, and matched failure signatures.

   When the compiler already has a ``ProductIR``, retrieval now also carries
   forward a small set of structural hints from that IR: market-data-derived
   retrieval features such as ``forward_rate`` plus semantic text markers built
   from the instrument, payoff family, model family, route-family labels, and
   reusable primitives. That keeps retrieval feature-first, but lets the final
   rerank prefer lessons that mention the exact analytical/helper contract
   already implied by the IR.

2. Diagnose failures during repair

   Retry feedback uses two inputs:

   - heuristic diagnosis from ``trellis.agent.test_resolution``
   - canonical regex-driven failure signatures from
     ``trellis.agent.knowledge.canonical.failure_signatures.yaml``

   The diagnosis helper can surface related canonical lessons by resolving the
   matched signature's ``probable_causes`` lesson ids.

3. Distill a lesson after a successful repair

   When a retry succeeds, ``trellis.agent.executor._record_resolved_failures``
   asks the model for one structured lesson with:

   - ``category``
   - ``title``
   - ``mistake``
   - ``why``
   - ``detect``
   - ``fix``

   The executor also decomposes the repaired product description to recover
   feature tags before persisting the lesson.

4. Validate and capture the lesson contract

   ``trellis.agent.test_resolution.record_lesson(...)`` now writes directly to
   the canonical lesson store. Internally it uses:

   - ``build_lesson_payload(...)`` to normalize the fields
   - ``validate_lesson_payload(...)`` to enforce the lesson contract
   - ``capture_lesson(...)`` to persist a candidate lesson if it is not a duplicate

   Deduplication happens at capture time. Lessons with the same title, or with
   very high title overlap, are suppressed instead of being duplicated.

5. Promote active lessons when confidence is high enough

   Promotion is explicit and confidence-gated:

   - ``capture_lesson(...)`` writes a ``candidate`` lesson
   - ``validate_lesson(...)`` moves it to ``validated`` when confidence is at least ``0.6`` and the fix is non-empty
   - ``promote_lesson(...)`` moves it to ``promoted`` when confidence is at least ``0.8``

   The executor's resolved-retry path currently records lessons at confidence
   ``0.5``. That means those lessons are captured as candidates and must be
   validated later before they influence normal retrieval.

Reflection-Sourced Learning
---------------------------

The reflection path can promote more aggressively because it has richer build
context. ``trellis.agent.knowledge.reflect._capture_structured_lesson(...)``
assigns confidence by discovery mode:

- ``0.4`` for unresolved failures
- ``0.6`` for first-attempt insights
- ``0.8`` for lessons discovered after a failed attempt that was then fixed

That path captures the lesson, runs the contract check, and can auto-validate
or auto-promote immediately when the confidence threshold is met.

Bounded Intra-Run Learning
--------------------------

Task runs can also use a bounded, ephemeral learning loop between a failed
target build and final task classification.  The contract lives in
``trellis.agent.intra_run_learning`` and uses
``KnowledgePatchCandidate`` records.  These records are not canonical lessons
or cookbooks.  They are one-shot prompt overlays derived from already-recorded
failure evidence, deterministic contract evidence, reflection gaps, candidate
lesson ids, and quant/model-validator observations when those observations
exist.

The mode boundary is explicit:

- ``strict``: never builds or applies an intra-run knowledge overlay
- ``assisted``: may retry one failed target once with a candidate overlay
- ``remediation``: may produce the same retry overlay while preserving richer
  repair evidence for branch/PR work

The overlay is appended by the existing stage-aware knowledge retriever in
``build_with_knowledge(...)``.  It is rendered with an explicit warning that
the guidance is ephemeral and not canonical.  It does not mutate
``canonical/cookbooks.yaml`` and it does not promote lessons.  If the retry
succeeds and later cross-validation passes, the normal promotion-candidate
pipeline may persist the successful generated route for deterministic review.

Candidate construction is deliberately conservative.  It skips provider/noise
failures, expected honest blocks, and repair-packet blockers.  For retryable
failures it records structured evidence where possible:

- callable signatures from the import registry and ``inspect.signature(...)``
  when a helper or constructor rejects an argument
- required primitive obligations, including import availability and signature
  when a semantic validator reports ``assembly.required_primitive_missing``
- comparison-contract evidence such as method prices, reference target,
  tolerance, selected route or binding, validation bundle, and payoff identity

Those records are persisted on the candidate as ``structured_evidence`` and
``repair_obligations``.  Candidate construction also assigns a
``contract_completeness`` score and ``retryable`` flag.  Assisted mode retries
only when the candidate has concrete structured obligations above the retry
threshold.  Prose-only candidates are retained as evidence with skip reasons
such as ``missing_structured_repair_obligation`` but do not call the builder
again.

The plain-text overlay remains only the rendered form of that evidence.  Each
task result records ``recovery_mode``, ``recovery_attempts``, and an
``intra_run_learning`` summary so downstream diagnostics can distinguish an
attempted candidate-knowledge retry from a skipped candidate.

Retry attribution is part of that contract.  Every recorded recovery attempt
now carries ``retry_attribution`` with the candidate id, patch type, structured
evidence count, repair-obligation count, changed build-input fields, and a
compact ``attribution_kind``.  ``contract_evidence_consumed`` is true only when
the candidate had structured contract evidence and the retry actually changed
deterministic build inputs such as ``knowledge_overlays`` or
``request_metadata.intra_run_learning_retry``.  Skipped candidates are still
recorded, but their attribution kind remains ``candidate_not_retryable`` or
``candidate_not_applied`` and ``deterministic_input_changed`` stays false.

Retry overlays also enter the deterministic request compiler.  When a retry
candidate carries available ``required_primitive`` or ``callable_signature``
obligations, ``compile_build_request(...)`` adds the corresponding module,
symbol, reusable primitive ref, and helper ref to the ``GenerationPlan`` before
validation and route-binding metadata are finalized.  The compiled request
records ``intra_run_learning_overlay_consumption`` so operators can see which
candidate ids were consumed and which generation-plan fields changed.  Binding,
comparison, and market-binding obligations are recorded as structured compiler
inputs; the compiler does not overwrite an existing checked backend binding
from an overlay.

What Future Builds Actually Reuse
---------------------------------

Future builds do not reuse every captured lesson.

Normal retrieval considers only active lessons:

- ``validated``
- ``promoted``

Candidate lessons are stored and indexed for audit and later review, but they
do not become normal build guidance until they pass the promotion gate.

The prompt surface therefore reuses:

- canonical principles
- active lessons
- method cookbook
- data contracts
- modeling requirements
- matched failure signatures

Deterministic Lesson-To-Test Projection
---------------------------------------

Validated and promoted lessons now also feed a separate deterministic
materialization seam in ``trellis.agent.knowledge.lesson_to_test``.

That layer does not mutate the canonical lesson store and does not ask the
model to write tests. Instead it:

- classifies one active lesson into a regression-template family
- attaches a stable target-test-file hint plus assertion and fixture focus
- renders a reviewable pytest-style fragment for the lesson
- ignores ``candidate`` and ``archived`` lessons so low-confidence guidance
  does not silently become regression authority

The base template families cover generic codegen, method-contract, convention,
and dependency-resilience failures. Semantic, lowering, bridge, and
route-boundary failures refine that same deterministic layer rather than
creating a separate prompt-only path.

Observability And Audit
-----------------------

Learning is observable, not implicit.

Replay and reflection traces can persist:

- the lesson contract report
- the lesson promotion outcome
- related build/run metadata
- source traces used to justify the lesson

That makes the learning mechanism auditable. A later reviewer can see not just
that a lesson exists, but how it was captured, whether the contract passed, and
whether it stayed candidate, validated, promoted, or duplicate.

Semantic Blocker Evidence In Traces
-----------------------------------

Task traces should preserve semantic evidence even when a request fails closed.
For conditional scheduled cashflows, the important distinction is:

- represented shape: the compiler understood the contract structure
- admitted route: a checked executable backend was selected for that structure

Range-accrual traces now use that split. ``SemanticImplementationBlueprint``
can carry:

- ``static_leg_contract_ir`` with the lowered ``ConditionalAccrualLeg``
- ``dynamic_contract_ir`` when callability is represented as an executable
  dynamic wrapper over that static base
- ``static_leg_lowering_selection`` when the checked single-index route admits
  the contract
- ``static_leg_admission_blockers`` when the shape is represented but not
  admitted on the plain static route

``trellis.agent.platform_requests._semantic_blueprint_summary(...)`` persists
those fields in request metadata, and ``trellis.agent.platform_traces`` copies
``dynamic_contract_ir`` and ``static_leg_admission_blockers`` into the
generation-boundary lowering summary. Failure triage, remediation packets, and
future learning prompts should prefer those structured ids and wrapper evidence
over raw error text.

Current conditional-accrual blocker ids include:

- ``conditional_range_accrual_callability_pending``
- ``conditional_range_accrual_interruption_state_pending``
- ``conditional_range_accrual_barrier_state_pending``
- ``conditional_accrual_spread_observable_pending``
- ``conditional_accrual_cms_rate_observable_pending``
- ``conditional_accrual_multi_index_predicate_pending``

Those ids mean "agent has useful semantic evidence, but the named route is not
admitted." For ``conditional_range_accrual_callability_pending``, the plain
static route is still blocked but an issuer-callable deterministic
``dynamic_contract_ir`` may be present. Interruption, barrier, CMS-spread, and
multi-index blockers still mean executable support is not present. They should
not be classified as missing parser knowledge, and
they should not trigger model-validator review of a non-existent payoff.

Short-Term Learning Benchmark
-----------------------------

The near-term learning claim for Trellis is narrower than autonomous library
development.

Today the honest claim is:

- repeated runs can carry forward validated or promoted knowledge
- that carry-forward can reduce failures, retries, elapsed time, and token use
- the effect can be measured without changing the underlying code revision

That is what ``scripts/run_task_learning_benchmark.py`` measures.

The benchmark uses a non-canary cohort from the active pricing-task manifests and repeated passes
at a fixed git revision. By default it also forces fresh builds so the score
is not dominated by trivial adapter reuse. The report records:

- success and failure deltas across passes
- task-level ``first_pass`` and ``attempts_to_success``
- stage-level ``retry_taxonomy``
- elapsed time and token usage
- shared-knowledge and lesson-promotion signals
- attribution buckets for:

  * knowledge-assisted improvements
  * residual knowledge gaps
  * residual implementation gaps
  * residual market/provider noise

That benchmark is the short-term learning milestone because it tests whether
the platform gets better at rerunning broader tasks with knowledge it has
already captured.

Current Boundaries
------------------

The learning loop is stronger than the old artifact-based path, but there are
still limits:

- heuristic diagnosis in ``test_resolution.py`` is still partly hand-authored
- the executor retry path captures candidate lessons but does not auto-promote them
- task reruns and human review are still needed to confirm whether a captured
  lesson should become stable guidance
- intra-run overlays are bounded repair inputs, not canonical cookbook
  updates; they become durable only through the existing validation and
  promotion gates
- the repeated-pass benchmark measures knowledge reuse only; it does not prove
  autonomous code authoring or autonomous primitive implementation

Those limits are intentional. They keep the learning loop explicit and
reviewable rather than silently mutating policy.
