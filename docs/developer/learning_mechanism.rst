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

Short-Term Learning Benchmark
-----------------------------

The near-term learning claim for Trellis is narrower than autonomous library
development.

Today the honest claim is:

- repeated runs can carry forward validated or promoted knowledge
- that carry-forward can reduce failures, retries, elapsed time, and token use
- the effect can be measured without changing the underlying code revision

That is what ``scripts/run_task_learning_benchmark.py`` measures.

The benchmark uses a non-canary cohort from ``TASKS.yaml`` and repeated passes
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
- the repeated-pass benchmark measures knowledge reuse only; it does not prove
  autonomous code authoring or autonomous primitive implementation

Those limits are intentional. They keep the learning loop explicit and
reviewable rather than silently mutating policy.
