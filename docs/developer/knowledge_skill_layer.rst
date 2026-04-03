Generated Skill Layer
=====================

The generated skill layer is a deterministic projection over the existing
knowledge artifacts. It does not replace the authored sources.

Source of truth
---------------

The source artifacts remain:

* lesson entries in ``trellis/agent/knowledge/lessons/entries/``
* canonical cookbooks in ``trellis/agent/knowledge/canonical/cookbooks.yaml``
* canonical principles in ``trellis/agent/knowledge/canonical/principles.yaml``
* canonical and discovered route guidance in ``trellis/agent/knowledge/canonical/routes.yaml``
  and ``trellis/agent/knowledge/routes/entries/``

Generated view
--------------

``trellis.agent.knowledge.skills`` projects those sources into typed
``SkillRecord`` entries and returns them through a deterministic
``GeneratedSkillIndex``.

The generated layer exists so retrieval, telemetry, and lineage work can read
one normalized surface without having to understand every underlying storage
layout.

Current executor use
--------------------

The retry loop now consumes this generated index for deterministic,
stage-aware skill selection. Retry-time prompt repair can pull a small set of
matching lesson, principle, cookbook, and route-hint records based on the
current failure stage, instrument/method scope, and compiled route boundary.
Those selected artifact ids are recorded in retrieval history, platform
traces, and task-diagnosis packets.

Current agent adoption
----------------------

The compiled shared-knowledge bundle now consumes generated skills for the
initial builder, review, and routing contexts as well.

That means:

* ``build_shared_knowledge_payload()`` appends deterministic generated-skill
  guidance to the builder, review, and routing prompt views.
* compiled requests and user-defined product compilation carry the selected
  generated artifact ids/titles in ``knowledge_summary``.
* the retry loop still layers additional stage-aware repair guidance on top of
  that shared bundle when a build advances past the initial attempt.

Cold-start similar-product retrieval
------------------------------------

The shared retrieval payload now also carries a deterministic similar-product
layer for sparse or novel builds.

When ``retrieve_for_task()`` or ``gap_check()`` sees a request with no static
decomposition, no promoted route, or thin lesson coverage, the knowledge store
now:

* ranks the nearest canonical products by expanded-feature overlap plus method
  compatibility
* records any promoted route ids attached to those nearby products
* borrows a small number of extra lessons from the top matches when the
  primary lesson set is intentionally sparse

The important point is that this does not create a prompt-only side channel.
``build_shared_knowledge_payload()`` threads the same ``similar_products`` and
``borrowed_lessons`` data through the builder, review, and routing prompt
views, and the resulting ``knowledge_summary`` now exposes the matched product
ids/scores plus the borrowed lesson ids/titles for traces and diagnostics.

Current public API
------------------

Use these entry points:

* ``load_skill_index()`` for the full generated index
* ``load_skill_lineage_index()`` for the cached lineage view keyed by
  ``skill_id``
* ``get_skill_record(skill_id)`` for one record
* ``get_skill_lineage(skill_id)`` for surfaced ancestry, supersession, and
  same-source context
* ``query_skill_records(...)`` for deterministic metadata filtering
* ``select_prompt_skill_artifacts(...)`` for audience/stage-aware prompt
  selection over the generated skill layer
* ``augment_prompt_with_skill_records(...)`` to append the selected records to
  one prompt surface
* ``clear_skill_index_cache()`` when tests or maintenance tooling need a fresh rebuild

Current scope
-------------

The first pass projects four artifact kinds:

* ``lesson``
* ``principle``
* ``cookbook``
* ``route_hint``

``route_hint`` records include deterministic instruction-lifecycle guidance
derived from the route registry, such as route-helper constraints,
schedule-builder hints, and route-note precedence.

Selection and trace surface
---------------------------

Shared knowledge summaries now expose:

* ``selected_artifact_ids``
* ``selected_artifact_titles``
* ``selected_artifacts_by_audience``

Those summary fields are the stable bridge between generated-skill selection
and downstream task traces, diagnostics, and telemetry rollups.

Telemetry rollups
-----------------

``trellis.agent.task_run_store`` now turns those selections into deterministic
per-run telemetry and rebuildable rollups.

The first-pass telemetry surface includes:

* run outcome and retry/degradation flags
* selected-artifact observations keyed by ``artifact_id``
* route observations keyed by route id and route family
* aggregated counters via ``load_latest_skill_telemetry_rollup()``
* route-health counters via ``load_latest_route_health_rollup()``
* stable ranking-input projections via ``load_latest_skill_ranking_inputs()``
  and ``load_latest_route_ranking_inputs()``

Route-health observations also carry instruction-resolution counts from
analytical traces when they exist, so maintenance tooling can answer both
"which skills were present for this outcome?" and "how healthy was the route
selection that consumed them?" without reparsing raw trace markdown.

The retained ranking-input contract is intentionally small:

* success, failure, blocked, retry, and degradation rates
* average retry count
* route/task coverage counts
* ``first_seen_at`` and ``last_seen_at`` recency markers

Lineage contract
----------------

Every generated ``SkillRecord`` now carries an explicit provenance shape:

* ``origin``
* ``source_kind``
* ``source_artifact``
* ``source_path``
* ``parents``
* ``supersedes``
* ``lineage_status``
* ``lineage_evidence``

The status/evidence pair makes absence explicit:

* ``source_root`` means the record is a source node for the current generated
  layer and no stronger parent/supersession claim is made.
* ``derived`` means the record has explicit parents.
* ``superseding`` means the record explicitly replaces older guidance.
* ``advisory`` means a plausible backfill existed but remained intentionally
  unresolved because the evidence was ambiguous.

Current populated lineage edges are:

* principles derive from their ``derived_from`` lesson set
* lessons preserve explicit lesson-to-lesson supersession when present
* route hints inherit cookbook parents when the route card names a method
  family with a matching canonical cookbook

Selected prompt artifacts now also carry compact lineage metadata
(``lineage_status`` and ``lineage_summary``), so retrieval/debug surfaces can
show not just the active guidance text but where that guidance came from.

The cleanup pass intentionally leaves ambiguous route-to-cookbook backfills
unresolved. If a route card names multiple cookbook-backed method families, the
record stays advisory rather than fabricating a parent edge.

Non-goals
---------

The generated skill layer does not yet own:

* adaptive or learned ranking beyond deterministic stage policies
* automatic prompt mutation or skill suppression policy
* generalized lineage across artifact types

Those remain separate follow-on issues built on top of this substrate.
