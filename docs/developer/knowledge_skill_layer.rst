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

Current public API
------------------

Use these entry points:

* ``load_skill_index()`` for the full generated index
* ``get_skill_record(skill_id)`` for one record
* ``query_skill_records(...)`` for deterministic metadata filtering
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

Non-goals
---------

The generated skill layer does not yet own:

* adaptive ranking
* artifact effectiveness telemetry
* generalized lineage across artifact types

Those remain separate follow-on issues built on top of this substrate.
