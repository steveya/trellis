Knowledge Maintenance
=====================

Trellis has two knowledge layers in the codebase:

- the canonical knowledge system in ``trellis/agent/knowledge/``
- legacy compatibility shims in ``trellis/agent/experience.py`` and ``trellis/agent/cookbooks.py``

New knowledge should be maintained in the canonical store first. The legacy files
exist so older call sites keep working.

What To Maintain
----------------

The canonical files are split by responsibility:

- ``canonical/features.yaml``: feature atoms and transitive ``implies`` chains
- ``canonical/decompositions.yaml``: product decomposition, preferred method, and method modules
- ``canonical/cookbooks.yaml``: method-family templates used during code generation
- ``canonical/data_contracts.yaml``: input conventions and conversions
- ``canonical/method_requirements.yaml``: modeling constraints that must hold
- ``canonical/failure_signatures.yaml``: regex-driven failure interpretation
- ``lessons/index.yaml`` and ``lessons/entries/*.yaml``: promoted, candidate, and archived lessons
- ``traces/``: cold-store build traces and platform audit traces

Lesson Lifecycle
----------------

The promotion pipeline in ``trellis.agent.knowledge.promotion`` is explicit:

1. ``capture_lesson(...)`` writes a candidate lesson if it is not a duplicate.
2. ``validate_lesson(...)`` moves it to ``validated`` when the fix is non-empty and confidence is at least ``0.6``.
3. ``promote_lesson(...)`` moves it to ``promoted`` when confidence reaches at least ``0.8``.
4. ``archive_lesson(...)`` retires stale or superseded knowledge without deleting history.

That lifecycle is important for quant maintenance because low-confidence lessons
should not quietly become production guidance.

Cookbooks, Memory, And Experience
---------------------------------

Cookbooks are now canonical YAML assets, while ``trellis.agent.cookbooks`` is a
thin compatibility shim that reads from them. The same is true for experience:
``trellis.agent.experience`` delegates to feature-based retrieval from the
knowledge store and falls back to the older YAML layout only when needed.

In practice:

- add new method templates in ``canonical/cookbooks.yaml``
- keep cookbook examples method-generic and import-safe
- use lessons for failures or edge cases, not for normal method definitions
- treat ``experience.yaml`` and ``experience/index.yaml`` as backward-compatibility artifacts

Recommended Maintenance Loop
----------------------------

For quant-facing knowledge work:

1. update the relevant canonical YAML files
2. run the knowledge-store tests
3. run remediation analysis if the change responds to task failures
4. re-run the affected tasks or targeted product tests

Useful commands:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_knowledge_store.py -x -q
   /Users/steveyang/miniforge3/bin/python3 scripts/remediate.py --analyze-only

Related Reading
---------------

- :doc:`extending_trellis`
- :doc:`../developer/audit_and_observability`
- :doc:`../developer/task_and_eval_loops`

