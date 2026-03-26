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

Related Reading
---------------

- :doc:`audit_and_observability`
- :doc:`../quant/knowledge_maintenance`
