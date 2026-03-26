Developer Overview
==================

At developer scope, Trellis is more than a pricing library. It is a request
compiler, knowledge-backed build system, validation pipeline, and audit trail
around the deterministic pricing engines.

Platform Surfaces
-----------------

The main entry points all compile into a common internal representation:

- ``trellis.ask(...)`` and ``Session.ask(...)`` for natural-language pricing
- ``Session.price(...)`` and direct market-state workflows for deterministic pricing
- ``Pipeline.run()`` for declarative batch and scenario execution
- structured user-defined and comparison requests in ``trellis.agent.platform_requests``

The canonical request/compiler layer lives in ``trellis.agent.platform_requests``.
It normalizes these surfaces into ``PlatformRequest`` and ``CompiledPlatformRequest``
objects with execution plans, method selection, knowledge payloads, and blocker reports.

Execution Flow
--------------

The operational flow is:

1. create a request from an entry surface
2. compile it into execution intent plus shared knowledge
3. execute deterministic pricing or agent-assisted build/validation
4. append trace events and optional external issue updates

That means developer work often crosses both the quant layer and the runtime
layer. A route-method change can alter knowledge retrieval, audit traces, and
task-batch behavior even when the underlying math is unchanged.

Where Things Live
-----------------

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Concern
     - Main modules
     - Notes
   * - Request compilation
     - ``trellis.agent.platform_requests``
     - Unifies ask, session, pipeline, user-defined, and comparison flows
   * - Agent loop
     - ``trellis.agent.quant``, ``planner``, ``builder``, ``critic``, ``executor``
     - Method routing, spec planning, code generation, and validation
   * - Knowledge system
     - ``trellis.agent.knowledge``
     - Retrieval, promotion, import registry, traces, and canonical YAML assets
   * - Audit and issue sync
     - ``platform_traces``, ``github_tracker``, ``linear_tracker``
     - YAML traces plus best-effort GitHub/Linear issue creation and comments
   * - Validation and evals
     - ``model_validator``, ``validation_report``, ``evals``
     - Deterministic and LLM-assisted grading around generated artifacts
   * - Task runtime
     - ``task_runtime.py``, ``scripts/*.py``, ``TASKS.yaml``, ``FRAMEWORK_TASKS.yaml``
     - Batch execution, reruns, benchmarking, remediation, and separate pricing-vs-framework task inventories

Read Next
---------

- :doc:`hosting_and_configuration`
- :doc:`audit_and_observability`
- :doc:`task_and_eval_loops`
- :doc:`../quant/index`
