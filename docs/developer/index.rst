Developer Documentation
=======================

The developer documentation is the full platform view. It includes the quant
documentation as a subset, then adds request compilation, hosting and
configuration, audit and issue-sync systems, validation, and task/evaluation
loops.

If you are only changing pricing constructs or knowledge assets, start with
:doc:`../quant/index`. Use this section when you need to run, host, observe,
or extend the platform itself.

Core Developer Topics
---------------------

.. toctree::
   :maxdepth: 1

   overview
   hosting_and_configuration
   audit_and_observability
   task_and_eval_loops
   task_diagnostics

Implementation Journey Notes
----------------------------

.. toctree::
   :maxdepth: 1

   implementation_journey_knowledge_system
   implementation_journey_route_family_handoff

Agent Architecture Reference
----------------------------

.. toctree::
   :maxdepth: 1

   ../agent/architecture
   ../agent/quant_agent
   ../agent/builder_agent
   ../agent/critic_agent
