Quant Documentation
===================

The quant documentation sits between the task-oriented :doc:`../user_guide/pricing`
and the broader :doc:`../developer/index`. It focuses on the pricing stack itself:
mathematical constructs, computational methods, agent linkage, and maintenance of
the knowledge assets that help Trellis reason about products and methods.

Use this section when you need to:

- understand how ``Session``, ``MarketState``, ``Payoff``, and the method families fit together
- extend instruments, payoffs, curves, models, or pricing engines
- connect deterministic pricing code to the agent build path
- maintain lessons, decompositions, cookbooks, and data-contract knowledge

Core Quant Topics
-----------------

.. toctree::
   :maxdepth: 1

   pricing_stack
   contract_algebra
   contract_ir
   dsl_algebra
   lattice_algebra
   differentiable_pricing
   analytical_route_cookbook
   basis_claim_patterns
   extending_trellis
   knowledge_maintenance

Mathematical And Computational Reference
----------------------------------------

.. toctree::
   :maxdepth: 1

   ../mathematical/bond_pricing
   ../mathematical/black76
   ../mathematical/swap_pricing
   ../mathematical/credit
   ../mathematical/fx
   ../mathematical/trees
   ../mathematical/monte_carlo
   ../mathematical/pde
   ../mathematical/transforms
   ../mathematical/processes
   ../mathematical/copulas
   ../mathematical/calibration
   ../mathematical/cashflow_engine
