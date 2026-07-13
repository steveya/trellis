Model Validator Agent
=====================

The runtime model-validator agent reviews residual conceptual, calibration,
numerical-model, and limitation risk after deterministic validation has
executed. It does not repeat deterministic checks, select routes, rewrite
generated code, review code style, or promote cookbook entries.

Runtime Orientation
-------------------

The LLM review receives the versioned ``model-validator-runtime-navigation``
card rendered from
``trellis/agent/knowledge/canonical/agent_orientations.yaml``. Its navigation
order starts with runtime evidence:

#. the compiled validation contract
#. the deterministic evidence packet
#. the quant challenger packet
#. admitted route, model-grammar, and method-requirement context
#. cookbook patterns as read-only supporting evidence
#. ``LIMITATIONS.md`` and the validation, calibration, quant, and audit docs

This order matters. A deterministic route, binding, or validation-bundle
failure is already concrete evidence and should not be converted into a second
LLM opinion. The validator is useful when executed evidence leaves a genuine
question about model suitability, calibration, numerical quality, or a support
boundary.

Review Policy
-------------

Build-time LLM model validation remains governed by the validation profile and
risk policy. Skipped and completed lifecycle events both record the orientation
contract identity, so an audit can establish which role contract governed a
review without persisting the full prompt card.

Implementation
--------------

.. autofunction:: trellis.agent.model_validator.validate_model

See Also
--------

- :doc:`../developer/runtime_agent_orientation` for the shared orientation
  contract
- :doc:`../developer/audit_and_observability` for cycle-report evidence
- :doc:`../mathematical/calibration` for calibration contracts and boundaries
- :doc:`../quant/index` for the quantitative documentation index
