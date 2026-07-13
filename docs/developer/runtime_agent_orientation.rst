Runtime Agent Orientation
=========================

Trellis uses two different orientation mechanisms for two different kinds of
agent.

``AGENTS.md`` applies to coding agents operating in a repository worktree. It
defines ownership, implementation workflow, tests, and collaboration rules.
Runtime quant and model-validator calls do not autonomously walk the worktree,
so adding nested ``AGENTS.md`` files would not reliably orient those calls.

Runtime roles instead receive compact, versioned orientation contracts at the
prompt boundary. The canonical source is
``trellis/agent/knowledge/canonical/agent_orientations.yaml`` and the typed
loader is ``trellis.agent.role_orientation``.

Contract Shape
--------------

Each role contract declares:

- a stable contract id and positive version
- a prompt-size budget
- the decisions the role owns and explicitly does not own
- an ordered list of runtime evidence, canonical knowledge, support contracts,
  and official documentation targets

The loader fails closed unless both ``quant`` and ``model_validator`` are
present and every navigation order is consecutive. Tests also verify that all
file-backed targets exist. Runtime targets use the ``runtime:`` prefix because
their evidence is supplied by the compiled request rather than read from disk.

Role Boundaries
---------------

The quant card begins with ``ProductIR`` and semantic-contract evidence, then
points to decompositions, routes, model grammar, method requirements, the
read-only cookbook catalog, and the quant docs index. It does not include the
builder's import-selection surface.

The model-validator card begins with the validation contract, deterministic
evidence, and quant challenger packet. It then points to model and route
contracts, the read-only cookbook catalog, current limitations, calibration
documentation, and audit contracts. This preserves deterministic-first review
ownership.

Prompt And Trace Behavior
-------------------------

Only the role that makes an LLM call receives its rendered card. The quant
card is injected into novel-product decomposition; the model-validator card is
injected into residual conceptual review. Cards are bounded and are not a
mechanism for loading every referenced file into a prompt.

Lifecycle events and cycle reports persist only ``role``, ``contract_id``, and
``version`` under ``orientation_contract``. That low-cardinality identity is
enough for replay and drift review while avoiding duplicate prompt payloads in
traces.

Cookbook Authority
------------------

Both role cards expose ``canonical/cookbooks.yaml`` as read-only validated
pattern evidence. Runtime calls cannot update or promote entries. Candidate
learning remains ephemeral until the governed remediation and promotion path
has independent validation evidence.
