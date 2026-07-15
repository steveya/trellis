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

The version-2 contracts are made operational by
``trellis.agent.orientation_resolution``. The resolver does not give hosted
roles filesystem tools. It follows the declared indexes inside the repository,
selects relevant source sections deterministically, and injects only the
bounded result at the LLM boundary.

Contract Shape
--------------

Each role contract declares:

- a stable contract id and positive version
- separate card, resolved-context, and per-resource character budgets
- the decisions the role owns and explicitly does not own
- an ordered list of runtime evidence, canonical knowledge, support contracts,
  and official documentation targets

The loader fails closed unless both ``quant`` and ``model_validator`` are
present and every navigation order is consecutive. Tests also verify that all
file-backed targets exist. Runtime targets use the ``runtime:`` prefix because
their evidence is supplied by the compiled request rather than read from disk.
File-backed resolution is confined to the repository root. Missing indexed
children and paths that escape that root fail closed. A slim installed wheel
may intentionally omit the repository's top-level documentation tree; in that
case the resolver keeps canonical packaged knowledge, records the unavailable
documentation resource ids as omissions, and continues within the same prompt
budget.

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

Builder API Navigation
----------------------

The builder has a separate navigation surface in
``canonical/api_map.yaml``. It is not part of either runtime role packet.
The typed ``ApiMapQuery`` selector uses product, payoff, method, model,
feature, and route cues to render only the relevant full cards. A no-query
``inspect_api_map`` call returns a bounded catalog of every canonical model
and utility card; callers can then request exact cards or submit semantic
fields. This keeps new canonical cards reachable without adding their names to
a second hand-maintained global order.

Builder prompt traces record selected and omitted API-map card identities.
Rendered cards have a hard character budget and mark truncation explicitly.
Exact symbols remain the import registry's responsibility after family
selection. Quant and model-validator regression tests reject API-map imports
or code templates in their resolved context.

Composition cards may join public primitives from more than one subsystem
without introducing a new product helper. The general
``analytical_gaussian_composition`` card points a builder handling general
critical-state analytical work to scalar Gaussian probability kernels and the
existing typed ``SolveRequest`` root surface. The more specific
``chooser_option_composition`` card is a complete bounded navigation packet:
it adds the scalar-diffusion market projection, contractual time function,
Black call/put kernels, discount/forward support, the bivariate Gaussian
kernel, and the bounded root contracts. It states the adapter-owned balance
equation and date/bracket obligations without importing the retained chooser
pricing wrapper. ``compound_option_composition`` supplies the corresponding
complete packet for European options on European options, adds the univariate
Gaussian kernel, and states the four subtype, critical-state, strict date, and
one-time scaling obligations without importing the retained compound wrapper.

Structured lane cards render every selected primitive in these bounded route
packets. They do not silently truncate the final solver or probability symbol;
compactness is enforced when the canonical semantic query selects the card,
not by making a selected composition incomplete.

Neither card enters the quant or model-validator role packets. Those roles
continue to reason from semantic contracts, quantitative documentation, and
executed evidence; implementation symbol selection remains builder-owned.
The quant and model-validator may consult the read-only cookbook catalog after
typed decomposition, but they cannot promote cookbook entries during task
execution or use those patterns to override chooser or compound semantic
contracts.

The ``fixed_lookback_analytical_composition`` card is the corresponding
builder-only analytical hot start. It selects scalar market resolution, exact
time, option normalization, discounting, and the univariate Gaussian CDF. It
does not inherit the bivariate probability or scalar-root imports from
``analytical_gaussian_composition``. The card states that generated code owns
the Conze-Viswanathan call/put formula, historical and runtime extreme
invariants, the analytic zero-carry limit, expiry settlement, and one-time
notional scaling. The retained analytical lookback wrapper remains comparison
evidence. Quant and model-validator continue to receive the semantic contract,
quantitative documentation, and executed price/Delta evidence rather than
implementation imports, and no cookbook entry is promoted by this route.

The ``path_statistic_composition`` card applies the same rule to path-dependent
Monte Carlo construction. Semantic aliases such as ``running_extremum``,
``squared_log_return``, and ``variance_swap`` lead the builder to exact
observation-step contracts, full-path parity functions, and bounded reducers.
The bare ``lookback_option`` alias is intentionally absent because an admitted
continuous lookback must not inherit discrete-observation state. For admitted
variance-swap Monte Carlo, the card also points to the product-neutral
scalar-diffusion market resolver and states that generated code owns
annualization and settlement. The general Monte Carlo card no longer
advertises the lookback or variance-swap product pricers as construction
imports. Quant still selects and challenges the model from product semantics,
and model-validator still reviews monitoring, annualization, calibration, and
residual numerical risk; neither role receives implementation imports.

The ``conditional_extremum_composition`` card is a separate continuous-state
route. It points builders to ``ScalarTransitionObservation``,
``ConditionalBridgeExtremumContract``, the transition reducer, exact
constant-parameter ``GBM`` bridge capability, and
``sobol_transition_inputs(...)``. It explicitly rejects treating discrete
``PathReducer`` state, Brownian path construction, or a product lookback helper
as the conditional-extremum primitive. The card also records the current
one-stochastic-reducer boundary and keeps derivative settlement in generated
adapter code. ``lookback_option`` plus a Monte Carlo method selects this card
only after the semantic contract has established fixed-strike continuous
monitoring. Canonical legacy tasks carry those terms in their manifest-backed
contracts; arbitrary sparse requests do not receive product defaults. The
admitted route then validates continuous monitoring, initializes the reducer
from the contractual running extremum, and fails closed for unsupported
dynamics or monitoring styles. Missing or conflicting strike/monitoring
semantics, floating strike, discrete monitoring, and non-scalar-GBM dynamics
become structured ``ProductIR.unresolved_primitives``. The pre-generation gate
blocks those contracts before an LLM can synthesize a generic Monte Carlo
fallback. Assisted recovery may clear those blockers only by producing an
explicit contract that matches an admitted route.

This separation is important for small-context agents. The semantic query
chooses one complete composition card, the card names the minimal public
symbols and ownership boundaries, and the import registry confirms those
symbols. The builder does not need to search the full package or infer the
private accumulator layout of a ``PathReducer``.

Bounded Resolution
------------------

The resolver builds a typed semantic query from the instrument, method,
features, model family, route identity, residual risks, and review trigger. It
then:

#. projects the already-ranked ``KnowledgeStore`` result into role-safe
   decomposition, promoted/validated lesson, model-grammar,
   method-requirement, and read-only
   cookbook evidence
#. consults the generated skill index but rejects route-helper records and
   construction-shaped guidance for these roles
#. follows declared RST ``toctree`` entries, including Markdown targets, and
   ranks actual document sections against the same query
#. strips code blocks and builder-only import/helper instructions
#. truncates each resource and the final packet independently, recording any
   omissions

Documentation ranking scores only the bounded text that could enter the
prompt. A keyword that appears later in a long unrelated section therefore
cannot make that section outrank a directly relevant heading.

Prompt And Trace Behavior
-------------------------

Only the role that makes an LLM call receives its rendered card. The quant
card and resolved context are injected into novel-product decomposition; the
model-validator card and resolved context are injected into residual conceptual
review. Known-product deterministic quant selection does not pretend that a
prompt packet was used.

Lifecycle events and cycle reports persist ``role``, ``contract_id``, and
``version`` under ``orientation_contract``. They persist a separate
``orientation_resolution`` summary containing whether a packet was injected,
selected resource and section ids, character and omission counts, and a content
digest. It also identifies documentation resources unavailable in a slim
installed runtime. Full excerpts are not copied into traces.

Cookbook Authority
------------------

Both role cards expose ``canonical/cookbooks.yaml`` as read-only validated
pattern evidence. The resolver exposes method descriptions and matching
solution-contract assumptions, payoff meaning, and market-data obligations; it
does not expose cookbook code templates to quant or model-validator. Runtime
calls cannot update or promote entries. Candidate learning remains ephemeral
until the governed remediation and promotion path has independent validation
evidence.
