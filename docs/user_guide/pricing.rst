Pricing
=======

Three Ways to Price
-------------------

**1. Direct** (quant writing code):

.. code-block:: python

   from trellis import price, Bond, YieldCurve
   result = price(bond, curve, settlement)

**2. Session** (trader in a notebook):

.. code-block:: python

   s = Session(curve=curve)
   result = s.price(bond)

**3. Natural language** (anyone):

.. code-block:: python

   result = trellis.ask("Price a 5Y cap at 4%")

Unified Runtime Boundary
------------------------

``trellis.ask(...)``, ``Session.ask(...)``, ``Session.price(...)``,
``Session.greeks(...)``, ``Session.analyze(...)``, and ``Pipeline.run()`` now
all compile into a platform request and execute through
``trellis.platform.executor.execute_compiled_request(...)``.

That unification is intentionally internal: the user-facing return types stay
the same, so ``AskResult``, ``PricingResult``, ``BookResult``, and the
analytics containers remain the notebook and scripting contracts.

Convenience market-data helpers such as ``Session(data_source="mock")`` still
work, but they now normalize into an explicit sandbox execution context behind
the scenes instead of defining a separate runtime path.

The Payoff Framework
--------------------

For instruments beyond bonds, use the ``Payoff`` protocol:

.. code-block:: python

   from trellis import CapPayoff, CapFloorSpec, price_payoff, MarketState
   from trellis.core.types import Frequency

   spec = CapFloorSpec(notional=1e6, strike=0.04,
                        start_date=start, end_date=end,
                        frequency=Frequency.QUARTERLY)
   cap = CapPayoff(spec)

   ms = MarketState(as_of=settle, settlement=settle,
                     discount=curve, vol_surface=FlatVol(0.20))
   pv = price_payoff(cap, ms)

Payoff ``requirements`` now use canonical market-data capability names. The
main labels are:

- ``discount_curve``
- ``forward_curve``
- ``black_vol_surface``
- ``credit_curve``
- ``fx_rates``
- ``spot``
- ``local_vol_surface``

FX Vanilla Options
------------------

FX vanillas are priced by resolving spot FX, domestic discounting, foreign
discounting, and Black volatility into a resolved-input helper surface, then
delegating to the checked analytical Garman-Kohlhagen helper. The route keeps
the Black76 basis decomposition explicit inside the helper, so the same pricing
math can be reused for both valuation and Greeks without rebuilding it in each
adapter.

Rates Calibration
-----------------

Cap/floor and swaption quotes use the same explicit multi-curve discipline.
The calibration helpers in ``trellis.models.calibration.rates`` solve for a
flat Black volatility against the selected discount and forecast curves, and
the returned calibration result keeps the selected curve names plus any caller
labels for the volatility or correlation source. That makes the calibration run
replayable without re-resolving market data.

Unified Lattice Pricing
-----------------------

The checked lattice entry points are now:

- ``trellis.models.trees.build_lattice(...)``
- ``trellis.models.trees.price_on_lattice(...)``

Those surfaces cover one-factor short-rate trees, CRR/Jarrow-Rudd equity
lattices, local-volatility trinomial lattices, and low-dimensional two-factor
product lattices. Legacy helpers such as ``build_rate_lattice(...)`` and
``build_spot_lattice(...)`` still work, but they are compatibility shims and
emit deprecation warnings.

For plain equity/rate pricing, prefer the helper routes or recipe compilers.
For new lattice integrations, target ``LatticeRecipe`` and the unified builder
instead of hand-assembling route-local tree logic.

Semantic Request Compilation
----------------------------

When a pricing request goes through the platform compiler, Trellis now tries to
draft a typed semantic contract first and only falls back to older family-local
paths when no checked semantic slice exists or the request is too incomplete.

This means helper-backed requests such as vanilla options, callable bonds,
rate-style swaptions, ranked-observation baskets, single-name CDS, and
nth-to-default basket credit now carry explicit contract, market-binding, and
route-lowering metadata before pricing or code generation starts.

They also now carry compiler-emitted lane obligations. In practice that means
the build loop sees the computational lane first (analytical, lattice, Monte
Carlo, PDE, and so on), the timeline and market bindings that lane requires,
and only then any exact checked backend binding that already satisfies those
obligations.

When the compiler does find such a checked backend, Trellis now records it as a
route-binding authority packet. That packet is not a second planning language;
it is a backend/provenance record containing the exact route ID, helper refs,
approved modules, validation bundle, and any canary tasks that cover the
binding. Build and review prompts consume that packet only after the lane
obligations so route guidance stays a backend-fit constraint rather than a
route-first synthesis plan.

For simple-derivative proving and regression, Trellis also supports a
knowledge-light build profile through
``scripts/run_knowledge_light_proving.py``. That mode intentionally minimizes
route-local cookbook help so the generated adapter has to lean on the semantic
contract, lane obligations, validation contract, and any exact backend helper
signature surfaced by the compiler.

For schedule-bearing requests, the practical boundary is now an explicit
timeline shape: ``ContractTimeline`` internally and ``tuple[date, ...]`` at the
agent-facing spec layer. Comma-separated date strings are no longer the
supported way to express call, put, exercise, or observation schedules.

Typed Governed Parse And Match
------------------------------

The governed MCP/runtime path now treats the semantic-contract stack as the
canonical trade-identity boundary.

That means:

- ``trellis.platform.services.TradeService`` normalizes natural-language or
  structured requests into a typed semantic contract plus ``ProductIR``
- incomplete requests surface explicit ``missing_fields`` and ``warnings``
  instead of silently dropping back into an opaque parse step
- ``TermSheet`` parsing remains available for older ask-style compatibility, but
  it is not the canonical governed contract boundary
- ``trellis.platform.services.ModelService`` performs deterministic model
  matching from typed registry criteria such as semantic id, exercise style,
  currencies, method family, and required market data
- lifecycle execution eligibility is still enforced later by the model
  execution gate; the match surface explains structural fit, it does not
  silently approve execution on its own

Governed MCP Price Trade MVP
----------------------------

The first MCP pricing surface is now ``trellis.price.trade``.

This MVP is intentionally narrow:

- it executes approved models only
- it requires explicit governed session/provider state
- it persists a canonical run record and canonical audit bundle for every
  successful or blocked call
- it does not silently fall back to candidate generation when no approved model
  is execution-eligible

The response can be projected in three modes:

- ``concise`` returns status, result summary, warnings, and a minimal audit
  pointer
- ``structured`` returns run id, structured result payload, warnings,
  provenance, and audit URI
- ``audit`` returns the structured payload plus the canonical audit bundle

Blocked responses are part of the contract. Missing provider bindings,
non-approved model matches, policy denial, and unsupported MVP execution
adapters return persisted blocked runs rather than opaque transport errors.

Testing The Local MCP Server
----------------------------

If you want to exercise the governed MCP surface from a host client instead of
calling Trellis directly in Python, the repo now ships a local streamable HTTP
wrapper.

Install the optional MCP dependency first:

.. code-block:: bash

   pip install -e ".[mcp]"

Start the local Trellis MCP server from the repo root:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/serve_trellis_mcp.py \
     --host 127.0.0.1 \
     --port 8000

That standard launch preserves the governed defaults. On a fresh local state
root, that means:

- new sessions start in ``research`` mode
- mock market data stays disabled by default
- ``trellis.price.trade`` still requires an approved governed model match

That is the right behavior for normal governed testing, but it is not the best
first-run experience if you only want to prove that prompt-to-output host
plumbing works.

For localhost prompt-flow demos, start the server in explicit demo mode:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/serve_trellis_mcp.py \
     --host 127.0.0.1 \
     --port 8000 \
     --demo \
     --demo-session-id demo

Demo mode is intentionally opt-in and local-only. It seeds:

- a ``demo`` session in ``sandbox`` mode
- explicit ``market_data.mock`` binding for that session
- mock-data allowance enabled for the session policy context
- a minimal approved built-in vanilla-option model so ``trellis.price.trade``
  can complete the standard AAPL European-option smoke flow

Use demo mode when you want to verify that Codex or Claude Code can drive the
MCP tool surface end to end. Do not treat demo mode as production governance or
market-quality validation.

The default MCP endpoint is:

.. code-block:: text

   http://127.0.0.1:8000/mcp

You can then attach a local MCP host to that URL.

For Codex:

.. code-block:: bash

   codex mcp add trellis --url http://127.0.0.1:8000/mcp

For Claude Code:

.. code-block:: bash

   claude mcp add --transport http trellis http://127.0.0.1:8000/mcp

After the host is connected, test the surface with the governed MCP tools
rather than ad hoc Python calls.

If you started with ``--demo``, a practical first pass is:

1. call ``trellis.session.get_context`` for session id ``demo`` and confirm
   that the run mode is ``sandbox`` and ``market_data.mock`` is bound
2. call ``trellis.trade.parse`` or ``trellis.model.match`` for
   ``European call on AAPL with strike 120 expiring 2026-12-31``
3. call ``trellis.price.trade`` with session id ``demo``
4. inspect the returned ``run_id`` plus
   ``trellis://runs/{run_id}`` and ``trellis://runs/{run_id}/audit``

If you started without ``--demo``, the practical first pass is:

1. call ``trellis.session.get_context`` for a fresh session id
2. call ``trellis.providers.list`` and ``trellis.providers.configure`` to bind
   market data explicitly
3. call ``trellis.trade.parse`` or ``trellis.model.match`` for a simple
   vanilla trade
4. call ``trellis.price.trade`` and inspect the returned ``run_id`` plus the
   corresponding ``trellis://runs/{run_id}`` and ``trellis://runs/{run_id}/audit``
   resources

This local host flow is intentionally thin:

- it uses the same tool, resource, and prompt contracts as the in-process
  Trellis MCP shell
- it is designed for localhost testing in Codex and Claude Code
- demo mode is a local prompt-flow convenience, not a replacement for the
  standard governed path
- public exposure, auth, and ChatGPT-facing remote deployment are separate
  follow-on concerns

Governed Candidate Lifecycle
----------------------------

The governed MCP surface now also exposes an explicit candidate-model lifecycle
instead of treating research generation as an implicit production path.

The current lifecycle tools are:

- ``trellis.model.generate_candidate`` to persist a draft model version plus
  canonical contract, methodology, lineage, validation-plan, and optional code
  artifacts
- ``trellis.model.validate`` to run deterministic manifest-level validation and
  persist a governed validation report tied to one exact model version
- ``trellis.model.promote`` to apply explicit ``validated``, ``approved``, or
  ``deprecated`` lifecycle transitions after review

Important constraints:

- draft generation does not make a model execution-eligible
- deterministic validation does not auto-approve a model
- production pricing still requires an explicitly approved model version
- deprecation removes execution eligibility without deleting the stored
  contract, code, validation, or lineage artifacts

Governed Model Store And Replay Bundles
---------------------------------------

The same governed registry now exposes a first-pass versioned model-store
surface:

- ``trellis.model.persist`` writes a new governed version with explicit lineage
  instead of mutating the previous one in place
- ``trellis.model.versions.list`` returns the stored version history for one
  governed model id
- ``trellis.model.diff`` compares two governed versions across contract, code,
  methodology, validation, and lineage metadata
- ``trellis.snapshot.persist_run`` turns a persisted governed run into a
  reproducibility bundle snapshot that can be referenced later by run id

The matching read surface is resource-based rather than write-tool based. The
main URIs are:

- ``trellis://models/{model_id}`` and version-sidecar URIs for contract, code,
  and validation reports
- ``trellis://runs/{run_id}``, ``.../audit``, ``.../inputs``, and ``.../outputs``
  for governed run inspection
- ``trellis://market-snapshots/{snapshot_id}`` for persisted market snapshots

Versioning constraints matter here:

- every governed model version needs its own validation result before it can be
  promoted
- promotion uses the latest validation for that exact version, not any
  historical pass from an earlier validation run or parent version
- metadata-only revisions still keep a version-specific code resource so later
  audit or regulatory review can inspect the implementation tied to the version
  being discussed
  and reproducibility bundles

Return Types
~~~~~~~~~~~~

``evaluate()`` returns either:

- **Cashflows** — undiscounted dated cashflows (bonds, caps, swaps)
- **PresentValue** — already-discounted PV (tree, MC, PDE methods)

``price_payoff()`` handles both automatically.

Greeks
------

Greeks are computed via automatic differentiation (autograd):

.. code-block:: python

   result = s.price(bond, greeks="all")
   print(result.greeks["dv01"])
   print(result.greeks["duration"])
   print(result.greeks["convexity"])
   print(result.greeks["key_rate_durations"])

   # Selective:
   result = s.price(bond, greeks=["dv01", "duration"])
   # or skip Greeks entirely:
   result = s.price(bond, greeks=None)

The same rule now applies to pipelines:

- ``compute(["price"])`` keeps pricing-only results with empty Greeks
- ``compute(["price", "dv01"])`` requests explicit risk outputs
- omitting ``compute(...)`` still defaults to the full pricing Greek set

Trace Artifacts
---------------

When a request goes through the build loop, Trellis records a machine-readable analytical trace plus a Markdown rendering of the same build steps. The task result and persisted task-run record keep the trace paths even for reused analytical routes, so you can inspect route selection, decomposition, validation, fallback decisions, and cache/reuse behavior after the run.

For migrated semantic families, the platform trace also includes the selected
DSL route, the typed family IR payload, the helper targets, and any structured
lowering errors. That makes it possible to see why a request selected, for
example, a CDS helper-backed analytical route or an nth-to-default copula
helper instead of a generic analytical or generic copula path.

The same trace now includes the compiled validation contract summary, including
deterministic check ids and normalized comparison relations such as
``within_tolerance``, ``<=``, and ``>=``. That makes comparison-task and
bound-check failures easier to replay without inferring intent from route names.

The trace also records semantic-validator failures tied to the compiled
primitive contract. If a generated adapter ignores a required checked helper,
the failure now shows up before the build is accepted instead of surfacing only
as a later comparison or runtime error.

For multi-curve market data, the compiled ``MarketState`` also remembers which
named curve was selected for the runtime path, so the trace can show both the
available curve set and the exact curve contract that was used to price the
request. That selection provenance is also repeated in the runtime contract and
task-run record so replay tools can explain the chosen discount, forecast, or
credit curve without re-resolving market data.

If the curves were bootstrapped from rate instruments, that source-selection
step is still preserved: the snapshot holds the named bootstrapped curves and
the runtime state records which one was actually used.

Missing Data Errors
-------------------

If required market data is missing, you get a helpful error:

.. code-block:: text

   MissingCapabilityError: Missing market data: ['black_vol_surface']
     Missing market data: 'black_vol_surface' — Black (lognormal) implied volatility surface.
       How to provide: Session(vol_surface=FlatVol(0.20))
