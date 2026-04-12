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

For fixed-rate bonds, the returned ``PricingResult`` now includes a solved
``ytm`` alongside ``clean_price``, ``dirty_price``, and
``accrued_interest``. Trellis computes accrued interest from the bond's coupon
schedule and day-count convention, and solves ``ytm`` as a nominal annual
yield compounded at the bond coupon frequency. Current bond reporting does not
model ex-coupon windows or settlement-lag conventions, so exact street parity
outside that scope should still be validated explicitly.

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

The corresponding quanto routes now follow the same shape. The supported
analytical and Monte Carlo slices bind through
``trellis.models.quanto_option`` while the FX vanilla analytical slice binds
through ``trellis.models.fx_vanilla``. The checked-in adapters under
``trellis.instruments._agent`` are intentionally thin compatibility shells
over those semantic-facing helpers rather than separate pricing
implementations.

Rates Calibration
-----------------

Cap/floor and swaption quotes use the same explicit multi-curve discipline.
The calibration helpers in ``trellis.models.calibration.rates`` solve for a
flat Black volatility against the selected discount and forecast curves, and
the returned calibration result keeps the selected curve names plus any caller
labels for the volatility or correlation source. Those helpers now also record
the typed ``SolveRequest`` payload and solved result metadata used for the
scalar root solve, so the calibration run is replayable without re-resolving
market data or reverse-engineering solver inputs from backend-specific calls.

Calibration workflows now also expose an explicit quote-map contract. Supported
quote families are ``Price``, ``ImpliedVol(Black)``, ``ImpliedVol(Normal)``,
``ParRate``, ``Spread``, and ``Hazard``. When a workflow needs an inverse
transform, failures are reported in calibration provenance instead of being
hidden inside helper logic.

Under the hood, those solve requests now dispatch through a backend registry.
The default backend is still SciPy, but backend capability checks now block
unsupported constraints, derivative hooks, or other solve features unless the
caller explicitly requests a fallback backend. That keeps calibration behavior
auditable instead of silently dropping unsupported solver features.

Equity-vol calibration helpers follow the same audit-first approach. The
supported Dupire local-vol workflow can now return a structured
``LocalVolCalibrationResult`` whose diagnostics and warning list make unstable
surface regions explicit. The legacy ``dupire_local_vol(...)`` helper still
returns a callable for model consumers, but that callable now carries sibling
``calibration_provenance``, ``calibration_diagnostics``, and
``calibration_warnings`` attributes so downstream review tools do not have to
guess whether a local-vol point came from a stable Dupire evaluation or a
fallback to the implied-vol surface.

The supported Heston smile workflow now follows the same pattern.
``calibrate_heston_smile_workflow(...)`` fits one supported strike smile onto a
runtime-ready Heston parameter set and returns a ``HestonSmileCalibrationResult``
with the typed solve request, solver provenance/replay artifacts, fit
diagnostics, and a reusable ``apply_to_market_state(...)`` helper. That means
later simulation or pricing code can consume the calibrated parameters from the
canonical ``model_parameters`` surface instead of rebuilding a route-local
process object.

The same MarketState handoff now exists for the local-vol workflow as well.
``calibrate_local_vol_surface_workflow(...)`` returns the hardened Dupire
result and can project the named local-vol surface back onto
``MarketState.local_vol_surface`` / ``local_vol_surfaces`` for later runtime
consumers.

Those migrated calibration workflows now share one typed runtime binding
surface as well. ``apply_to_market_state(...)`` still keeps the compatibility
fields populated, but the authoritative binding metadata lives under
``MarketState.market_provenance["calibrated_objects"]`` and
``MarketState.market_provenance["selected_calibrated_objects"]``. Use
``MarketState.materialized_calibrated_object(object_kind=...)`` when you need
to inspect which calibrated parameter set or surface is currently selected,
which workflow produced it, and which multi-curve roles were bound at runtime.

Single-name CDS calibration now follows that same pattern. The supported entry
point is ``calibrate_single_name_credit_curve_workflow(...)``, which accepts
tenor quotes in spread or hazard form, records the discount-plus-default
potential binding explicitly, and materializes the calibrated ``CreditCurve``
through the shared runtime binding surface. The current slice is intentionally
bounded to reduced-form single-name credit; basket credit and hybrid credit
calibration remain out of scope.

Those supported calibration paths now also have a checked replay/tolerance
pack. In practice that means the workflow-level solver provenance, replay
artifacts, fit-quality tolerances, and cold-versus-warm benchmark baselines are
locked down for the supported Hull-White, SABR, Heston, local-vol, and
single-name credit fixtures. Warm-start baselines remain explicit for the
three workflows that expose warm-seed hooks (Hull-White, SABR, Heston).
Those checked baseline fixtures now come from the same bounded mock-snapshot
contracts the rest of the proving path uses: rates, Heston, and local-vol
canaries read the seeded ``synthetic_generation_contract`` surface, while the
single-name credit canary reads the derived
``model_consistency_contract`` compatibility payload.
If you change solver wiring or runtime consumers, review the calibration
benchmark artifact in ``docs/benchmarks/calibration_workflows.md`` alongside the
workflow tests before treating the change as desk-safe.

Boundary hardening now also includes targeted negative canaries for:

- missing calibration binding (for example no discount curve for credit fit)
- unsupported quote-map families
- invalid calibrated-object materialization kinds
- rates multi-curve binding drift

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

Route matching is also now family-first. When the semantic compiler has
already emitted an explicit ``route_families`` contract, Trellis uses that
family identity ahead of broader engine-family preferences. Generic engine
families still matter for fallback scoring, but they no longer override a more
specific family decision coming from the semantic layer.

When the compiler does find such a checked backend, Trellis now records it as a
route-binding authority packet. That packet is not a second planning language;
it now separates a thin compatibility route alias from the nested backend
binding facts. The outer packet keeps the exact route ID, validation bundle,
and canary ownership; the nested ``backend_binding`` record carries the helper
refs, primitive refs, approved modules, admissibility contract, and stable
binding id that the runtime actually reuses. Build and review prompts consume
that packet only after the lane obligations so route guidance stays a
backend-fit constraint rather than a route-first synthesis plan.

For the migrated FX and quanto exact-helper lanes, that backend-fit packet is
now intentionally helper-only. Prompt and trace surfaces expose the semantic-
facing helper binding such as
``price_fx_vanilla_analytical(market_state, spec)`` or
``price_quanto_option_analytical_from_market_state(market_state, spec)``
instead of surfacing raw kernels or route-local input-mapping instructions as
live build authority.

The trace boundary also exposes a family-first ``construction_identity``
summary. For operators, that is now the primary readout:

- native checked fits surface the stable backend binding id
- otherwise Trellis surfaces the lowered family IR type or lane family
- the route alias is still present, but only as secondary provenance

For fully migrated exact-helper families, Trellis can now mark that alias as
internal-only. In that case, prompts and trace summaries omit the alias
entirely and rely on the backend binding id instead, while raw validation and
replay metadata still retain the route id for compatibility.

That family-first rule now also applies inside transform pricing. The thin
vanilla transform helper is only selected for ``equity_diffusion`` claims. If
the same European payoff is being priced under a different model family, such
as Heston stochastic volatility, Trellis keeps the request on the raw FFT/COS
kernel surface instead of pretending the vanilla helper is a universal
transform authority.

Under the hood, that route family now lowers onto a native
``TransformPricingIR`` contract before admissibility. Operators will therefore
see transform-specific facts in the trace summary, such as:

- the characteristic-function family
- the transform backend capability
- the terminal payoff kind
- quote and strike semantics

rather than a fallback view of the broader upstream option family.

For the migrated PDE routes, the same trace boundary now also exposes a compact
`family_ir_summary` alongside the raw lowering metadata. That summary is the
operator-facing view of the PDE contract: state variable, operator family,
control style, event transforms, event dates, and whether the route is still
passing through the transitional `VanillaEquityPDEIR` wrapper.

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

Calibration Provenance
----------------------

Rates volatility calibration and SABR smile fitting now expose governed solver
artifacts alongside the calibrated result. The raw typed ``solve_request`` and
``solve_result`` are still present, but callers should treat
``solver_provenance`` and ``solver_replay_artifact`` as the stable review
surface.

In practice that means downstream tools can inspect:

- which solve backend ran
- which explicit solver options were requested
- whether the solve succeeded and why it terminated
- the residual diagnostics associated with the solved point

without knowing whether the underlying calibration came from a scalar rates
root solve or a vector SABR least-squares fit.

The supported Hull-White workflow uses that same governed surface, but its
output is reusable by later tree helpers as well as inspectable by review
tools. ``calibrate_hull_white(...)`` returns the solver artifacts, the repriced
strip residuals, and a stable ``model_parameters`` payload; calling
``result.apply_to_market_state(...)`` makes those calibrated parameters the
runtime default for callable-bond, Bermudan-swaption, and ZCB-option helpers.

SABR smile fitting now exposes an analogous review surface. Callers that need a
reusable calibration artifact can build a ``SABRSmileSurface`` first and then
run ``fit_sabr_smile_surface(...)`` to inspect:

- the normalized smile-grid input with stable labels and optional weights
- explicit warnings when the input smile does not bracket the forward or lacks
  an exact ATM point
- per-strike fit residuals plus RMS / weighted RMS diagnostics

The older ``calibrate_sabr(...)`` helper remains available as a compatibility
wrapper, but it now routes through that same smile-surface assembly and
diagnostic layer.

For new code, the supported entry point is
``calibrate_sabr_smile_workflow(...)``. It accepts the raw smile inputs and
returns the full ``SABRSmileCalibrationResult`` directly, so callers do not
need to assemble the smile surface and fit stages by hand unless they
explicitly want that intermediate reusable artifact.

Typed Governed Parse And Match
------------------------------

The governed MCP/runtime path now treats the semantic-contract stack as the
canonical trade-identity boundary.

That means:

- ``trellis.platform.services.TradeService`` normalizes natural-language or
  structured requests into a typed semantic contract plus ``ProductIR``
- schedule-bearing structured requests can use the same alias family the
  semantic contract layer already recognizes, including
  ``observation_schedule`` / ``observation_dates``,
  ``fixing_schedule`` / ``fixing_dates``, ``exercise_schedule`` /
  ``exercise_date``, ``call_schedule`` / ``call_dates``, and
  ``expiry_date`` / ``expiry``; the governed parser normalizes those onto one
  canonical schedule surface before model match
- the first desk-oriented structured-rates trade-entry slice now includes
  ``range_accrual`` requests with explicit ``reference_index``,
  ``coupon_definition``, ``range_condition``, and ``observation_schedule``
  fields; the canonical settlement profile is filled in automatically and
  optional callability hooks are preserved as trade-entry metadata
- that same governed structured surface now also accepts ``callable_bond`` and
  ``bermudan_swaption`` requests with typed schedule-bearing term fields such
  as notional, coupon or strike, start/end or swap-end dates, and explicit
  exercise schedules; route defaults such as par call price, coupon frequency,
  and day-count basis are surfaced later as reviewable assumptions instead of
  being silently hidden
- incomplete requests surface explicit ``missing_fields`` and ``warnings``
  instead of silently dropping back into an opaque parse step
- the same contract family now has a generic imported-position wrapper:
  ``TradeService.parse_position(...)`` normalizes one flat or nested book row
  onto ``PositionImportContract`` with stable ``position_id``,
  ``instrument_type``, ``quantity``, ``structured_trade``, and ``field_map``
  fields before delegating to the same structured trade parser
- that same ingestion boundary now accepts mixed supported desk books from
  flat files: ``TradeService.load_positions(...)`` loads normalized row lists,
  while ``TradeService.load_positions_csv(...)`` and
  ``TradeService.load_positions_json(...)`` accept CSV and JSON inputs and
  return ``ImportedBookLoadResult`` with a stable parsed-position book,
  per-row ``row_results``, and deterministic ``parsed`` / ``partial`` /
  ``incomplete`` / ``invalid`` load status for desk review
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
- it can reuse an activated imported market snapshot from
  ``trellis.snapshot.import_files`` when the desk is working from explicit
  local market-data files
- it persists a canonical run record and canonical audit bundle for every
  successful or blocked call
- it does not silently fall back to candidate generation when no approved model
  is execution-eligible

The response can be projected in three modes:

- ``concise`` returns status, result summary, warnings, and a minimal audit
  pointer
- ``structured`` returns run id, structured result payload, warnings,
  provenance, audit URI, and a trader-facing ``desk_review`` bundle
- ``audit`` returns the structured payload plus the canonical audit bundle

The ``desk_review`` bundle is the desk-readable projection layered on top of
the canonical run record. It summarizes:

- trade and route identity (semantic id, model, method, adapter, engine)
- assumption categories for explicit, defaulted, synthetic/proxy, and missing
  inputs
- trader-facing warning items with stable categories
- deterministic ``driver_narrative`` text linked back to the selected route,
  assumptions, and warnings
- deterministic ``scenario_commentary`` text linked back to the returned
  scenario ladder, assumptions, and warnings
- schedule/event highlights such as observation and call dates
- the scenario ladder returned by the checked pricing route
- stable run, audit, and market-snapshot references

Blocked responses are part of the contract. Missing provider bindings,
non-approved model matches, policy denial, and unsupported MVP execution
adapters return persisted blocked runs rather than opaque transport errors.
When that happens, ``desk_review`` is still returned, but its
``driver_narrative`` is blocker-specific and explicitly says the run stopped
before route selection or pricing execution rather than implying that an
approved pricing path succeeded.

The first desk-oriented checked structures on this surface are now:

- ``range_accrual_discounted`` for the first single-index range-accrual note slice
- ``callable_bond_tree`` for issuer-call fixed-income structures
- ``bermudan_swaption_tree`` for the first supported Bermudan rates option slice

When the approved model match selects the range-accrual adapter,
``trellis.price.trade`` expects explicit ``notional`` plus the normalized
range-accrual contract terms, resolves the discount curve, forecast-curve
proxy, and optional fixing history from the active market snapshot, and
returns:

- total PV plus separate coupon-leg and principal-leg PV
- a first parallel-shift risk measure as ``risk.parallel_curve_pv01``
- a small trader-style scenario ladder for parallel curve moves
- a deterministic validation bundle with fixing-coverage, reference-bound, and
  PV-reconciliation checks

If the active snapshot does not contain a dedicated forecast curve for the
reference index, the route falls back to the selected discount curve as the
projection proxy and surfaces that assumption in the response warnings and
validation bundle. The same note is also grouped into
``desk_review.assumptions.synthetic_inputs`` so the trader-facing output makes
the proxy explicit without forcing the user to inspect the raw validation
payload first.

For callable bonds and Bermudan swaptions, the governed surface now packages
the checked lattice helpers into the same run/audit contract. Callable bonds
also have a bounded Hull-White PDE proof route on the event-aware rollback
lane, so the compiler can now cross-check issuer-callable fixed-income requests
through either the tree or PDE path when the market bindings and model family
fit that slice. These routes surface typed exercise schedules in
``desk_review.schedule_summary``, project reviewable event rows for exercise or
maturity boundaries, and group defaulted route assumptions such as par call
price, schedule frequency, or day-count basis into
``desk_review.assumptions.defaulted_inputs``.

Under those public wrappers, the reusable short-rate fixed-income helper layer
now owns coupon schedule compilation, embedded exercise/control semantics,
straight-bond reference PV, and generic lattice/PDE event assembly. That keeps
the callable-bond user surface stable while letting later short-rate claim
families reuse the same helper substrate instead of copying callable-bond-local
 glue code.

Callable-bond trade runs now also expose ``result.oas_duration`` plus
``result.callable_scenario_explain``. The scenario explain payload is the
callable-specific desk review surface: it shows how the callable price, the
straight-bond reference price, and the embedded call option value move under a
standard parallel rate ladder.

The derived ``desk_review.driver_narrative`` and
``desk_review.scenario_commentary`` fields are intentionally not free-form LLM
output. They are deterministic text summaries built from the same stored route
result, assumption buckets, warning pack, and scenario ladder that power the
rest of the desk-review bundle, so the narrative surface stays traceable to the
underlying evidence.

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

First Exotic Desk Workflow
--------------------------

The first packaged explicit-input exotic workflow is the imported-snapshot desk
path for ``range_accrual``, ``callable_bond``, and ``bermudan_swaption``.

In-process Python or notebook example:

.. code-block:: python

   from trellis.mcp.server import bootstrap_mcp_server

   server = bootstrap_mcp_server(state_root="tmp/mcp_state")

   imported = server.call_tool(
       "trellis.snapshot.import_files",
       {
           "session_id": "desk_demo",
           "manifest_path": "/abs/path/range_accrual_snapshot.yaml",
           "activate_session": True,
           "reference_date": "2026-04-04",
       },
   )
   response = server.call_tool(
       "trellis.price.trade",
       {
           "session_id": "desk_demo",
           "structured_trade": {
               "instrument_type": "range_accrual",
               "reference_index": "SOFR",
               "coupon_rate": 0.0525,
               "lower_bound": 0.015,
               "upper_bound": 0.0325,
               "observation_schedule": (
                   "2026-01-15",
                   "2026-04-15",
                   "2026-07-15",
                   "2026-10-15",
               ),
               "notional": 1_000_000.0,
               "payout_currency": "USD",
               "reporting_currency": "USD",
               "preferred_method": "analytical",
           },
           "output_mode": "audit",
           "valuation_date": "2026-04-04",
       },
   )

   response["result"]["price"]
   response["desk_review"]["assumptions"]
   response["desk_review"]["scenario_summary"]
   response["desk_review"]["audit_refs"]["snapshot_uri"]

For MCP hosts, use the guided prompt ``exotic_desk_one_trade`` with the target
session id. That prompt walks the host through:

1. importing the explicit market snapshot
2. parsing and matching one supported desk trade
3. calling ``trellis.price.trade`` with ``output_mode="audit"``
4. reading ``desk_review`` first, then the persisted run/audit resources

The same prompt is exposed over the local streamable HTTP transport after the
host is attached to ``http://127.0.0.1:8000/mcp``. The goal is to keep the
workflow identical across in-process Python, MCP, and local HTTP hosts: import
the snapshot, price the supported trade, then review ``desk_review`` plus the
canonical run/audit resources instead of reconstructing workflow state by hand.

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
  reproducibility bundle snapshot that can be referenced later by run id,
  including a concrete governed market-snapshot contract that can be
  rehydrated for replay or review

The matching read surface is resource-based rather than write-tool based. The
main URIs are:

- ``trellis://models/{model_id}`` and version-sidecar URIs for contract, code,
  and validation reports
- ``trellis://runs/{run_id}``, ``.../audit``, ``.../inputs``, and ``.../outputs``
  for governed run inspection
- ``trellis://market-snapshots/{snapshot_id}`` for persisted market snapshots,
  including imported snapshots and reproducibility bundles with their embedded
  governed market data

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

``Pipeline.run()`` now returns a dict-like ``ScenarioResultCube`` rather than a
plain ``dict``. Existing code can still index it by scenario name, but the cube
also preserves ``scenario_specs`` plus per-scenario provenance and exposes
aggregation helpers for later desk-review workflows. Pipeline execution now
flows through a reusable compiled compute plan as well, so callers can inspect
``Pipeline.compile_compute_plan().to_dict()`` before running or recover the
serialized plan later from ``cube.compute_plan``. Named saved scenario
templates can be loaded from ``market_snapshot.metadata["scenario_templates"]``
by using ``{"scenario_template": "template_name"}`` inside ``scenarios(...)``,
and the executed cube can be projected into a stable pod-review payload with
``cube.to_batch_output()``. The same cube now also exposes
``cube.pnl_attribution()`` for scenario-by-scenario top-contributor ranking:

.. code-block:: python

   cube = (
       Pipeline()
       .instruments(book)
       .market_data(curve=curve)
       .scenarios(
           [
               {"name": "base", "shift_bps": 0},
               {"name": "up100", "shift_bps": 100},
           ]
       )
       .run()
   )

   cube["up100"].total_mv
   cube.scenario_specs["up100"]
   cube.compute_plan["scenario_count"]
   cube.book_ladder("total_mv").metadata["deltas"]["up100"]
   cube.position_ladder("mv")["10Y"]["up100"]
   cube.to_batch_output()["book_pnl"]["values"]["up100"]
   cube.pnl_attribution()["scenario_attribution"]["up100"]["top_contributors"][0]

The aggregation helpers preserve scenario provenance in their attached
``.metadata`` payload, so downstream explain code does not have to reconstruct
which scenario pack, shift template, or pipeline settings produced a ladder.
The published ``book_pnl`` and ``position_pnl`` payloads now put actual P&L
deltas in ``values`` and keep the underlying scenario levels under
``metadata["levels"]``.

The runtime analytics surface now also exposes spot ``delta`` and ``gamma``
plus roll-down ``theta`` through ``Session.analyze(...)``. Delta and gamma use
finite-difference repricing on one selected spot binding, while theta rolls the
market-state dates forward and reports ``V(t + dt) - V(t)`` for the requested
calendar-day step:

.. code-block:: python

   result = s.analyze(
       payoff,
       measures=[
           {"delta": {"bump_pct": 1.0}},
           {"gamma": {"bump_pct": 1.0}},
           {"theta": {"day_step": 1}},
       ],
   )

Delta and gamma require a usable spot binding. Trellis accepts either
``market_state.spot`` or one selected underlier spot from the backing market
snapshot. If no unambiguous spot binding exists, the runtime now raises an
explicit error instead of silently omitting the measure.

When you request ``key_rate_durations`` through ``Session.price(...)``,
``Session.greeks(...)``, or ``Session.risk_report(...)``, Trellis now uses the
same interpolation-aware bucket engine as ``Session.analyze(...)`` and returns
numeric tenor keys on the active curve grid, for example ``{1.0: ..., 5.0:
...}`` instead of legacy string labels such as ``"KRD_5.0y"``.

Those risk surfaces are now dict-like objects with attached methodology
metadata. In practice that means you can inspect
``result.greeks["key_rate_durations"].metadata`` or
``result.scenario_pnl.metadata`` to see whether Trellis used direct
zero-curve bucket shocks or a rebuild-based quote-space workflow, which bucket
grid it used, and whether any fallback occurred.

For interpolation-aware custom KRD buckets, use ``Session.analyze(...)`` with
an explicit ``key_rate_durations`` measure configuration. The requested tenor
grid becomes the piecewise-linear risk grid, so off-grid buckets such as ``7Y``
no longer collapse to zero just because the base curve only carries ``5Y`` and
``10Y`` knots:

.. code-block:: python

   from trellis.core.payoff import DeterministicCashflowPayoff

   krd = s.analyze(
       DeterministicCashflowPayoff(bond),
       measures=[
           {
               "key_rate_durations": {
                   "tenors": (2.0, 5.0, 7.0, 10.0),
                   "bump_bps": 25.0,
               }
           }
       ],
   ).key_rate_durations

If the active discount curve came from a bootstrapped snapshot with recorded
``bootstrap_runs`` provenance, ``Session.analyze(...)`` can also request a
quote-space rebuild workflow:

.. code-block:: python

   rebuild_krd = s.analyze(
       DeterministicCashflowPayoff(bond),
       measures=[
           {
               "key_rate_durations": {
                   "methodology": "curve_rebuild",
                   "bump_bps": 25.0,
               }
           }
       ],
   ).key_rate_durations

   assert rebuild_krd.metadata["resolved_methodology"] == "curve_rebuild"

The same analytics surface now supports named twist and butterfly rate
scenarios through ``scenario_pnl``:

.. code-block:: python

   scenario_pnl = s.analyze(
       DeterministicCashflowPayoff(bond),
       measures=[
           {
               "scenario_pnl": {
                   "scenario_packs": ("twist", "butterfly"),
                   "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
                   "pack_amplitude_bps": 25.0,
               }
           }
       ],
   ).scenario_pnl

When you request scenario packs without an explicit parallel-shift ladder,
Trellis now treats that as a pack-only request. Pass
``"include_parallel_shifts": True`` if you want both the named scenarios and
the parallel ``shifts_bps`` ladder in the same result.

Bootstrap-backed sessions can also run the named packs through the rebuild
workflow so the shocks follow the quoted curve instruments instead of only the
final zero curve:

.. code-block:: python

   rebuild_scenarios = s.analyze(
       DeterministicCashflowPayoff(bond),
       measures=[
           {
               "scenario_pnl": {
                   "methodology": "curve_rebuild",
                   "scenario_packs": ("twist", "butterfly"),
                   "bucket_tenors": (1.0, 2.0, 5.0, 10.0),
               }
           }
       ],
   ).scenario_pnl

Each named entry reports P&L relative to the base valuation while keeping the
bucket assumptions explicit in the scenario name, for example
``"twist_steepener_25bp"`` or ``"butterfly_belly_up_25bp"``. The attached
``.metadata`` payload records the resolved methodology, bucket conventions, and
any fallback reason when a rebuild request had to drop back to zero-curve
shocks. The saved ``scenario_templates`` emitted by that metadata now preserve
the concrete methodology as well, so replaying a rebuild-based saved template
through ``Pipeline.scenarios([{"scenario_template": ...}])`` stays in
quote-space instead of silently degrading to a zero-curve bump.

Bucketed vega now uses the same explicit substrate idea on the volatility
side. Request ``vega`` with both ``expiries`` and ``strikes`` to get a
dict-like expiry/strike surface instead of one coarse scalar:

.. code-block:: python

   vega_surface = s.analyze(
       payoff,
       measures=[
           {
               "vega": {
                   "expiries": (1.0, 1.5, 2.0),
                   "strikes": (90.0, 100.0, 110.0),
                   "bump_pct": 1.0,
               }
           }
       ],
   ).vega

   assert vega_surface[1.5][100.0] != 0.0
   assert vega_surface.metadata["bucket_convention"] == "expiry_strike"

The attached ``.metadata`` payload records the configured bucket grid, the
resolved surface type, and any warnings. Two warning codes are especially
useful in practice:

- ``interpolated_surface_bucket`` means the requested bucket does not land on
  an observed surface node and was synthesized from the surrounding grid
- ``flat_surface_expanded`` means the runtime started from ``FlatVol`` and had
  to expand it onto the requested bucket grid before shocking it

When you omit ``expiries`` and ``strikes``, ``vega`` still returns the older
coarse scalar sensitivity.

The supported pod-risk workflows now also have a checked throughput baseline in
``docs/benchmarks/pod_risk_workflows.md``. That report covers the shared
scenario-result cube path, rebuild-based rates-risk workflows, bucketed vega,
and the spot-risk measure bundle through the same public/runtime entrypoints
shown above.

Callable-bond analytics now have two dedicated runtime measures as well:

- ``oas_duration`` reports effective duration on the callable tree, with an
  optional market-price anchor if you want the current OAS solved first
- ``callable_scenario_explain`` returns a callable-specific rate-shock ladder
  showing callable price, straight-bond reference price, and the implied call
  option value under each scenario

These callable measures are intentionally explicit about scope. Today they are
implemented for callable-bond payoffs with a real call schedule; non-callable
payoffs fail with a direct error instead of silently returning an empty field.

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

If an analytical trace stores that binding only under
``runtime_contract.snapshot_reference.selected_curve_names`` (instead of a
top-level ``selected_curve_names`` context field), replay summaries still read
the same names back. The replay/debug path therefore uses one curve-role
contract even when trace producers emit slightly different context layouts.

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
