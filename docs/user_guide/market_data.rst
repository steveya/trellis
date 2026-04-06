Market Data
===========

Canonical Capability Names
--------------------------

When a payoff, plan, or runtime contract declares required market data, use
the canonical capability names rather than older aliases:

- ``discount_curve``
- ``forward_curve``
- ``fixing_history``
- ``black_vol_surface``
- ``credit_curve``
- ``fx_rates``
- ``spot``
- ``local_vol_surface``
- ``jump_parameters``
- ``model_parameters``

Yield Curves
------------

.. code-block:: python

   from trellis import YieldCurve

   # Flat curve
   curve = YieldCurve.flat(0.045)

   # From Treasury yields (BEY → continuous)
   curve = YieldCurve.from_treasury_yields({
       0.25: 0.045, 1.0: 0.047, 2.0: 0.048,
       5.0: 0.045, 10.0: 0.044, 30.0: 0.046,
   })

   # From bootstrap with explicit curve conventions
   from trellis import (
       BootstrapConventionBundle,
       BootstrapCurveInputBundle,
       BootstrapInstrument,
       bootstrap_yield_curve,
   )
   from trellis.curves.bootstrap import bootstrap_curve_result
   from trellis.core.types import Frequency

   bundle = BootstrapCurveInputBundle(
       curve_name="usd_ois_boot",
       currency="USD",
       rate_index="USD-SOFR-3M",
       conventions=BootstrapConventionBundle(
           swap_fixed_frequency=Frequency.ANNUAL,
           swap_float_frequency=Frequency.QUARTERLY,
       ),
       instruments=(
           BootstrapInstrument(0.25, 0.04, "deposit", label="DEP3M"),
           BootstrapInstrument(2.0, 0.045, "swap", label="SWAP2Y"),
           BootstrapInstrument(5.0, 0.048, "swap", label="SWAP5Y"),
       ),
   )
   curve = bootstrap_yield_curve(bundle)
   result = bootstrap_curve_result(bundle)

   assert result.solver_provenance.backend["backend_id"] == "scipy"
   assert result.diagnostics.max_abs_residual < 1e-8

Legacy lists of ``BootstrapInstrument`` still work, but Trellis now
normalizes them onto the same typed bundle surface with explicit default
conventions.

Data Providers
--------------

.. code-block:: python

   # Mock (offline, no API key)
   from trellis.data.mock import MockDataProvider
   yields = MockDataProvider().fetch_yields()

   # FRED (requires FRED_API_KEY)
   from trellis.data.fred import FredDataProvider
   yields = FredDataProvider().fetch_yields()

   # Treasury.gov (no key needed, but requires internet)
   from trellis.data.treasury_gov import TreasuryGovDataProvider
   yields = TreasuryGovDataProvider().fetch_yields()

Mock Data
~~~~~~~~~

The mock provider ships 4 historical snapshots (no network needed):

- 2019-09-15: Pre-COVID normal curve (~1.6-2.1%)
- 2020-03-15: COVID crisis, near-zero front end
- 2023-10-15: Peak rates, inverted curve (~4.5-5.3%)
- 2024-11-15: Easing cycle (~4.2-4.6%)

Resolved market snapshots keep a ``provenance`` payload alongside the market
objects so replay and calibration traces can tell whether a curve came from a
direct quote, a resolver merge, or a bootstrap input bundle.

When a snapshot uses ``discount_curve_bootstraps`` or
``forecast_curve_bootstraps``, the provenance now records both the full
bootstrap bundle and the executed bootstrap solve artifacts for each named
curve.

The ``bootstrap_inputs`` section keeps the market-input contract:

- curve identity such as ``curve_name``, ``currency``, and ``rate_index``
- the explicit ``BootstrapConventionBundle`` used for deposits, futures, and swaps
- the ordered market-instrument surface with instrument labels and tenor inputs

The sibling ``bootstrap_runs`` section keeps the realized solve path:

- the typed ``solve_request`` and normalized ``solve_result``
- governed ``solver_provenance`` and replay artifacts
- bootstrap diagnostics including residuals and Jacobian metadata

This replaces the old flat list of anonymous bootstrap instruments and gives
later calibration, replay, and validation workflows a stable curve-input plus
solve-output contract.

That same provenance now also powers the rebuild-based rates-risk workflows.
When ``Session.analyze(...)`` requests ``methodology="curve_rebuild"`` for
``key_rate_durations`` or ``scenario_pnl``, Trellis reconstructs the quoted
bootstrap bundle from this stored payload, bumps the relevant market quotes,
rebuilds the curve, and then reprices on the rebuilt discount surface.

Trellis also exposes a reusable expiry/strike bucket substrate for implied
volatility surfaces. Use ``build_vol_surface_shock_surface(...)`` to
re-express a supported surface on a requested bucket grid and then apply
bucket bumps in volatility basis points:

.. code-block:: python

   from trellis.models import GridVolSurface, build_vol_surface_shock_surface

   surface = GridVolSurface(
       expiries=(1.0, 2.0),
       strikes=(90.0, 110.0),
       vols=((0.25, 0.22), (0.27, 0.24)),
   )

   shock_surface = build_vol_surface_shock_surface(
       surface,
       expiries=(1.0, 1.5, 2.0),
       strikes=(90.0, 100.0, 110.0),
   )
   bucketed = shock_surface.bucketed_surface()
   bumped = shock_surface.apply_bumps({(1.5, 100.0): 25.0})

Each bucket records whether it matches an exact surface node plus the expiry
and strike support brackets used to anchor interpolation. The helper also
surfaces explicit warnings for approximate inputs. Today ``GridVolSurface`` and
``FlatVol`` are supported; when a flat surface is expanded onto the bucket
grid, Trellis emits the warning code ``flat_surface_expanded`` so later vega
and scenario routes can disclose that the bucket surface was synthesized.

That same market-data surface now carries reusable model parameter payloads as
well. Calibration workflows can attach a supported parameter set to
``MarketState.model_parameters`` and ``MarketState.model_parameter_sets`` so
later pricing helpers consume the calibrated model inputs instead of relying on
route-local constants.

When parameter packs come from market-data resolution instead of calibration
workflows, use ``resolve_market_snapshot(..., model_parameter_sources=...)``
to declare the source kind explicitly. The resolver currently supports two
branches:

- ``direct_quote`` for quoted/provider parameter packs
- ``bootstrap`` for deterministic curve-derived parameter packs

Each resolved pack is recorded in ``snapshot.provenance["market_parameter_sources"]``.
Bootstrap-derived packs also persist their entry contract in
``snapshot.provenance["bootstrap_inputs"]["model_parameters"]`` so replay and
tracing can reconstruct how each parameter was derived.

.. code-block:: python

   from datetime import date

   from trellis.data.resolver import resolve_market_snapshot

   snapshot = resolve_market_snapshot(
       as_of=date(2024, 11, 15),
       source="treasury_gov",
       model_parameter_sources={
           "quanto_direct": {
               "source_kind": "direct_quote",
               "source_ref": "market_feed.quanto",
               "parameters": {"quanto_correlation": 0.35, "vol_fx": 0.12},
           },
           "curve_bootstrap_pack": {
               "source_kind": "bootstrap",
               "source_ref": "rates.curve_samples",
               "bootstrap_inputs": {
                   "entries": [
                       {
                           "parameter": "zero_1y",
                           "curve_family": "discount_curves",
                           "curve_name": "discount",
                           "measure": "zero_rate",
                           "tenor": 1.0,
                       },
                   ]
               },
           },
       },
       default_model_parameters="curve_bootstrap_pack",
   )

   assert snapshot.provenance["market_parameter_sources"]["quanto_direct"]["source_kind"] == "direct_quote"
   assert snapshot.provenance["market_parameter_sources"]["curve_bootstrap_pack"]["source_kind"] == "bootstrap"

Unsupported source kinds or mixed direct/bootstrap payloads fail closed with a
``ValueError``.

For example, the supported Hull-White strip workflow can calibrate one
parameter set and project it back onto the runtime state:

.. code-block:: python

   calibrated = calibrate_hull_white(instruments, market_state)
   market_state = calibrated.apply_to_market_state(market_state)

   assert market_state.model_parameters["model_family"] == "hull_white"
   assert market_state.model_parameter_sets["hull_white"]["sigma"] > 0.0

The same ``model_parameters`` surface can carry runtime Heston parameters for
later pricing or simulation workflows:

.. code-block:: python

   from dataclasses import replace

   from trellis.models.processes.heston import (
       build_heston_parameter_payload,
       resolve_heston_runtime_binding,
   )

   market_state = replace(
       market_state,
       model_parameters=build_heston_parameter_payload(
           kappa=2.0,
           theta=0.04,
           xi=0.3,
           rho=-0.7,
           v0=0.04,
       ),
   )
   binding = resolve_heston_runtime_binding(market_state)

   assert binding.process.state_dim == 2
   assert binding.model_parameters["model_family"] == "heston"

The supported Heston calibration workflow can now write that same payload back
onto runtime state directly:

.. code-block:: python

   from trellis.models.calibration import calibrate_heston_smile_workflow

   calibrated = calibrate_heston_smile_workflow(
       100.0,
       1.0,
       [80.0, 90.0, 100.0, 110.0, 120.0],
       [0.24, 0.22, 0.20, 0.19, 0.18],
       rate=0.02,
       parameter_set_name="heston_equity",
   )
   market_state = calibrated.apply_to_market_state(market_state)

   assert market_state.model_parameter_sets["heston_equity"]["model_family"] == "heston"

Local-vol workflows use the same canonical runtime handoff:

.. code-block:: python

   from trellis.models.calibration import calibrate_local_vol_surface_workflow

   local_vol = calibrate_local_vol_surface_workflow(
       strikes=[80.0, 90.0, 100.0, 110.0],
       expiries=[0.25, 0.5, 1.0, 2.0],
       implied_vols=[
           [0.22, 0.21, 0.20, 0.19],
           [0.23, 0.21, 0.20, 0.19],
           [0.24, 0.22, 0.21, 0.20],
           [0.25, 0.23, 0.22, 0.21],
       ],
       S0=100.0,
       r=0.02,
       surface_name="equity_local_vol",
   )
   market_state = local_vol.apply_to_market_state(market_state)

   assert "equity_local_vol" in market_state.local_vol_surfaces

Mock snapshots also expose the synthetic-prior family, seed, and parameter set
used to build the embedded regime bundle, so proving and replay runs can show
exactly which prior was sampled.

Mock snapshots now expose a seeded
``synthetic_generation_contract`` under
``snapshot.provenance["prior_parameters"]``. This is the generator-side
authority contract for the bounded synthetic path, and it separates:

- seeded model packs
- synthetic quote bundles
- runtime target names

The older compatibility
``model_consistency_contract`` is now derived from that seeded contract so the
existing replay and benchmark fixtures keep working while the family-specific
synthetic generators migrate.

For rates specifically, the synthetic generator now carries a richer bounded
rates model pack rather than only parallel shifts:

- tenor-shaped discount-curve shift parameters for the non-USD discount curves
- tenor-shaped forecast-basis parameters for named forecast curves
- a SABR-style rate-vol model used to generate the synthetic rate-vol surface

This keeps the rates bundle deterministic and cheap, but makes it materially
more consistent for proving, calibration fixtures, and multi-curve demos.

For credit, the synthetic generator is now hazard-first:

- the credit model pack stores seeded hazard-rate knots
- the CDS-style spread grid in the quote bundle is derived from those hazards
- the runtime ``CreditCurve`` in the snapshot is built from the same hazard
  authority surface

This keeps the synthetic credit path aligned with the typed reduced-form credit
workflow instead of letting quote space and runtime space drift apart.

For the migrated calibration workflows, the derived
``model_consistency_contract`` still records the bounded deterministic rates,
credit, and volatility assumptions used to build the synthetic snapshot:

- rates curve roles and forecast-basis inputs
- reduced-form credit spread grids and recovery
- the named vol/local-vol/model-parameter packs included in the snapshot

This is synthetic provenance for proving, demos, and regression fixtures. It
is intentionally explicit so downstream tooling can show which bounded model
assumptions were used, and it should not be treated as live market data.

The calibration benchmark/replay pack now consumes the same contract for the
single-name credit fixture, so synthetic spread/recovery assumptions used by
mock snapshots line up with the checked calibration boundary in
``docs/benchmarks/calibration_workflows.{json,md}``.

For schedule-bound rates workflows, the mock provider also ships deterministic
recent fixing histories for the named rate indices in the snapshot. Those
histories are first-class snapshot components, not opaque metadata blobs:
``MarketSnapshot.fixing_histories`` holds the named history family,
``default_fixing_history`` selects the default history when one exists, and
``MarketState.fixing_histories`` carries the same surface into runtime pricing.

File-Based Snapshot Import
--------------------------

The exotic-desk workflow can now import an explicit market snapshot from a
local YAML or JSON manifest and persist it as a governed snapshot record.

The manifest supports named families for:

- ``discount_curves``
- ``forecast_curves``
- ``vol_surfaces``
- ``credit_curves``
- ``fx_rates``
- ``underlier_spots``
- ``fixing_histories``

Curve, surface, FX, and credit components can be supplied inline or by
referencing component files relative to the manifest. Fixing histories can be
loaded from structured YAML/JSON payloads or simple ``date,value`` CSV files.

The persisted snapshot keeps:

- the resolved component contract
- source-file paths for audit and replay
- stable named-component summaries
- typed named fixing histories plus the default fixing-history selection when
  provided
- warnings for missing expected families, synthetic inputs, defaulted
  selections, and stale ``as_of`` dates

The MCP/HTTP surface exposes this through ``trellis.snapshot.import_files``.
When called with ``activate_session=true``, the imported snapshot becomes the
active market-data source for that governed session and later
``trellis.price.trade`` calls reuse the persisted snapshot instead of resolving
an external provider. Passing a later ``valuation_date`` to
``trellis.price.trade`` changes the runtime settlement date while keeping the
imported market data fixed at the snapshot manifest ``as_of`` date.

If the manifest includes fixing histories, Trellis normalizes them into the
canonical typed market-data surface:

- ``MarketSnapshot.fixing_history(name=None)`` resolves the selected or named
  history as a ``date -> fixing`` mapping
- ``MarketSnapshot.to_market_state(..., fixing_history=\"SOFR\")`` carries the
  selected fixing history into runtime state and records the selected name in
  ``MarketState.selected_curve_names``
- ``MarketState.available_capabilities`` includes ``fixing_history`` whenever
  the runtime state carries any historical fixings

Named component requests now sit on top of that same snapshot surface.
``MarketSnapshot.resolve_request(...)`` accepts a request-time selection such
as ``discount_curve``, ``forecast_curve``, ``vol_surface``, or
``fixing_history`` plus optional named ``scenario_templates`` and returns a
stable ``SnapshotSelectionResult`` with:

- the requested component names
- the resolved runtime ``selected_curve_names``
- the available named components on the snapshot
- resolved scenario-template specs from ``snapshot.metadata``
- explicit warnings when a requested component/template is missing or the
  snapshot is stale for the requested reference date

For persisted imported snapshots, ``SnapshotService.resolve_market_state(...)``
wraps that same request-driven selection flow and merges the import-manifest
warning surface so later book workflows can reuse named components without
hand-assembling a ``MarketState`` first.

Correlation Sources
-------------------

Basket and quanto routes do not treat correlation as an unstated constant.
The resolved source can be:

- an explicit matrix or scalar
- an empirical estimate from observed path data
- an implied source tied to liquid market inputs
- a synthetic prior for mock, stress, or proving runs

The trace keeps the source kind, sample size, estimator, seed, and any
regularization that was needed before pricing.

Multi-Curve
-----------

Post-2008, use separate discount and forecast curves:

.. code-block:: python

   s = Session(
       curve=ois_curve,                          # OIS for discounting
       forecast_curves={"USD-SOFR-3M": sofr_curve},  # SOFR for forwards
   )

Credit Curves
-------------

.. code-block:: python

   from trellis import CreditCurve

   cc = CreditCurve.flat(0.02)  # 200bp hazard rate
   cc = CreditCurve.from_spreads({5.0: 0.012}, recovery=0.4)
