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
