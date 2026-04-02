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

FX Vanilla Options
------------------

FX vanillas are priced by resolving spot FX, domestic discounting, foreign
discounting, and Black volatility, then assembling the terminal payoff from
Black76 basis claims. Trellis keeps that decomposition explicit in the
analytical route, so the same route can be reused for pricing and Greeks.

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

   MissingCapabilityError: Missing market data: ['black_vol']
     Missing market data: 'black_vol' — Black (lognormal) implied volatility surface.
       How to provide: Session(vol_surface=FlatVol(0.20))
