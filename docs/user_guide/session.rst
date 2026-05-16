Session & Pipeline
==================

Session
-------

A ``Session`` is an immutable market snapshot for pricing. It holds a yield curve,
optional vol surface, credit curve, and other market data.

.. code-block:: python

   from trellis import Session, YieldCurve, FlatVol
   from datetime import date

   s = Session(
       curve=YieldCurve.flat(0.05),
       settlement=date(2024, 11, 15),
       vol_surface=FlatVol(0.20),
   )

**Auto-resolution**: if no curve is provided, the session fetches from a data provider:

.. code-block:: python

   s = Session(data_source="mock")  # uses built-in mock data

Scenario Analysis
~~~~~~~~~~~~~~~~~

Sessions are immutable — scenario methods return new sessions:

.. code-block:: python

   s_up = s.with_curve_shift(+100)       # +100bp parallel shift
   s_bumped = s.with_tenor_bumps({10.0: +50})  # +50bp at 10Y
   s_off_grid = s.with_tenor_bumps({7.0: +25})  # inserts a local 7Y shock node
   s_new = s.with_curve(other_curve)     # replace curve entirely
   s_vol = s.with_vol_surface(FlatVol(0.30))

``with_tenor_bumps(...)`` now routes through the shared curve-shock substrate.
Exact tenor requests still move the matching knot directly, while off-grid
tenors insert a bumped node into the shocked curve so repricing follows the
same interpolation logic as the base curve. The resulting curve keeps optional
``curve_shock_surface`` and ``curve_shock_warnings`` metadata for callers that
need to inspect sparse-support or endpoint-extension warnings.

Pipeline
--------

For batch pricing with scenarios:

.. code-block:: python

   from trellis import Pipeline

   results = (
       Pipeline()
       .instruments(book)
       .market_data(curve=curve)
       .scenarios([
           {"name": "base", "shift_bps": 0},
           {"name": "up100", "shift_bps": 100},
       ])
       .output_csv("output/{scenario}.csv")
       .run()
   )

Named rate scenario packs are also available for desk-style curve stresses:

.. code-block:: python

   results = (
       Pipeline()
       .instruments(book)
       .market_data(curve=curve)
       .scenarios([
           {
               "scenario_pack": "twist",
               "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
               "amplitude_bps": 25.0,
           }
       ])
       .run()
   )

The current built-in packs are ``"twist"`` (steepener + flattener) and
``"butterfly"`` (belly-up + belly-down). Each pack expands into named
``tenor_bumps`` scenarios on the shared bucket-shock substrate, so the same
assumptions flow through both pipeline and one-trade analytics surfaces.

Book & BookResult
-----------------

A ``Book`` holds a collection of instruments with notionals:

.. code-block:: python

   from trellis import Book, Bond
   book = Book(
       {"10Y": bond_10y, "5Y": bond_5y},
       notionals={"10Y": 25_000_000, "5Y": 10_000_000},
   )
   result = s.price(book)
   print(f"Total MV: {result.total_mv:,.0f}")
   print(f"Book DV01: {result.book_dv01:,.0f}")

``Session.risk_report(book)`` also includes a bounded ``portfolio_aad`` payload
for supported bond books on a shared ``YieldCurve``. The legacy report values
remain tenor-keyed, while ``portfolio_aad["metadata"]`` carries
``risk_factor_coordinates``, ``sparse_risk_vector``, and a serialized
``portfolio_aad_result`` keyed by canonical risk-factor IDs. Unsupported
positions are listed explicitly and are excluded from AAD risk rather than
silently bumped.

The same factorized portfolio-AAD result contract is available directly for
bounded vanilla equity option books on one shared flat volatility surface or
one shared grid volatility surface:

.. code-block:: python

   from dataclasses import dataclass
   from datetime import date

   from trellis.book import Book, portfolio_aad_equity_option_vol_risk
   from trellis.core.market_state import MarketState
   from trellis.curves.yield_curve import YieldCurve
   from trellis.models.vol_surface import FlatVol

   @dataclass(frozen=True)
   class VanillaOption:
       spot: float
       strike: float
       expiry_date: date
       option_type: str = "call"
       notional: float = 1.0
       exercise_style: str = "european"

   market = MarketState(
       as_of=date(2024, 11, 15),
       settlement=date(2024, 11, 15),
       discount=YieldCurve.flat(0.05),
       vol_surface=FlatVol(0.20),
   )
   book = Book(
       {"call": VanillaOption(100.0, 100.0, date(2025, 11, 15))},
       notionals={"call": 1_000.0},
   )
   aad = portfolio_aad_equity_option_vol_risk(
       book,
       market,
       vol_surface_name="spx_flat",
       currency="USD",
   )
   print(aad.risk_vector.to_payload())

This option lane is intentionally narrower than ``Session.risk_report(...)``:
it supports European call/put specs on one shared ``FlatVol`` or
``GridVolSurface`` plus bounded smooth-interior American/Bermudan specs over
``FlatVol``, returns a typed ``PortfolioAADResult`` directly, and reports
unsupported positions as unsupported instead of inserting a finite-difference
fallback.

Arithmetic-average Asian options use a separate path-summary lane:

.. code-block:: python

   from trellis.book import portfolio_aad_arithmetic_asian_vol_risk

   aad = portfolio_aad_arithmetic_asian_vol_risk(
       book,
       market,
       vol_surface_name="spx_flat",
       currency="USD",
   )

That lane supports bounded European arithmetic-average call/put specs over a
shared ``FlatVol`` and reports barrier, knock, grid-vol path, and other
discontinuous or unsupported path features as fail-closed metadata.

Single-name quanto correlation risk has its own bounded hybrid lane:

.. code-block:: python

   from trellis.analytics import QuantoCorrelationAADMarketContext
   from trellis.book import portfolio_aad_quanto_correlation_risk

   context = QuantoCorrelationAADMarketContext(
       resolved_inputs=resolved_quanto_inputs,
       correlation_name="sx5e_eurusd",
       factor_a="SX5E",
       factor_b="EURUSD",
       currency="EUR",
   )
   aad = portfolio_aad_quanto_correlation_risk(book, context)

Only the scalar correlation in ``resolved_quanto_inputs`` is differentiated;
the already-resolved curves, spots, FX spot, and volatility inputs are held
fixed and remain outside the hybrid AAD claim.

For route-level scalar hybrid AD metadata, resolve the quanto inputs with the
factor graph enabled and call a graph-backed derivative helper directly:

.. code-block:: python

   from trellis.analytics import (
       HybridCorrelationStructureRequest,
       HybridDerivativeRequest,
       SparseRiskVector,
       admit_hybrid_ad_lane,
       differentiate_quanto_correlation_matrix,
       differentiate_quanto_scalar_correlation,
       differentiate_quanto_scalar_inputs,
   )
   from trellis.models.resolution.quanto import resolve_quanto_inputs

   resolved = resolve_quanto_inputs(
       market,
       spec,
       include_hybrid_factor_graph=True,
   )
   hybrid = differentiate_quanto_scalar_correlation(
       spec,
       resolved,
       HybridDerivativeRequest(coordinate_space="unconstrained"),
   )
   hybrid_vector = differentiate_quanto_scalar_inputs(spec, resolved)
   # contract_ir is the terminal quanto ContractIR for the same product.
   admission = admit_hybrid_ad_lane(
       contract_ir,
       product_family="quanto_option",
       derivative_method="vjp",
   )
   admitted_hybrid_vector = differentiate_quanto_scalar_inputs(
       spec,
       resolved,
       HybridDerivativeRequest(semantic_admission=admission),
   )
   spot_factor = next(
       factor for factor in hybrid_vector.risk_vector
       if factor.object_type == "spot"
   )
   hybrid_hvp = differentiate_quanto_scalar_inputs(
       spec,
       resolved,
       HybridDerivativeRequest(
           derivative_method="hvp",
           hvp_direction=SparseRiskVector.from_items(((spot_factor, 1.0),)),
       ),
   )
   matrix_request = HybridCorrelationStructureRequest(
       object_name="cross_asset_correlation",
       factors=("EUR", "EURUSD", "USD-OIS"),
       correlation_matrix=(
           (1.0, resolved.corr, 0.10),
           (resolved.corr, 1.0, -0.20),
           (0.10, -0.20, 1.0),
       ),
   )
   matrix_admission = admit_hybrid_ad_lane(
       contract_ir,
       product_family="quanto_option",
       derivative_method="vjp",
       correlation_structure="correlation_matrix",
   )
   hybrid_matrix = differentiate_quanto_correlation_matrix(
       spec,
       resolved,
       matrix_request,
       HybridDerivativeRequest(semantic_admission=matrix_admission),
   )
   print(hybrid.risk_vector.to_payload())
   print(hybrid_vector.risk_vector.to_payload())
   print(admitted_hybrid_vector.method_metadata["semantic_admission"])
   print(hybrid_hvp.risk_vector.to_payload())
   print(hybrid_hvp.method_metadata["resolved_derivative_method"])
   print(hybrid_matrix.risk_vector.to_payload())
   print(hybrid_matrix.method_metadata["resolved_derivative_method"])

The scalar-correlation helper returns a ``HybridDerivativeResult`` with the
typed ``HybridFactorGraph`` payload and one sparse scalar-correlation risk
vector. The scalar-input helper uses executable chart context from the same
graph and returns a sparse vector for supported graph-owned underlier spot, FX
spot, domestic/foreign curve-node, flat/grid vol-node, and scalar-correlation
coordinates. The same helper can return a bounded directional HVP when the
request supplies a non-empty sparse ``hvp_direction`` over graph-owned factors.
For terminal quanto contracts, a checked direct correlation-matrix payload can
also return bounded matrix-coordinate VJP/HVP through
``differentiate_quanto_correlation_matrix(...)``. The executable matrix lane is
strictly direct-entry and smooth-interior: the matrix must be finite,
symmetric, unit diagonal, bounded, positive semidefinite, and above the
minimum-eigenvalue floor. Trellis does not project or repair matrices, and it
does not support correlation surfaces through this lane. Hybrid ``jvp``
requests, surface requests, near-boundary matrices, and broader hybrid product
graphs fail closed.
When a ContractIR semantic contract is available, ``admit_hybrid_ad_lane(...)``
is the admission guard for the scalar-input and matrix-coordinate helpers:
supported same-lane terminal quanto VJP/HVP admissions are copied into result
metadata, while wrong-lane, planned, or unsupported admissions return an empty
risk vector with the admission reason in diagnostics.
For path-dependent, discontinuous-event, early-exercise, or DynamicContractIR
hybrid shapes, that fail-closed result also carries ``semantic_state_policy``
metadata. It tells you whether the blocked shape was a smooth path summary, a
discontinuous event monitor, an early-exercise control, or dynamic state; it
does not mean those shapes have executable pathwise hybrid AD.

For small books that combine already-supported lanes, use the mixed dispatcher:

.. code-block:: python

   from trellis.analytics import (
       BondCurveAADMarketContext,
       VanillaEquityOptionVolAADMarketContext,
   )
   from trellis.book import portfolio_aad_supported_book_risk

   aad = portfolio_aad_supported_book_risk(
       book,
       bond_curve_context=BondCurveAADMarketContext(
           curve=curve,
           settlement=date(2024, 11, 15),
           curve_name="usd_ois",
           currency="USD",
       ),
       equity_option_vol_context=VanillaEquityOptionVolAADMarketContext(
           market_state=market,
           vol_surface_name="spx_flat",
           currency="USD",
       ),
   )

It combines the explicit bond, vanilla/grid-vol option, bounded
arithmetic-Asian, and scalar quanto-correlation AAD lanes when their contexts
are supplied. Unsupported positions remain listed in the typed result and are
excluded from AAD risk.
