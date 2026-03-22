"""Capability inventory: market data vs computational methods.

Market data capabilities require populated MarketState fields.
Computational method capabilities are always available (library code).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketDataCapability:
    """A market data requirement satisfied by a MarketState field."""

    name: str
    description: str
    market_state_field: str
    providing_modules: tuple[str, ...]
    example_usage: str
    how_to_provide: str  # user-facing hint


@dataclass(frozen=True)
class MethodCapability:
    """A computational method available in the library (always importable)."""

    name: str
    description: str
    providing_modules: tuple[str, ...]
    example_usage: str
    requires_market_data: tuple[str, ...]  # which market data it needs


# ---------------------------------------------------------------------------
# Market data capabilities (require populated MarketState fields)
# ---------------------------------------------------------------------------

MARKET_DATA: list[MarketDataCapability] = [
    MarketDataCapability(
        name="discount",
        description="Risk-free discounting via a zero-rate curve.",
        market_state_field="discount",
        providing_modules=("trellis.curves.yield_curve",),
        example_usage="df = market_state.discount.discount(t)",
        how_to_provide="Session(curve=YieldCurve.flat(0.05))",
    ),
    MarketDataCapability(
        name="forward_rate",
        description="Forward rate extraction (auto-derived from discount curve).",
        market_state_field="forward_curve",
        providing_modules=("trellis.curves.forward_curve",),
        example_usage="F = market_state.forward_curve.forward_rate(t1, t2)",
        how_to_provide="Automatically available when discount curve is provided.",
    ),
    MarketDataCapability(
        name="black_vol",
        description="Black (lognormal) implied volatility surface.",
        market_state_field="vol_surface",
        providing_modules=("trellis.models.vol_surface",),
        example_usage="sigma = market_state.vol_surface.black_vol(expiry, strike)",
        how_to_provide="Session(vol_surface=FlatVol(0.20))",
    ),
    MarketDataCapability(
        name="credit",
        description="Credit curve with survival probabilities and hazard rates.",
        market_state_field="credit_curve",
        providing_modules=("trellis.curves.credit_curve",),
        example_usage="S = market_state.credit_curve.survival_probability(t)",
        how_to_provide="Session(credit_curve=CreditCurve.flat(0.02))",
    ),
    MarketDataCapability(
        name="forecast_rate",
        description="Forecast curves for multi-curve pricing.",
        market_state_field="forecast_curves",
        providing_modules=("trellis.curves.forward_curve",),
        example_usage="fwd = market_state.forecast_forward_curve('USD-SOFR-3M')",
        how_to_provide="Session(forecast_curves={'USD-SOFR-3M': sofr_curve})",
    ),
    MarketDataCapability(
        name="state_space",
        description="Discrete states with probabilities for scenario-weighted pricing.",
        market_state_field="state_space",
        providing_modules=("trellis.core.state_space",),
        example_usage="prob = market_state.state_space.probability('fed_cuts')",
        how_to_provide="Session(state_space=StateSpace(states={...}))",
    ),
    MarketDataCapability(
        name="fx",
        description="FX spot rates for cross-currency pricing.",
        market_state_field="fx_rates",
        providing_modules=("trellis.instruments.fx",),
        example_usage="fx = market_state.fx_rates['EURUSD']",
        how_to_provide="Session(fx_rates={'EURUSD': FXRate(1.10, 'USD', 'EUR')})",
    ),
]

# ---------------------------------------------------------------------------
# Computational method capabilities (always available — library code)
# ---------------------------------------------------------------------------

METHODS: list[MethodCapability] = [
    MethodCapability(
        name="rate_tree",
        description="Binomial/trinomial rate tree for backward induction pricing.",
        providing_modules=(
            "trellis.models.trees.binomial",
            "trellis.models.trees.trinomial",
            "trellis.models.trees.backward_induction",
        ),
        example_usage=(
            "from trellis.models.trees import BinomialTree, backward_induction\n"
            "tree = BinomialTree.crr(S0, T, n_steps, r, sigma)\n"
            "price = backward_induction(tree, payoff_fn, r, 'american')"
        ),
        requires_market_data=("discount", "black_vol"),
    ),
    MethodCapability(
        name="monte_carlo",
        description="Monte Carlo simulation engine with variance reduction.",
        providing_modules=(
            "trellis.models.monte_carlo.engine",
            "trellis.models.monte_carlo.discretization",
            "trellis.models.monte_carlo.lsm",
        ),
        example_usage=(
            "from trellis.models.monte_carlo import MonteCarloEngine\n"
            "from trellis.models.processes import GBM\n"
            "engine = MonteCarloEngine(GBM(r, sigma), n_paths=10000)\n"
            "result = engine.price(S0, T, payoff_fn, r)"
        ),
        requires_market_data=("discount",),
    ),
    MethodCapability(
        name="pde_solver",
        description="Finite difference PDE solvers (Crank-Nicolson, implicit, PSOR).",
        providing_modules=(
            "trellis.models.pde.crank_nicolson",
            "trellis.models.pde.implicit_fd",
            "trellis.models.pde.psor",
        ),
        example_usage=(
            "from trellis.models.pde import crank_nicolson_1d, Grid\n"
            "grid = Grid(0, 300, 200, T, 200)\n"
            "V = crank_nicolson_1d(grid, sigma_fn, r_fn, payoff)"
        ),
        requires_market_data=("discount",),
    ),
    MethodCapability(
        name="fft_pricing",
        description="FFT and COS transform-based option pricing.",
        providing_modules=(
            "trellis.models.transforms.fft_pricer",
            "trellis.models.transforms.cos_method",
        ),
        example_usage=(
            "from trellis.models.transforms import fft_price\n"
            "price = fft_price(char_fn, S0, K, T, r)"
        ),
        requires_market_data=("discount",),
    ),
    MethodCapability(
        name="copula",
        description="Copula methods for correlated default simulation.",
        providing_modules=(
            "trellis.models.copulas.gaussian",
            "trellis.models.copulas.factor",
        ),
        example_usage=(
            "from trellis.models.copulas import GaussianCopula, FactorCopula\n"
            "copula = FactorCopula(n_names=100, correlation=0.3)"
        ),
        requires_market_data=("credit",),
    ),
    MethodCapability(
        name="waterfall",
        description="Cash flow waterfall engine for structured products.",
        providing_modules=(
            "trellis.models.cashflow_engine.waterfall",
            "trellis.models.cashflow_engine.prepayment",
            "trellis.models.cashflow_engine.amortization",
        ),
        example_usage=(
            "from trellis.models.cashflow_engine import Waterfall, Tranche, PSA\n"
            "wf = Waterfall([Tranche('A', 80e6, 0.04, 0)])"
        ),
        requires_market_data=("discount",),
    ),
]

# ---------------------------------------------------------------------------
# Unified lookups
# ---------------------------------------------------------------------------

_MARKET_DATA_NAMES = frozenset(c.name for c in MARKET_DATA)
_METHOD_NAMES = frozenset(c.name for c in METHODS)
_ALL_NAMES = _MARKET_DATA_NAMES | _METHOD_NAMES

# Backward-compatible alias
CAPABILITIES = MARKET_DATA  # type: ignore
_KNOWN_NAMES = _ALL_NAMES


def discover_capabilities() -> dict:
    """Return the full capability inventory, split by type."""
    return {
        "market_data": list(MARKET_DATA),
        "methods": list(METHODS),
    }


def analyze_gap(requirements: set[str]) -> tuple[set[str], set[str]]:
    """Check which requirements are satisfiable.

    Returns ``(satisfied, missing)`` where *missing* means the library
    doesn't have the capability at all (not just missing market data).
    """
    satisfied = requirements & _ALL_NAMES
    missing = requirements - _ALL_NAMES
    return satisfied, missing


def check_market_data(
    requirements: set[str],
    market_state,
) -> list[str]:
    """Check which required market data fields are missing from MarketState.

    Returns a list of user-friendly error messages. Empty list = all good.
    """
    errors = []
    available = market_state.available_capabilities

    for cap in MARKET_DATA:
        if cap.name in requirements and cap.name not in available:
            errors.append(
                f"Missing market data: '{cap.name}' — {cap.description}\n"
                f"  How to provide: {cap.how_to_provide}"
            )

    return errors


def capability_summary() -> str:
    """Formatted summary for injection into LLM prompts."""
    lines = ["## Market Data (from MarketState)\n"]
    for c in MARKET_DATA:
        lines.append(f"### `{c.name}`")
        lines.append(f"{c.description}")
        lines.append(f"- MarketState field: `{c.market_state_field}`")
        lines.append(f"- Usage: `{c.example_usage}`")
        lines.append("")

    lines.append("\n## Computational Methods (import from trellis.models)\n")
    lines.append("These are always available — construct them in evaluate().\n")
    for c in METHODS:
        lines.append(f"### `{c.name}`")
        lines.append(f"{c.description}")
        lines.append(f"- Requires market data: {list(c.requires_market_data)}")
        lines.append(f"- Usage:\n```python\n{c.example_usage}\n```")
        lines.append("")

    return "\n".join(lines)
