"""Registry of what the library can do, split into two categories.

Market data capabilities (e.g. discount_curve, vol_surface) require
the user to supply data via MarketState or Session.

Computational method capabilities (e.g. monte_carlo, rate_tree) are
always available because they are built-in library code.

This module also provides helpers to check whether a payoff's requirements
are satisfiable and to generate human-readable summaries.
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
        name="discount_curve",
        description="Risk-free discounting via a zero-rate curve.",
        market_state_field="discount",
        providing_modules=("trellis.curves.yield_curve",),
        example_usage="df = market_state.discount.discount(t)",
        how_to_provide="Session(curve=YieldCurve.flat(0.05))",
    ),
    MarketDataCapability(
        name="forward_curve",
        description="Forward-rate term structure, either default or forecast-specific.",
        market_state_field="forward_curve",
        providing_modules=("trellis.curves.forward_curve",),
        example_usage="F = market_state.forward_curve.forward_rate(t1, t2)",
        how_to_provide="Automatically available when discount curve is provided.",
    ),
    MarketDataCapability(
        name="black_vol_surface",
        description="Black (lognormal) implied volatility surface.",
        market_state_field="vol_surface",
        providing_modules=("trellis.models.vol_surface",),
        example_usage="sigma = market_state.vol_surface.black_vol(expiry, strike)",
        how_to_provide="Session(vol_surface=FlatVol(0.20))",
    ),
    MarketDataCapability(
        name="credit_curve",
        description="Credit curve with survival probabilities and hazard rates.",
        market_state_field="credit_curve",
        providing_modules=("trellis.curves.credit_curve",),
        example_usage="S = market_state.credit_curve.survival_probability(t)",
        how_to_provide="Session(credit_curve=CreditCurve.flat(0.02))",
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
        name="fx_rates",
        description="FX spot rates for cross-currency pricing.",
        market_state_field="fx_rates",
        providing_modules=("trellis.instruments.fx",),
        example_usage="fx = market_state.fx_rates['EURUSD']",
        how_to_provide="Session(fx_rates={'EURUSD': FXRate(1.10, 'USD', 'EUR')})",
    ),
    MarketDataCapability(
        name="spot",
        description="Underlier spot price for equity/FX-style products.",
        market_state_field="spot",
        providing_modules=("trellis.data.schema",),
        example_usage="s0 = market_state.spot",
        how_to_provide=(
            "Use a MarketSnapshot with underlier_spots={'SPX': 5000.0} "
            "or Session(market_snapshot=resolve_market_snapshot(source='mock'))."
        ),
    ),
    MarketDataCapability(
        name="local_vol_surface",
        description="Local-volatility surface/function sigma(S, t).",
        market_state_field="local_vol_surface",
        providing_modules=(
            "trellis.models.processes.local_vol",
            "trellis.models.calibration.local_vol",
        ),
        example_usage="sigma = market_state.local_vol_surface(S, t)",
        how_to_provide=(
            "Use a MarketSnapshot with local_vol_surfaces={'surface': local_vol_fn}."
        ),
    ),
    MarketDataCapability(
        name="jump_parameters",
        description="Jump-diffusion parameter pack such as Merton lambda/mean/vol.",
        market_state_field="jump_parameters",
        providing_modules=("trellis.models.processes.jump_diffusion",),
        example_usage="lam = market_state.jump_parameters['lam']",
        how_to_provide=(
            "Use a MarketSnapshot with jump_parameter_sets={'merton': {...}}."
        ),
    ),
    MarketDataCapability(
        name="model_parameters",
        description="Model-specific parameter pack such as Heston parameters.",
        market_state_field="model_parameters",
        providing_modules=("trellis.models.processes.heston",),
        example_usage="rho = market_state.model_parameters['rho']",
        how_to_provide=(
            "Use a MarketSnapshot with model_parameter_sets={'heston': {...}}."
        ),
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
            "trellis.models.trees.algebra",
            "trellis.models.trees.lattice",
        ),
        example_usage=(
            "from trellis.models.trees import build_lattice, price_on_lattice\n"
            "from trellis.models.trees.algebra import BINOMIAL_1F_TOPOLOGY, LOG_SPOT_MESH, LATTICE_MODEL_REGISTRY\n"
            "lattice = build_lattice(BINOMIAL_1F_TOPOLOGY, LOG_SPOT_MESH, LATTICE_MODEL_REGISTRY['crr'], spot=S0, rate=r, sigma=sigma, maturity=T, n_steps=n_steps)\n"
            "price = price_on_lattice(lattice, contract_spec)"
        ),
        requires_market_data=("discount_curve", "black_vol_surface"),
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
        requires_market_data=("discount_curve",),
    ),
    MethodCapability(
        name="qmc",
        description=(
            "Low-discrepancy Monte Carlo accelerators such as Sobol sampling "
            "and Brownian-bridge constructions."
        ),
        providing_modules=(
            "trellis.models.qmc",
            "trellis.models.monte_carlo.variance_reduction",
            "trellis.models.monte_carlo.brownian_bridge",
        ),
        example_usage=(
            "from trellis.models.qmc import sobol_normals, brownian_bridge\n"
            "Z = sobol_normals(4096, 64)\n"
            "W = brownian_bridge(T=1.0, n_steps=64, n_paths=4096)"
        ),
        requires_market_data=(),
    ),
    MethodCapability(
        name="pde_solver",
        description="Finite difference PDE solvers (theta-method, PSOR).",
        providing_modules=(
            "trellis.models.pde.theta_method",
            "trellis.models.pde.psor",
        ),
        example_usage=(
            "import numpy as np\n"
            "from trellis.models.pde.grid import Grid\n"
            "from trellis.models.pde.theta_method import theta_method_1d\n"
            "from trellis.models.pde.operator import BlackScholesOperator\n"
            "grid = Grid(x_min=0.0, x_max=300.0, n_x=200, T=T, n_t=200)\n"
            "terminal = np.maximum(grid.x - K, 0.0)\n"
            "op = BlackScholesOperator(lambda s, t: sigma, lambda t: r)\n"
            "V = theta_method_1d(\n"
            "    grid,\n"
            "    op,\n"
            "    terminal,\n"
            "    theta=0.5,\n"
            "    lower_bc_fn=lambda t: 0.0,\n"
            "    upper_bc_fn=lambda t: 300.0 - K * np.exp(-r * (T - t)),\n"
            ")"
        ),
        requires_market_data=("discount_curve",),
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
        requires_market_data=("discount_curve",),
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
        requires_market_data=("credit_curve",),
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
        requires_market_data=("discount_curve",),
    ),
]

# ---------------------------------------------------------------------------
# Unified lookups
# ---------------------------------------------------------------------------

_MARKET_DATA_NAMES = frozenset(c.name for c in MARKET_DATA)
_METHOD_NAMES = frozenset(c.name for c in METHODS)
_ALL_NAMES = _MARKET_DATA_NAMES | _METHOD_NAMES
_MARKET_DATA_BY_NAME = {c.name: c for c in MARKET_DATA}

# Backward-compatible alias
CAPABILITIES = MARKET_DATA  # type: ignore
_KNOWN_NAMES = _ALL_NAMES


def discover_capabilities() -> dict[str, list]:
    """Return all capabilities as {"market_data": [...], "methods": [...]}."""
    return {
        "market_data": list(MARKET_DATA),
        "methods": list(METHODS),
    }


def analyze_gap(requirements: set[str]) -> tuple[set[str], set[str]]:
    """Split requirements into ones the library supports and ones it does not.

    Returns:
        (satisfied, missing) — *satisfied* are names the library recognizes
        (either as market data or computational methods). *missing* are names
        the library has never heard of.
    """
    normalized_requirements = {normalize_capability_name(requirement) for requirement in requirements}
    satisfied = {requirement for requirement in normalized_requirements if requirement in _ALL_NAMES}
    missing = normalized_requirements - _ALL_NAMES
    return satisfied, missing


def normalize_capability_name(requirement: str) -> str:
    """Return the normalized capability label.

    Capability aliases have been retired; callers should already use the
    canonical market-data names from ``MARKET_DATA``.
    """
    return str(requirement).strip()


def validate_market_data_requirements(
    requirements: set[str] | frozenset[str] | tuple[str, ...],
    *,
    allow_methods: bool = False,
) -> set[str]:
    """Return canonical market-data requirements or raise on unknown names."""
    normalized = set()
    unknown = set()
    for requirement in requirements:
        label = normalize_capability_name(requirement)
        if label in _MARKET_DATA_NAMES:
            normalized.add(label)
            continue
        if allow_methods and label in _METHOD_NAMES:
            continue
        if label:
            unknown.add(label)
    if unknown:
        raise ValueError(
            "Unknown market-data requirements "
            f"{sorted(unknown)}. Use canonical capability names "
            f"{sorted(_MARKET_DATA_NAMES)}."
        )
    return normalized


def normalize_market_data_requirements(requirements: set[str] | frozenset[str] | tuple[str, ...]) -> set[str]:
    """Normalize requirement names and keep only canonical market-data ones."""
    normalized = set()
    for requirement in requirements:
        canonical = normalize_capability_name(requirement)
        if canonical in _MARKET_DATA_NAMES:
            normalized.add(canonical)
    return normalized


def _capability_for_requirement(requirement: str) -> MarketDataCapability | None:
    """Look up the MarketDataCapability for a requirement name.

    Returns None if the name does not correspond to a market-data capability.
    """
    canonical = normalize_capability_name(requirement)
    if canonical not in _MARKET_DATA_NAMES:
        return None
    return _MARKET_DATA_BY_NAME[canonical]


def check_market_data(
    requirements: set[str],
    market_state,
) -> list[str]:
    """Check which required market data fields are missing from MarketState.

    Returns a list of human-readable error messages describing each missing
    field and how to provide it. An empty list means everything is satisfied.
    """
    errors = []
    available = market_state.available_capabilities
    normalized_requirements = validate_market_data_requirements(
        requirements,
        allow_methods=True,
    )
    for requirement in normalized_requirements:
        cap = _capability_for_requirement(requirement)
        if cap is None:
            continue
        if requirement in available:
            continue
        errors.append(
            f"Missing market data: '{requirement}' — {cap.description}\n"
            f"  How to provide: {cap.how_to_provide}"
        )

    return errors


def capability_summary(
    requirements: set[str] | frozenset[str] | tuple[str, ...] | None = None,
    *,
    include_methods: bool = True,
) -> str:
    """Build a Markdown summary of available capabilities.

    Used to give the LLM agent context about what market data and methods
    exist. When *requirements* is provided, only matching market-data
    entries are included (to keep prompts focused). Set *include_methods*
    to False to omit the computational methods section.
    """
    lines = ["## Market Data (from MarketState)\n"]
    normalized_requirements = (
        normalize_market_data_requirements(requirements)
        if requirements is not None
        else None
    )
    for c in MARKET_DATA:
        if normalized_requirements is not None and c.name not in normalized_requirements:
            continue
        lines.append(f"### `{c.name}`")
        lines.append(f"{c.description}")
        lines.append(f"- MarketState field: `{c.market_state_field}`")
        lines.append(f"- Usage: `{c.example_usage}`")
        lines.append("")

    if include_methods:
        lines.append("\n## Computational Methods (import from trellis.models)\n")
        lines.append("These are always available — construct them in evaluate().\n")
        for c in METHODS:
            lines.append(f"### `{c.name}`")
            lines.append(f"{c.description}")
            lines.append(f"- Requires market data: {list(c.requires_market_data)}")
            lines.append(f"- Usage:\n```python\n{c.example_usage}\n```")
            lines.append("")

    return "\n".join(lines)
