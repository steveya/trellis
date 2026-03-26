"""Import registry — authoritative symbol-to-module lookup for agent codegen.

The formatted registry is still injected into prompts, but Tranche 2B also
exposes structured lookup helpers so the executor can validate imports instead
of relying on prompt text alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

_REGISTRY_CACHE: str | None = None
_REGISTRY_DATA_CACHE: dict[str, tuple[str, ...]] | None = None
_SYMBOL_INDEX_CACHE: dict[str, tuple[str, ...]] | None = None


def get_import_registry() -> str:
    """Return the compact import registry for prompt injection."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE

    _REGISTRY_CACHE = _format_registry(get_registry_snapshot())
    return _REGISTRY_CACHE


def get_registry_snapshot() -> dict[str, tuple[str, ...]]:
    """Return a cached copy of the live import registry.

    The returned mapping is module path -> exported public symbols.
    """
    global _REGISTRY_DATA_CACHE
    if _REGISTRY_DATA_CACHE is not None:
        return _REGISTRY_DATA_CACHE

    try:
        live_registry = _build_registry_data_from_introspection()
        _REGISTRY_DATA_CACHE = _merge_registry_snapshots(
            live_registry,
            _parse_static_registry(_STATIC_REGISTRY),
        )
    except Exception:
        _REGISTRY_DATA_CACHE = _parse_static_registry(_STATIC_REGISTRY)
    return _REGISTRY_DATA_CACHE


def list_module_exports(module_path: str) -> tuple[str, ...]:
    """Return public exported symbols for a module path."""
    return get_registry_snapshot().get(module_path, ())


def module_exists(module_path: str) -> bool:
    """Return whether a module path is present in the registry."""
    return module_path in get_registry_snapshot()


def find_symbol_modules(symbol: str) -> tuple[str, ...]:
    """Return every registry module exporting ``symbol``."""
    global _SYMBOL_INDEX_CACHE
    if _SYMBOL_INDEX_CACHE is None:
        index: dict[str, list[str]] = defaultdict(list)
        for module_path, exports in get_registry_snapshot().items():
            for exported in exports:
                index[exported].append(module_path)
        _SYMBOL_INDEX_CACHE = {
            name: tuple(sorted(paths))
            for name, paths in index.items()
        }
    return _SYMBOL_INDEX_CACHE.get(symbol, ())


def resolve_import_candidates(symbols: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Resolve a batch of symbols to candidate modules."""
    return {
        symbol: find_symbol_modules(symbol)
        for symbol in sorted(set(symbols))
    }


def is_valid_import(module_path: str, symbol: str | None = None) -> bool:
    """Validate that a module or module+symbol pair exists in the registry."""
    exports = get_registry_snapshot().get(module_path)
    if exports is None:
        return False
    if symbol is None:
        return True
    return symbol in exports


def reset_registry_cache() -> None:
    """Clear registry caches for tests."""
    global _REGISTRY_CACHE, _REGISTRY_DATA_CACHE, _SYMBOL_INDEX_CACHE
    _REGISTRY_CACHE = None
    _REGISTRY_DATA_CACHE = None
    _SYMBOL_INDEX_CACHE = None


def _build_registry_data_from_introspection() -> dict[str, tuple[str, ...]]:
    """Build the structured registry dynamically by introspecting the package."""
    import importlib
    import inspect
    import pkgutil
    from pathlib import Path

    pkg = importlib.import_module("trellis")
    pkg_path = Path(pkg.__file__).parent

    registry: dict[str, tuple[str, ...]] = {}

    include_prefixes = (
        "trellis.models.",
        "trellis.core.",
        "trellis.curves.",
    )
    exclude_prefixes = (
        "trellis.instruments._agent.",
        "trellis.agent.",
        "trellis.data.",
    )

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=[str(pkg_path)], prefix="trellis.",
    ):
        if not any(modname.startswith(prefix) for prefix in include_prefixes):
            continue
        if any(modname.startswith(prefix) for prefix in exclude_prefixes):
            continue

        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        symbols = []
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if getattr(obj, "__module__", "") == modname:
                    symbols.append(name)

        if symbols:
            registry[modname] = tuple(sorted(symbols))

    for mod_path in (
        "trellis.instruments.bond",
        "trellis.instruments.cap",
        "trellis.instruments.callable_bond",
        "trellis.instruments.barrier_option",
        "trellis.instruments.swap",
        "trellis.instruments.nth_to_default",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue

        symbols = []
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if getattr(obj, "__module__", "") == mod_path:
                    symbols.append(name)
        if symbols:
            registry[mod_path] = tuple(sorted(symbols))

    return dict(sorted(registry.items()))


def _parse_static_registry(registry_text: str) -> dict[str, tuple[str, ...]]:
    """Parse the static fallback text registry into structured data."""
    registry: dict[str, tuple[str, ...]] = {}
    for line in registry_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("from ") or " import " not in stripped:
            continue
        _, rest = stripped.split("from ", 1)
        module_path, symbols_text = rest.split(" import ", 1)
        symbols = tuple(
            sorted(
                symbol.strip()
                for symbol in symbols_text.split(",")
                if symbol.strip()
            )
        )
        if symbols:
            registry[module_path.strip()] = symbols
    return dict(sorted(registry.items()))


def _merge_registry_snapshots(
    primary: dict[str, tuple[str, ...]],
    fallback: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Merge live and fallback registries, preserving live modules and exports.

    The fallback registry carries a few compatibility re-exports that are not
    discoverable via plain introspection, such as lazy ``__getattr__`` exports.
    """
    merged: dict[str, tuple[str, ...]] = {}
    for module_path in sorted(set(primary) | set(fallback)):
        symbols = set(primary.get(module_path, ()))
        symbols.update(fallback.get(module_path, ()))
        if symbols:
            merged[module_path] = tuple(sorted(symbols))
    return merged


def _format_registry(registry: dict[str, tuple[str, ...]]) -> str:
    """Format the registry as import statements."""
    lines = [
        "## AVAILABLE IMPORTS — use ONLY these, NEVER invent module paths\n",
        "**CRITICAL**: If a module or symbol is not listed below, it does NOT exist.",
        "Do NOT guess paths like 'pde_solver', 'simulation', 'pricing', 'pdesolvers'.",
        "Do NOT use CamelCase in module paths (e.g., NOT 'coxRossRubinstein').",
        "Every import in your code MUST come from this list.\n",
    ]

    # Group by top-level category
    groups = {
        "Core": [],
        "Curves": [],
        "Models — Analytical": [],
        "Models — Trees": [],
        "Models — Monte Carlo": [],
        "Models — QMC": [],
        "Models — PDE": [],
        "Models — Transforms (FFT/COS)": [],
        "Models — Processes": [],
        "Models — Copulas": [],
        "Models — Calibration": [],
        "Models — Cashflow Engine": [],
        "Models — Vol Surface": [],
        "Instruments (reference)": [],
    }

    for mod, symbols in sorted(registry.items()):
        line = f"from {mod} import {', '.join(symbols)}"

        if "trellis.core." in mod:
            groups["Core"].append(line)
        elif "trellis.curves." in mod:
            groups["Curves"].append(line)
        elif "trellis.models.black" in mod:
            groups["Models — Analytical"].append(line)
        elif "trellis.models.analytical" in mod:
            groups["Models — Analytical"].append(line)
        elif "trellis.models.trees" in mod:
            groups["Models — Trees"].append(line)
        elif "trellis.models.monte_carlo" in mod:
            groups["Models — Monte Carlo"].append(line)
        elif "trellis.models.qmc" in mod:
            groups["Models — QMC"].append(line)
        elif "trellis.models.pde" in mod:
            groups["Models — PDE"].append(line)
        elif "trellis.models.transforms" in mod:
            groups["Models — Transforms (FFT/COS)"].append(line)
        elif "trellis.models.processes" in mod:
            groups["Models — Processes"].append(line)
        elif "trellis.models.copulas" in mod:
            groups["Models — Copulas"].append(line)
        elif "trellis.models.calibration" in mod:
            groups["Models — Calibration"].append(line)
        elif "trellis.models.cashflow" in mod:
            groups["Models — Cashflow Engine"].append(line)
        elif "trellis.models.vol" in mod:
            groups["Models — Vol Surface"].append(line)
        elif "trellis.instruments" in mod:
            groups["Instruments (reference)"].append(line)

    for group_name, group_lines in groups.items():
        if group_lines:
            lines.append(f"\n### {group_name}")
            lines.extend(group_lines)

    return "\n".join(lines)


# Static fallback — used if introspection fails
_STATIC_REGISTRY = """\
## AVAILABLE IMPORTS — use ONLY these, NEVER invent module paths

If a module or symbol is not listed below, it does NOT exist.

### Core
from trellis.core.date_utils import generate_schedule, year_fraction, add_months
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency

### Curves
from trellis.curves.yield_curve import YieldCurve
from trellis.curves.forward_curve import ForwardCurve
from trellis.curves.credit_curve import CreditCurve

### Models — Analytical
from trellis.models.black import black76_call, black76_put
from trellis.models.analytical.jamshidian import zcb_option_hw
from trellis.models.analytical.barrier import barrier_option_price, down_and_out_call, down_and_in_call

### Models — Trees
from trellis.models.trees.lattice import build_rate_lattice, build_spot_lattice, lattice_backward_induction, build_generic_lattice, calibrate_lattice
from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction

### Models — Monte Carlo
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.lsm import longstaff_schwartz, laguerre_basis
from trellis.models.monte_carlo.primal_dual import primal_dual_mc, primal_dual_mc_result
from trellis.models.monte_carlo.stochastic_mesh import stochastic_mesh, stochastic_mesh_result
from trellis.models.monte_carlo.discretization import euler_maruyama, milstein, exact_simulation
from trellis.models.monte_carlo.variance_reduction import antithetic, control_variate, sobol_normals
from trellis.models.monte_carlo.schemes import Euler, Milstein, Exact, LogEuler, LaguerreBasis, PolynomialBasis

### Models — QMC
from trellis.models.qmc import brownian_bridge, sobol_normals

### Models — PDE
from trellis.models.pde.theta_method import theta_method_1d
from trellis.models.pde.operator import BlackScholesOperator, CEVOperator, HeatOperator, PDEOperator
from trellis.models.pde.rate_operator import HullWhitePDEOperator
from trellis.models.pde.psor import psor_1d
from trellis.models.pde.grid import Grid
from trellis.models.pde.thomas import thomas_solve

### Models — Transforms (FFT/COS)
from trellis.models.transforms.cos_method import cos_price
from trellis.models.transforms.fft_pricer import fft_price

### Models — Processes
from trellis.models.processes.gbm import GBM
from trellis.models.processes.heston import Heston
from trellis.models.processes.hull_white import HullWhite
from trellis.models.processes.vasicek import Vasicek
from trellis.models.processes.cir import CIR
from trellis.models.processes.sabr import SABRProcess
from trellis.models.processes.jump_diffusion import MertonJumpDiffusion
from trellis.models.processes.local_vol import LocalVol

### Models — Copulas
from trellis.models.copulas.gaussian import GaussianCopula
from trellis.models.copulas.factor import FactorCopula
from trellis.models.copulas.student_t import StudentTCopula

### Models — Calibration
from trellis.models.calibration.implied_vol import implied_vol, implied_vol_jaeckel
from trellis.models.calibration.local_vol import dupire_local_vol
from trellis.models.calibration.sabr_fit import calibrate_sabr

### Models — Cashflow Engine
from trellis.models.cashflow_engine.prepayment import PSA, CPR, RateDependent
from trellis.models.cashflow_engine.waterfall import Waterfall, Tranche
from trellis.models.cashflow_engine.amortization import level_pay, scheduled

### Models — Vol Surface
from trellis.models.vol_surface import FlatVol, VolSurface
"""
