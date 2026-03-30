"""Import registry — authoritative symbol-to-module lookup for agent codegen.

The formatted registry is still injected into prompts, but Tranche 2B also
exposes structured lookup helpers so the executor can validate imports instead
of relying on prompt text alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import subprocess
from pathlib import Path

from trellis.agent.knowledge.schema import PackageMap, RepoFact, SymbolMap, TestMap

_REPO_ROOT = Path(__file__).resolve().parents[3]

_REGISTRY_CACHE: dict[str, str] = {}
_REGISTRY_DATA_CACHE: dict[str, dict[str, tuple[str, ...]]] = {}
_SYMBOL_INDEX_CACHE: dict[str, dict[str, tuple[str, ...]]] = {}
_PACKAGE_MAP_CACHE: dict[str, PackageMap] = {}
_TEST_MAP_CACHE: dict[str, TestMap] = {}
_REPO_FACTS_CACHE: dict[str, tuple[RepoFact, ...]] = {}


def get_import_registry() -> str:
    """Return the compact import registry for prompt injection."""
    revision = get_repo_revision()
    cached = _REGISTRY_CACHE.get(revision)
    if cached is not None:
        return cached

    formatted = _format_registry(get_registry_snapshot())
    _REGISTRY_CACHE[revision] = formatted
    return formatted


def get_repo_revision() -> str:
    """Return the current git revision used to key live repo facts."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
        ).strip()
        return revision or "unknown"
    except Exception:
        return "unknown"


def get_registry_snapshot() -> dict[str, tuple[str, ...]]:
    """Return a cached copy of the live import registry.

    The returned mapping is module path -> exported public symbols.
    """
    revision = get_repo_revision()
    cached = _REGISTRY_DATA_CACHE.get(revision)
    if cached is not None:
        return cached

    try:
        live_registry = _build_registry_data_from_introspection()
        snapshot = _merge_registry_snapshots(
            live_registry,
            _parse_static_registry(_STATIC_REGISTRY),
        )
    except Exception:
        snapshot = _parse_static_registry(_STATIC_REGISTRY)
    _REGISTRY_DATA_CACHE[revision] = snapshot
    return snapshot


def get_symbol_map() -> SymbolMap:
    """Return the live symbol map keyed by repo revision."""
    revision = get_repo_revision()
    snapshot = get_registry_snapshot()
    cached = _SYMBOL_INDEX_CACHE.get(revision)
    if cached is not None:
        return SymbolMap(
            repo_revision=revision,
            module_to_symbols=snapshot,
            symbol_to_modules=cached,
        )

    symbol_to_modules = _build_symbol_index(snapshot)
    _SYMBOL_INDEX_CACHE[revision] = symbol_to_modules
    return SymbolMap(
        repo_revision=revision,
        module_to_symbols=snapshot,
        symbol_to_modules=symbol_to_modules,
    )


def get_package_map() -> PackageMap:
    """Return a revision-keyed package-to-module map."""
    revision = get_repo_revision()
    cached = _PACKAGE_MAP_CACHE.get(revision)
    if cached is not None:
        return cached

    package_to_modules, module_to_package = _build_package_map(get_registry_snapshot())
    package_map = PackageMap(
        repo_revision=revision,
        package_to_modules=package_to_modules,
        module_to_package=module_to_package,
    )
    _PACKAGE_MAP_CACHE[revision] = package_map
    return package_map


def get_test_map() -> TestMap:
    """Return a revision-keyed map of test directories and likely test targets."""
    revision = get_repo_revision()
    cached = _TEST_MAP_CACHE.get(revision)
    if cached is not None:
        return cached

    directory_to_tests, symbol_to_tests = _build_test_map()
    test_map = TestMap(
        repo_revision=revision,
        directory_to_tests=directory_to_tests,
        symbol_to_tests=symbol_to_tests,
    )
    _TEST_MAP_CACHE[revision] = test_map
    return test_map


def get_repo_facts() -> tuple[RepoFact, ...]:
    """Return a compact set of live repo facts for prompt-time validation."""
    revision = get_repo_revision()
    cached = _REPO_FACTS_CACHE.get(revision)
    if cached is not None:
        return cached

    symbol_map = get_symbol_map()
    package_map = get_package_map()
    test_map = get_test_map()
    facts = (
        RepoFact(
            kind="revision",
            key="git_commit",
            value=revision,
            repo_revision=revision,
        ),
        RepoFact(
            kind="symbol_map",
            key="module_count",
            value=str(len(symbol_map.module_to_symbols)),
            repo_revision=revision,
        ),
        RepoFact(
            kind="package_map",
            key="package_count",
            value=str(len(package_map.package_to_modules)),
            repo_revision=revision,
        ),
        RepoFact(
            kind="test_map",
            key="directory_count",
            value=str(len(test_map.directory_to_tests)),
            repo_revision=revision,
        ),
    )
    _REPO_FACTS_CACHE[revision] = facts
    return facts


def list_module_exports(module_path: str) -> tuple[str, ...]:
    """Return public exported symbols for a module path."""
    return get_registry_snapshot().get(module_path, ())


def module_exists(module_path: str) -> bool:
    """Return whether a module path is present in the registry."""
    return module_path in get_registry_snapshot()


def find_symbol_modules(symbol: str) -> tuple[str, ...]:
    """Return every registry module exporting ``symbol``."""
    index = _get_symbol_index()
    return index.get(symbol, ())


def resolve_import_candidates(symbols: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Resolve a batch of symbols to candidate modules."""
    return {
        symbol: find_symbol_modules(symbol)
        for symbol in sorted(set(symbols))
    }


def suggest_tests_for_symbol(symbol: str) -> tuple[str, ...]:
    """Return likely test targets for a symbol, module, or package token."""
    token = symbol.split(".")[-1].replace("_", "").lower()
    test_map = get_test_map()
    matches: list[str] = []
    for directory, tests in test_map.directory_to_tests.items():
        for test_path in tests:
            stem = Path(test_path).stem.lower().replace("_", "")
            if token and token in stem:
                matches.append(test_path)
    return tuple(dict.fromkeys(sorted(matches)))


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
    global _PACKAGE_MAP_CACHE, _TEST_MAP_CACHE, _REPO_FACTS_CACHE
    _REGISTRY_CACHE = {}
    _REGISTRY_DATA_CACHE = {}
    _SYMBOL_INDEX_CACHE = {}
    _PACKAGE_MAP_CACHE = {}
    _TEST_MAP_CACHE = {}
    _REPO_FACTS_CACHE = {}


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


def _build_symbol_index(registry: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Invert the registry into symbol -> module candidates."""
    index: dict[str, list[str]] = defaultdict(list)
    for module_path, exports in registry.items():
        for exported in exports:
            index[exported].append(module_path)
    return {
        name: tuple(sorted(paths))
        for name, paths in index.items()
    }


def _get_symbol_index() -> dict[str, tuple[str, ...]]:
    """Return the revision-keyed symbol index."""
    revision = get_repo_revision()
    cached = _SYMBOL_INDEX_CACHE.get(revision)
    if cached is not None:
        return cached
    index = _build_symbol_index(get_registry_snapshot())
    _SYMBOL_INDEX_CACHE[revision] = index
    return index


def _build_package_map(
    registry: dict[str, tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Group live modules into stable package roots."""
    package_to_modules: dict[str, list[str]] = defaultdict(list)
    module_to_package: dict[str, str] = {}
    for module_path in sorted(registry):
        package = _module_package_root(module_path)
        package_to_modules[package].append(module_path)
        module_to_package[module_path] = package
    return (
        {package: tuple(sorted(modules)) for package, modules in package_to_modules.items()},
        module_to_package,
    )


def _module_package_root(module_path: str) -> str:
    """Return the coarse package root for a module path."""
    parts = module_path.split(".")
    if len(parts) <= 2:
        return module_path
    return ".".join(parts[:2])


def _build_test_map() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Build revision-scoped test-directory and symbol-hint maps."""
    tests_root = _REPO_ROOT / "tests"
    directory_to_tests: dict[str, list[str]] = defaultdict(list)
    symbol_to_tests: dict[str, set[str]] = defaultdict(set)
    if not tests_root.exists():
        return {}, {}

    for path in sorted(tests_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel_path = path.relative_to(_REPO_ROOT).as_posix()
        rel_dir = path.parent.relative_to(_REPO_ROOT).as_posix()
        directory_to_tests[rel_dir].append(rel_path)

        stem = path.stem.lower()
        token = stem[5:] if stem.startswith("test_") else stem
        token = token.replace("_", "")
        if token:
            symbol_to_tests[token].add(rel_path)

    return (
        {directory: tuple(sorted(paths)) for directory, paths in directory_to_tests.items()},
        {symbol: tuple(sorted(paths)) for symbol, paths in symbol_to_tests.items()},
    )


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
        elif "trellis.models.resolution" in mod:
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
from trellis.models.black import black76_call, black76_put, black76_asset_or_nothing_call, black76_asset_or_nothing_put, black76_cash_or_nothing_call, black76_cash_or_nothing_put
from trellis.models.analytical import terminal_vanilla_from_basis
from trellis.models.analytical.jamshidian import zcb_option_hw
from trellis.models.analytical.barrier import barrier_option_price, down_and_out_call, down_and_in_call
from trellis.models.resolution.quanto import ResolvedQuantoInputs, resolve_quanto_correlation, resolve_quanto_foreign_curve, resolve_quanto_inputs, resolve_quanto_underlier_spot
from trellis.models.resolution.basket_semantics import ResolvedBasketSemantics, resolve_basket_semantics

### Models — Trees
from trellis.models.trees.lattice import build_rate_lattice, build_spot_lattice, lattice_backward_induction, build_generic_lattice, calibrate_lattice
from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction

### Models — Monte Carlo
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.basket_state import build_basket_path_requirement, evaluate_ranked_observation_basket_paths, evaluate_ranked_observation_basket_state, observation_step_indices
from trellis.models.monte_carlo.profiling import MonteCarloPathKernelBenchmark, benchmark_path_kernel
from trellis.models.monte_carlo.lsm import longstaff_schwartz, laguerre_basis
from trellis.models.monte_carlo.primal_dual import primal_dual_mc, primal_dual_mc_result
from trellis.models.monte_carlo.stochastic_mesh import stochastic_mesh, stochastic_mesh_result
from trellis.models.monte_carlo.discretization import euler_maruyama, milstein, exact_simulation
from trellis.models.monte_carlo.variance_reduction import antithetic, control_variate, sobol_normals
from trellis.models.monte_carlo.schemes import Euler, Milstein, Exact, LogEuler, LaguerreBasis, PolynomialBasis
from trellis.models.monte_carlo.ranked_observation_payoffs import build_ranked_observation_basket_initial_state, build_ranked_observation_basket_process, build_ranked_observation_basket_state_payoff, price_ranked_observation_basket_monte_carlo, recommended_ranked_observation_basket_mc_engine_kwargs, terminal_ranked_observation_basket_payoff
from trellis.models.monte_carlo.semantic_basket import RankedObservationBasketMonteCarloPayoff, RankedObservationBasketSpec

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
from trellis.models.calibration.rates import RatesCalibrationResult, calibrate_cap_floor_black_vol, calibrate_swaption_black_vol, swaption_terms
from trellis.models.calibration.sabr_fit import calibrate_sabr

### Models — Cashflow Engine
from trellis.models.cashflow_engine.prepayment import PSA, CPR, RateDependent
from trellis.models.cashflow_engine.waterfall import Waterfall, Tranche
from trellis.models.cashflow_engine.amortization import level_pay, scheduled

### Models — Vol Surface
from trellis.models.vol_surface import FlatVol, VolSurface
"""
