"""Remediation loop: analyze task failures, fix knowledge gaps, re-run.

Usage:
    python scripts/remediate.py                   # analyze + fix + re-run
    python scripts/remediate.py --analyze-only     # just show what's wrong
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_all_results() -> list[dict]:
    """Load all task result files."""
    results = []
    for f in sorted(ROOT.glob("task_results_*.json")):
        with open(f) as fh:
            results.extend(json.load(fh))
    return results


def analyze_failures(results: list[dict]) -> dict:
    """Categorize failures into actionable groups."""
    failures = [r for r in results if not r.get("success")]

    categories = {
        "import_hallucination": [],    # Wrong import paths
        "missing_cookbook": [],         # No cookbook for method
        "missing_decomposition": [],   # LLM decomposition fell through
        "implementation_gap": [],      # Library doesn't support feature
        "validation_failure": [],      # Code ran but produced wrong results
        "other": [],
    }

    for r in failures:
        err = r.get("failures", [""])[0]
        gaps = r.get("knowledge_gaps", [])
        conf = r.get("gap_confidence", 1.0)

        if "No module named" in err or "ImportError" in err:
            categories["import_hallucination"].append(r)
        elif any("cookbook" in g.lower() for g in gaps):
            categories["missing_cookbook"].append(r)
        elif conf < 0.3:
            categories["missing_decomposition"].append(r)
        elif "validation" in err.lower() or "assert" in err.lower():
            categories["validation_failure"].append(r)
        else:
            categories["other"].append(r)

    return categories


def analyze_platform_traces() -> dict[str, int]:
    """Summarize unified platform traces by execution action."""
    try:
        from trellis.agent.platform_traces import (
            load_platform_traces,
            summarize_platform_traces,
        )

        traces = load_platform_traces()
        return summarize_platform_traces(traces)
    except Exception:
        return {}


def fix_import_knowledge():
    """Enrich the knowledge system with explicit import path lessons."""
    from trellis.agent.knowledge.promotion import capture_lesson, validate_lesson, promote_lesson

    # The #1 failure: agents hallucinate import paths
    import_lessons = [
        {
            "category": "convention",
            "title": "Use exact trellis import paths — never guess module names",
            "severity": "critical",
            "symptom": "ModuleNotFoundError or ImportError on trellis modules",
            "root_cause": (
                "The agent invents module paths like 'pde_solver', 'simulation', "
                "'pdesolvers' that don't exist. The actual module paths are:\n"
                "- PDE: trellis.models.pde.theta_method (theta_method_1d, BlackScholesOperator)\n"
                "- PDE: trellis.models.pde.operator (PDEOperator base)\n"
                "- MC: trellis.models.monte_carlo.engine (MonteCarloEngine)\n"
                "- MC: trellis.models.monte_carlo.lsm (LSM, LaguerreBasis)\n"
                "- MC: trellis.models.monte_carlo.discretization (schemes)\n"
                "- Trees: trellis.models.trees.lattice (build_rate_lattice, lattice_backward_induction)\n"
                "- FFT: trellis.models.transforms.fft_pricer (fft_price)\n"
                "- FFT: trellis.models.transforms.cos_method (cos_price)\n"
                "- Processes: trellis.models.processes.gbm (GBM)\n"
                "- Processes: trellis.models.processes.heston (HestonProcess)\n"
                "- Processes: trellis.models.processes.hull_white (HullWhiteProcess)\n"
                "- Black: trellis.models.black (black76_call, black76_put)\n"
                "- Copulas: trellis.models.copulas.gaussian (GaussianCopula)\n"
                "- Copulas: trellis.models.copulas.factor (FactorCopula)"
            ),
            "fix": (
                "ALWAYS use the exact import paths listed above. "
                "NEVER guess or invent module names. "
                "If unsure, use trellis.models.black for analytical, "
                "trellis.models.monte_carlo.engine for MC, "
                "trellis.models.trees.lattice for trees, "
                "trellis.models.pde.theta_method for PDE, "
                "trellis.models.transforms.cos_method for FFT/COS."
            ),
            "method": None,  # applies to ALL methods
            "features": ["discounting"],  # broad match
        },
    ]

    for lesson in import_lessons:
        lid = capture_lesson(
            category=lesson["category"],
            title=lesson["title"],
            severity=lesson["severity"],
            symptom=lesson["symptom"],
            root_cause=lesson["root_cause"],
            fix=lesson["fix"],
            method=lesson["method"],
            features=lesson["features"],
            confidence=1.0,  # this is certain knowledge
            validation="Import paths verified against actual library structure",
        )
        if lid:
            validate_lesson(lid)
            promote_lesson(lid)
            print(f"  Created + promoted: {lid}")
        else:
            print(f"  Skipped (duplicate): {lesson['title'][:40]}...")


def fix_pde_cookbook():
    """Add a PDE cookbook since it's missing."""
    import yaml

    cookbook_path = ROOT / "trellis" / "agent" / "knowledge" / "canonical" / "cookbooks.yaml"
    data = {}
    if cookbook_path.exists():
        data = yaml.safe_load(cookbook_path.read_text()) or {}

    if "pde_solver" not in data:
        data["pde_solver"] = {
            "template": """\
## Cookbook: PDE (finite difference theta-method)
Use this pattern for European/American options via PDE.

```python
def evaluate(self, market_state):
    from trellis.core.date_utils import year_fraction
    from trellis.models.pde.theta_method import theta_method_1d
    from trellis.models.pde.operator import BlackScholesOperator
    import numpy as np

    spec = self._spec
    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    r = float(market_state.discount.zero_rate(T))
    sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

    # Grid: S from 0 to ~4*spot
    S_max = 4.0 * spec.spot
    n_S = 200
    n_t = max(200, int(T * 252))

    # Terminal payoff
    S_grid = np.linspace(0, S_max, n_S + 1)
    # >>> INSTRUMENT-SPECIFIC: define terminal payoff <<<
    payoff = np.maximum(S_grid - spec.strike, 0)  # call example

    # PDE operator
    op = BlackScholesOperator(r=r, sigma=sigma)

    # Solve: theta=0.5 for Crank-Nicolson, theta=1.0 for fully implicit
    V = theta_method_1d(op, payoff, S_grid, T, n_t, theta=0.5)

    # Interpolate to get price at spot
    price = float(np.interp(spec.spot, S_grid, V))
    return price
```
""",
            "description": "Finite difference PDE solver using theta-method (Crank-Nicolson or implicit)",
            "applicable_instruments": ["european_option", "american_option", "barrier_option"],
            "version": "auto",
        }
        with open(cookbook_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print("  Added PDE cookbook")

    if "fft_pricing" not in data:
        data["fft_pricing"] = {
            "template": """\
## Cookbook: FFT/COS Transform Pricing
Use this pattern for options under models with known characteristic functions (Heston, VG, etc.)

```python
def evaluate(self, market_state):
    from trellis.core.date_utils import year_fraction
    from trellis.models.transforms.cos_method import cos_price
    from trellis.models.processes.heston import HestonProcess
    import numpy as np

    spec = self._spec
    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    r = float(market_state.discount.zero_rate(T))

    # >>> INSTRUMENT-SPECIFIC: set up process and parameters <<<
    # Heston example:
    # v0 is VARIANCE not vol: v0 = sigma^2. sigma=0.20 means v0=0.04
    process = HestonProcess(
        v0=spec.v0, kappa=spec.kappa, theta=spec.theta,
        sigma_v=spec.sigma_v, rho=spec.rho,
    )

    # COS method expects CF of log-return log(S_T/S0), NOT log(S_T)
    # Truncation: center on first cumulant, width from second cumulant
    price = cos_price(
        char_fn=process.characteristic_function,
        S0=spec.spot, K=spec.strike, T=T, r=r,
        N=256,  # number of terms
        option_type="call",  # or "put"
    )

    return float(price) * spec.notional
```
""",
            "description": "FFT and COS transform-based pricing using characteristic functions",
            "applicable_instruments": ["heston_option", "european_option"],
            "version": "auto",
        }
        with open(cookbook_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print("  Added FFT/COS cookbook")


def fix_method_requirements():
    """Add missing method requirements for PDE and FFT."""
    import yaml

    req_path = ROOT / "trellis" / "agent" / "knowledge" / "canonical" / "method_requirements.yaml"
    data = yaml.safe_load(req_path.read_text()) if req_path.exists() else {}

    if "pde_solver" not in data:
        data["pde_solver"] = [
            "GRID: Use at least 200 spatial points and 200+ time steps. "
            "For barrier options, use non-uniform grid concentrated near the barrier.",
            "BOUNDARY CONDITIONS: Set V(0,t)=0 for calls, V(S_max,t)=S_max-K*exp(-r*(T-t)) for calls. "
            "For puts: V(0,t)=K*exp(-r*(T-t)), V(S_max,t)=0.",
            "STABILITY: Crank-Nicolson (theta=0.5) is second-order but may oscillate near discontinuities. "
            "Use Rannacher smoothing (2-4 implicit steps at start) for digital/barrier payoffs.",
        ]
        print("  Added PDE requirements")

    if "fft_pricing" not in data:
        data["fft_pricing"] = [
            "CHARACTERISTIC FUNCTION CONVENTION: COS method expects CF of log-return log(S_T/S0). "
            "FFT (Carr-Madan) expects CF of log(S_T). Do NOT mix them.",
            "HESTON v0 IS VARIANCE: v0=0.04 means vol=20%. Do NOT pass vol directly as v0.",
            "COS TRUNCATION: Center on first cumulant c1, width L*sqrt(c2) where c2 is second cumulant. "
            "Default symmetric [-L, L] can overflow for high drift or long maturity.",
        ]
        print("  Added FFT requirements")

    with open(req_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def fix_data_contracts():
    """Add missing data contracts for PDE and FFT methods."""
    import yaml

    contracts_path = ROOT / "trellis" / "agent" / "knowledge" / "canonical" / "data_contracts.yaml"
    data = yaml.safe_load(contracts_path.read_text()) if contracts_path.exists() else []

    methods_present = {c["method"] for c in data}

    if "pde_solver" not in methods_present:
        data.append({
            "name": "VOL_BLACK_FOR_PDE",
            "method": "pde_solver",
            "source": "market_state.vol_surface.black_vol(T, K)",
            "convention": "Black lognormal implied vol (annualized)",
            "typical_range": "0.10 to 0.60",
            "model_expects": "Black lognormal vol (same units — no conversion needed)",
            "conversion": "none — use directly as sigma in BlackScholesOperator or GBM diffusion",
            "model_range": "0.10 to 0.60",
            "warning": "",
        })
        print("  Added PDE data contract")

    if "fft_pricing" not in methods_present:
        data.append({
            "name": "VOL_FOR_FFT_HESTON",
            "method": "fft_pricing",
            "source": "Heston model parameters (v0, kappa, theta, sigma_v, rho)",
            "convention": "v0 is VARIANCE (not vol). v0=0.04 means 20% vol.",
            "typical_range": "v0: 0.01 to 0.16 (vol: 10% to 40%)",
            "model_expects": "Heston variance v0, NOT Black implied vol",
            "conversion": "v0 = sigma_Black^2. Example: 20% vol → v0 = 0.04",
            "model_range": "v0: 0.01 to 0.16",
            "warning": "NEVER pass Black vol directly as v0. A Black vol of 0.20 is v0=0.04, NOT v0=0.20.",
        })
        print("  Added FFT data contract")

    with open(contracts_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def print_analysis(categories: dict):
    """Print failure analysis."""
    total = sum(len(v) for v in categories.values())
    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS: {total} total failures")
    print(f"{'='*60}")

    for cat, tasks in categories.items():
        if not tasks:
            continue
        print(f"\n{cat.upper()} ({len(tasks)}):")
        for r in tasks:
            err = r.get("failures", [""])[0][:80]
            print(f"  {r['task_id']}: {err}")

    platform_summary = analyze_platform_traces()
    if platform_summary:
        print(f"\nPLATFORM TRACE SUMMARY ({sum(platform_summary.values())}):")
        for action, count in sorted(platform_summary.items()):
            print(f"  {action}: {count}")


def rerun_failed(results: list[dict], model: str = "gpt-5-mini"):
    """Re-run failed tasks after knowledge fixes."""
    from scripts.run_tasks import run_task, build_market_state
    from trellis.agent.task_runtime import load_tasks

    failed_tasks_by_id = {r["task_id"]: r for r in results if not r.get("success")}
    if not failed_tasks_by_id:
        print("No failures to re-run!")
        return []

    all_tasks = load_tasks(status=None, root=ROOT)
    tasks_to_rerun = [t for t in all_tasks if t["id"] in failed_tasks_by_id]
    print(f"\nRe-running {len(tasks_to_rerun)} failed tasks...")

    ms = build_market_state()
    rerun_results = []
    for task in tasks_to_rerun:
        result = run_task(task, ms, model)
        rerun_results.append(result)

    # Save rerun results
    with open(ROOT / "task_results_rerun.json", "w") as f:
        json.dump(rerun_results, f, indent=2, default=str)

    ok = sum(1 for r in rerun_results if r.get("success"))
    print(f"\nRe-run: {ok}/{len(rerun_results)} now succeed")
    return rerun_results


def main():
    analyze_only = "--analyze-only" in sys.argv

    print("Loading results...")
    results = load_all_results()
    if not results:
        print("No results found. Run tasks first.")
        return

    ok = sum(1 for r in results if r.get("success"))
    print(f"Loaded {len(results)} results: {ok} success, {len(results)-ok} failures")

    categories = analyze_failures(results)
    print_analysis(categories)

    if analyze_only:
        return

    print(f"\n{'='*60}")
    print("FIXING KNOWLEDGE GAPS")
    print(f"{'='*60}")

    # Fix #1: Import paths (biggest failure cause)
    print("\n1. Import path knowledge:")
    fix_import_knowledge()

    # Fix #2: Missing cookbooks
    print("\n2. Missing cookbooks:")
    fix_pde_cookbook()

    # Fix #3: Missing method requirements
    print("\n3. Missing method requirements:")
    fix_method_requirements()

    # Fix #4: Missing data contracts
    print("\n4. Missing data contracts:")
    fix_data_contracts()

    # Reload knowledge store
    from trellis.agent.knowledge import reload
    reload()
    print("\nKnowledge store reloaded.")

    # Re-run failed tasks
    print(f"\n{'='*60}")
    print("RE-RUNNING FAILED TASKS")
    print(f"{'='*60}")
    rerun_failed(results)


if __name__ == "__main__":
    main()
