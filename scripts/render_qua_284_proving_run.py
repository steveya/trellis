#!/usr/bin/env python3
"""Generate the QUA-284 arbitrary-derivative proving-run report."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from collections.abc import Mapping
from datetime import date
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from trellis.cli_paths import resolve_repo_path
from trellis.agent.proving_run_report import render_proving_run_report
from trellis.agent.task_run_store import load_latest_task_run_records
from trellis.core.market_state import MarketState
from trellis.curves.forward_curve import ForwardCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.ranked_observation_payoffs import (
    build_ranked_observation_basket_initial_state,
    build_ranked_observation_basket_process,
    build_ranked_observation_basket_state_payoff,
)
from trellis.models.monte_carlo.semantic_basket import RankedObservationBasketSpec
from trellis.models.resolution.basket_semantics import resolve_basket_semantics
from trellis.models.vol_surface import FlatVol

DEFAULT_OUTPUT = ROOT / "docs" / "qua-284-arbitrary-derivative-proving-run.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the Markdown report.",
    )
    args = parser.parse_args()
    output_path = resolve_repo_path(args.output, DEFAULT_OUTPUT)

    source_record = _load_source_record()
    pricing = _run_mock_pricing_case()

    report = {
        "title": "QUA-284 Arbitrary-Derivative Proving Run",
        "task": source_record["task"],
        "prompt": source_record["prompt"],
        "build_result": source_record["build_result"],
        "deterministic_decisions": source_record["deterministic_decisions"],
        "agent_decisions": source_record["agent_decisions"],
        "semantic": source_record["semantic"],
        "product_ir": source_record["product_ir"],
        "semantic_trace": source_record["semantic_trace"],
        "assembly": source_record["assembly"],
        "pricing": pricing["pricing"],
        "reproducibility": pricing["reproducibility"],
        "output_path": str(output_path),
    }

    markdown = render_proving_run_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    print(f"Wrote {output_path}")
    print(
        json.dumps(
            {
                "task_id": source_record["task"]["id"],
                "semantic_contract": source_record["semantic"]["semantic_id"],
                "clean_price": pricing["pricing"]["clean_price"],
                "dirty_price": pricing["pricing"]["dirty_price"],
                "greeks": pricing["pricing"]["greeks"],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


def _load_source_record() -> dict[str, Any]:
    records = load_latest_task_run_records()
    candidates = [record for record in records if _is_qua284_proving_record(record)]
    if not candidates:
        raise RuntimeError("Could not find a ranked-observation proving task run in task_runs/latest")

    record = max(candidates, key=lambda item: str(item.get("persisted_at") or ""))
    result = dict(record.get("result") or {})
    runtime_contract = dict(result.get("runtime_contract") or {})
    semantic = dict(runtime_contract.get("semantic_contract") or {})
    product = dict(semantic.get("product") or {})
    blueprint = dict(semantic.get("blueprint") or {})
    methods = dict(semantic.get("methods") or {})

    return {
        "task": dict(record.get("task") or {}),
        "prompt": str(runtime_contract.get("description") or ""),
        "deterministic_decisions": {
            "semantic_contract_id": runtime_contract.get("semantic_contract_id"),
            "semantic_version": semantic.get("semantic_version"),
            "candidate_methods": list(methods.get("candidate_methods") or []),
            "preferred_method": methods.get("preferred_method"),
            "target_modules": list(blueprint.get("target_modules") or []),
            "simulation_seed": runtime_contract.get("simulation_seed"),
            "sample_source": runtime_contract.get("sample_source"),
            "sample_indexing": runtime_contract.get("sample_indexing"),
            "simulation_stream_id": runtime_contract.get("simulation_stream_id"),
        },
        "agent_decisions": {
            "contract_surface": runtime_contract.get("semantic_contract_id"),
            "route_family": "family-name-free semantic basket route",
            "assembly_components": [
                "trellis.models.resolution.basket_semantics.resolve_basket_semantics",
                "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_process",
                "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_state_payoff",
                "trellis.models.monte_carlo.engine.MonteCarloEngine",
            ],
        },
        "semantic": {
            "semantic_id": semantic.get("semantic_id"),
            "semantic_version": semantic.get("semantic_version"),
            "instrument_class": product.get("instrument_class"),
            "underlier_structure": product.get("underlier_structure"),
            "payoff_family": product.get("payoff_family"),
            "payoff_rule": product.get("payoff_rule"),
            "settlement_rule": product.get("settlement_rule"),
            "observation_schedule": list(product.get("observation_schedule") or []),
            "constituents": list(product.get("constituents") or []),
            "selection_scope": product.get("selection_scope"),
            "selection_operator": product.get("selection_operator"),
            "selection_count": product.get("selection_count"),
            "lock_rule": product.get("lock_rule"),
            "aggregation_rule": product.get("aggregation_rule"),
            "required_inputs": list((semantic.get("market_data") or {}).get("required_inputs") or []),
            "target_modules": list(blueprint.get("target_modules") or []),
            "primitive_families": list(blueprint.get("primitive_families") or []),
        },
        "product_ir": {
            "instrument_class": product.get("instrument_class"),
            "underlier_structure": product.get("underlier_structure"),
            "payoff_family": product.get("payoff_family"),
            "payoff_rule": product.get("payoff_rule"),
            "settlement_rule": product.get("settlement_rule"),
            "state_dependence": product.get("state_dependence"),
            "schedule_dependence": product.get("schedule_dependence"),
            "multi_asset": product.get("multi_asset"),
            "required_inputs": list((semantic.get("market_data") or {}).get("required_inputs") or []),
        },
        "semantic_trace": {
            "task_id": record.get("task_id"),
            "task_kind": record.get("task_kind"),
            "run_id": record.get("run_id"),
            "persisted_at": record.get("persisted_at"),
            "market": dict(record.get("market") or {}),
            "summary": dict(record.get("summary") or {}),
            "workflow": dict(record.get("workflow") or {}),
        },
        "build_observability": dict(result.get("build_observability") or {}),
        "build_result": dict(record.get("summary") or {}),
        "assembly": {
            "resolver": "trellis.models.resolution.basket_semantics.resolve_basket_semantics",
            "process_builder": "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_process",
            "state_payoff_builder": "trellis.models.monte_carlo.ranked_observation_payoffs.build_ranked_observation_basket_state_payoff",
            "engine": "trellis.models.monte_carlo.engine.MonteCarloEngine",
            "adapter": "trellis.models.monte_carlo.semantic_basket.RankedObservationBasketMonteCarloPayoff",
        },
    }


def _is_qua284_proving_record(record: Mapping[str, Any]) -> bool:
    task = dict(record.get("task") or {})
    result = dict(record.get("result") or {})
    runtime_contract = dict(result.get("runtime_contract") or {})
    semantic_contract_id = str(runtime_contract.get("semantic_contract_id") or result.get("semantic_contract_id") or "")
    title = str(task.get("title") or "").lower()
    return semantic_contract_id == "ranked_observation_basket" or "himalaya ranked observation basket" in title


def _run_mock_pricing_case() -> dict[str, Any]:
    settlement = date(2024, 11, 15)
    expiry = date(2025, 3, 15)
    constituents = ("AAPL", "MSFT", "NVDA")
    underlier_spots = {
        "AAPL": 190.0,
        "MSFT": 410.0,
        "NVDA": 130.0,
    }
    obs_dates = "2025-01-15,2025-02-15,2025-03-15"
    discount_rate = 0.05
    vol = 0.20
    corr = (
        (1.0, 0.35, 0.25),
        (0.35, 1.0, 0.30),
        (0.25, 0.30, 1.0),
    )
    spec = RankedObservationBasketSpec(
        notional=100.0,
        strike=0.02,
        expiry_date=expiry,
        constituents=",".join(constituents),
        observation_dates=obs_dates,
        selection_rule="best_of_remaining",
        lock_rule="remove_selected",
        aggregation_rule="average_locked_returns",
        option_type="call",
        n_paths=8192,
        n_steps=128,
        seed=20260328,
        mc_method="exact",
        correlation_matrix_key="correlation_matrix",
    )
    market_state = _build_market_state(
        settlement,
        underlier_spots,
        discount_rate=discount_rate,
        vol=vol,
        corr=corr,
    )
    base_price, base_resolved = _price_with_common_shocks(market_state, spec)
    shocks = _fixed_shocks(spec.n_paths, spec.n_steps, len(constituents), spec.seed)

    spot_bumps: dict[str, float] = {}
    for constituent in constituents:
        up_state = _build_market_state(
            settlement,
            {**underlier_spots, constituent: underlier_spots[constituent] + 1.0},
            discount_rate=discount_rate,
            vol=vol,
            corr=corr,
        )
        down_state = _build_market_state(
            settlement,
            {**underlier_spots, constituent: underlier_spots[constituent] - 1.0},
            discount_rate=discount_rate,
            vol=vol,
            corr=corr,
        )
        up_price, _ = _price_with_common_shocks(up_state, spec, shocks=shocks)
        down_price, _ = _price_with_common_shocks(down_state, spec, shocks=shocks)
        spot_bumps[constituent] = round((up_price - down_price) / 2.0, 10)

    vol_up_state = _build_market_state(
        settlement,
        underlier_spots,
        discount_rate=discount_rate,
        vol=vol + 0.01,
        corr=corr,
    )
    vol_down_state = _build_market_state(
        settlement,
        underlier_spots,
        discount_rate=discount_rate,
        vol=vol - 0.01,
        corr=corr,
    )
    vol_up_price, _ = _price_with_common_shocks(vol_up_state, spec, shocks=shocks)
    vol_down_price, _ = _price_with_common_shocks(vol_down_state, spec, shocks=shocks)
    common_vega = round((vol_up_price - vol_down_price) / 0.02, 10)

    corr_bump = 0.05
    corr_up = _bump_correlation(corr, corr_bump)
    corr_down = _bump_correlation(corr, -corr_bump)
    corr_up_state = _build_market_state(
        settlement,
        underlier_spots,
        discount_rate=discount_rate,
        vol=vol,
        corr=corr_up,
    )
    corr_down_state = _build_market_state(
        settlement,
        underlier_spots,
        discount_rate=discount_rate,
        vol=vol,
        corr=corr_down,
    )
    corr_up_price, _ = _price_with_common_shocks(corr_up_state, spec, shocks=shocks)
    corr_down_price, _ = _price_with_common_shocks(corr_down_state, spec, shocks=shocks)
    rho_sensitivity = round((corr_up_price - corr_down_price) / (2.0 * corr_bump), 10)

    pricing = {
        "clean_price": round(base_price, 10),
        "dirty_price": round(base_price, 10),
        "accrued_interest": 0.0,
        "greeks": {
            "spot_deltas": spot_bumps,
            "parallel_spot_delta": round(sum(spot_bumps.values()), 10),
            "common_vega": common_vega,
            "correlation_sensitivity": rho_sensitivity,
            "pricing_method": "monte_carlo",
            "pricing_note": "Common-random-number finite differences over the deterministic mock run.",
        },
    }
    reproducibility = {
        "settlement": settlement.isoformat(),
        "expiry": expiry.isoformat(),
        "discount_rate": discount_rate,
        "vol": vol,
        "correlation_matrix": [list(row) for row in corr],
        "resolved_correlation_matrix": [list(row) for row in base_resolved.correlation_matrix],
        "correlation_preflight": _serialize_correlation_preflight(base_resolved.correlation_preflight),
        "underlier_spots": underlier_spots,
        "notional": spec.notional,
        "strike": spec.strike,
        "observation_dates": list(_parse_dates(obs_dates)),
        "n_paths": spec.n_paths,
        "n_steps": spec.n_steps,
        "seed": spec.seed,
        "shock_shape": list(shocks.shape),
        "price_engine": "MonteCarloEngine",
        "payoff_adapter": "RankedObservationBasketMonteCarloPayoff",
        "resolved_time_to_expiry": round(base_resolved.T, 10),
    }
    return {"pricing": pricing, "reproducibility": reproducibility}


def _build_market_state(
    settlement: date,
    underlier_spots: dict[str, float],
    *,
    discount_rate: float,
    vol: float,
    corr: tuple[tuple[float, ...], ...],
) -> MarketState:
    return MarketState(
        as_of=settlement,
        settlement=settlement,
        discount=YieldCurve.flat(discount_rate),
        vol_surface=FlatVol(vol),
        underlier_spots=underlier_spots,
        forecast_curves={name: ForwardCurve(YieldCurve.flat(0.0)) for name in underlier_spots},
        model_parameters={"correlation_matrix": [list(row) for row in corr]},
    )


def _price_with_common_shocks(
    market_state: MarketState,
    spec: RankedObservationBasketSpec,
    *,
    shocks: np.ndarray | None = None,
) -> tuple[float, object]:
    resolved = resolve_basket_semantics(market_state, spec)
    process = build_ranked_observation_basket_process(resolved)
    engine = MonteCarloEngine(
        process,
        n_paths=spec.n_paths,
        n_steps=spec.n_steps,
        seed=spec.seed,
        method=spec.mc_method,
    )
    payoff = build_ranked_observation_basket_state_payoff(spec, resolved)
    if shocks is None:
        shocks = _fixed_shocks(spec.n_paths, spec.n_steps, len(resolved.constituent_names), spec.seed)
    price_result = engine.price(
        build_ranked_observation_basket_initial_state(resolved),
        resolved.T,
        payoff,
        discount_rate=0.0,
        storage_policy="auto",
        return_paths=False,
        shocks=shocks,
    )
    price = float(spec.notional) * float(resolved.domestic_df) * float(price_result["price"])
    return price, resolved


def _fixed_shocks(n_paths: int, n_steps: int, factor_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_paths, n_steps, factor_dim))


def _bump_correlation(
    corr: tuple[tuple[float, ...], ...],
    bump: float,
) -> tuple[tuple[float, ...], ...]:
    arr = np.asarray(corr, dtype=float)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("correlation matrix must be square")
    bumped = arr.copy()
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[1]):
            bumped[i, j] = np.clip(bumped[i, j] + bump, -0.999, 0.999)
            bumped[j, i] = bumped[i, j]
    np.fill_diagonal(bumped, 1.0)
    return tuple(tuple(float(cell) for cell in row) for row in bumped)


def _parse_dates(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _serialize_correlation_preflight(report: object) -> dict[str, Any] | None:
    if report is None:
        return None
    return asdict(report)


if __name__ == "__main__":
    raise SystemExit(main())
