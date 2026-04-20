"""Phase 3 parity and closure cohort for the ContractIR structural compiler.

The goal of this module is narrower than a general benchmark runner:

- prove the admitted structural families bind deterministically
- compare structural shadow authority against the current request compiler path
- leave an explicit blocked outcome for arithmetic Asians

This is the checked evidence surface Phase 4 will consume before any selector
flip or route-retirement deletion is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib import import_module
import json
from pathlib import Path
from typing import Callable, Mapping

from trellis.agent.contract_ir_solver_compiler import (
    build_contract_ir_term_environment,
    compile_contract_ir_solver,
    execute_contract_ir_solver_decision,
)
from trellis.agent.knowledge.decompose import decompose_to_contract_ir
from trellis.agent.knowledge.import_registry import get_repo_revision
from trellis.agent.platform_requests import compile_build_request
from trellis.agent.valuation_context import build_valuation_context
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.equity_exotics import price_equity_variance_swap_analytical
from trellis.models.basket_option import price_basket_option_analytical
from trellis.models.black import (
    black76_asset_or_nothing_call,
    black76_asset_or_nothing_put,
    black76_call,
    black76_cash_or_nothing_call,
    black76_put,
)
from trellis.models.rate_style_swaption import price_swaption_black76
from trellis.models.vol_surface import FlatVol


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ContractIRSolverParityCase:
    case_id: str
    family_id: str
    description: str
    instrument_type: str
    preferred_method: str
    expected_source: str
    expected_shadow_status: str
    expected_declaration_id: str
    reference_price: Callable[[object, MarketState], float] | None
    market_state_factory: Callable[[], MarketState]
    tolerance_abs: float = 1e-12
    tolerance_rel: float = 1e-12


def _import_ref(ref: str):
    module_name, _, symbol = str(ref or "").rpartition(".")
    if not module_name or not symbol:
        raise ValueError(f"Invalid import ref {ref!r}")
    return getattr(import_module(module_name), symbol)


def _equity_market_state(*, underlier: str = "AAPL", spot: float = 165.0, rate: float = 0.03, vol: float = 0.2) -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(rate),
        vol_surface=FlatVol(vol),
        spot=spot,
        underlier_spots={underlier: spot},
    )


def _swaption_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
    )


def _basket_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.20),
        underlier_spots={"SPX": 4500.0, "NDX": 15000.0},
        model_parameters={
            "correlation_matrix": ((1.0, 0.35), (0.35, 1.0)),
            "underlier_carry_rates": {"SPX": 0.0, "NDX": 0.0},
        },
    )


def _variance_market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 1),
        settlement=date(2025, 1, 1),
        discount=YieldCurve.flat(0.02),
        vol_surface=FlatVol(0.25),
        spot=5000.0,
        underlier_spots={"SPX": 5000.0},
    )


def _black_vanilla_reference(decision, market_state: MarketState) -> float:
    call = decision.declaration_id.endswith("_call")
    T = float(decision.call_kwargs["T"])
    K = float(decision.call_kwargs["K"])
    df = float(market_state.discount.discount(T))
    forward = float(market_state.spot) / max(df, 1e-12)
    kernel = black76_call if call else black76_put
    return float(df) * kernel(forward, K, 0.2, T)


def _digital_reference(decision, market_state: MarketState) -> float:
    T = float(decision.call_kwargs["T"])
    K = float(decision.call_kwargs["K"])
    df = float(market_state.discount.discount(T))
    forward = float(market_state.spot) / max(df, 1e-12)
    if decision.declaration_id == "black76_cash_digital_call":
        return 2.0 * df * black76_cash_or_nothing_call(forward, K, 0.2, T)
    if decision.declaration_id == "black76_asset_digital_put":
        return df * black76_asset_or_nothing_put(forward, K, 0.2, T)
    if decision.declaration_id == "black76_asset_digital_call":
        return df * black76_asset_or_nothing_call(forward, K, 0.2, T)
    raise AssertionError(f"Unhandled digital declaration {decision.declaration_id!r}")


def _swaption_reference(decision, market_state: MarketState) -> float:
    return float(price_swaption_black76(market_state, decision.call_kwargs["spec"]))


def _basket_reference(decision, market_state: MarketState) -> float:
    return float(price_basket_option_analytical(market_state, decision.call_kwargs["spec"]))


def _variance_reference(decision, market_state: MarketState) -> float:
    return float(price_equity_variance_swap_analytical(market_state, decision.call_kwargs["spec"]))


def _parity_cases() -> tuple[ContractIRSolverParityCase, ...]:
    return (
        ContractIRSolverParityCase(
            case_id="vanilla_call",
            family_id="vanilla_option",
            description="European call on AAPL strike 150 expiring 2025-11-15",
            instrument_type="european_option",
            preferred_method="analytical",
            expected_source="semantic_blueprint",
            expected_shadow_status="bound",
            expected_declaration_id="black76_vanilla_call",
            reference_price=_black_vanilla_reference,
            market_state_factory=_equity_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="vanilla_put",
            family_id="vanilla_option",
            description="European put on AAPL strike 150 expiring 2025-11-15",
            instrument_type="european_option",
            preferred_method="analytical",
            expected_source="semantic_blueprint",
            expected_shadow_status="bound",
            expected_declaration_id="black76_vanilla_put",
            reference_price=_black_vanilla_reference,
            market_state_factory=_equity_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="digital_cash_call",
            family_id="digital_option",
            description="Cash-or-nothing digital call on AAPL paying $2 if spot > 150 at expiry 2025-11-15",
            instrument_type="digital_option",
            preferred_method="analytical",
            expected_source="request_decomposition",
            expected_shadow_status="bound",
            expected_declaration_id="black76_cash_digital_call",
            reference_price=_digital_reference,
            market_state_factory=_equity_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="digital_asset_put",
            family_id="digital_option",
            description="Asset-or-nothing digital put on AAPL if spot < 150 at expiry 2025-11-15",
            instrument_type="digital_option",
            preferred_method="analytical",
            expected_source="request_decomposition",
            expected_shadow_status="bound",
            expected_declaration_id="black76_asset_digital_put",
            reference_price=_digital_reference,
            market_state_factory=_equity_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="swaption_payer",
            family_id="rate_style_swaption",
            description="European payer swaption on USD-IRS-5Y strike 5% expiring 2025-11-15",
            instrument_type="swaption",
            preferred_method="analytical",
            expected_source="semantic_blueprint",
            expected_shadow_status="bound",
            expected_declaration_id="helper_swaption_payer_black76",
            reference_price=_swaption_reference,
            market_state_factory=_swaption_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="swaption_receiver",
            family_id="rate_style_swaption",
            description="European receiver swaption on USD-IRS-5Y strike 5% expiring 2025-11-15",
            instrument_type="swaption",
            preferred_method="analytical",
            expected_source="semantic_blueprint",
            expected_shadow_status="bound",
            expected_declaration_id="helper_swaption_receiver_black76",
            reference_price=_swaption_reference,
            market_state_factory=_swaption_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="basket_call",
            family_id="basket_option",
            description="European basket call on {SPX 50%, NDX 50%} strike 4500 expiring 2025-11-15",
            instrument_type="basket_option",
            preferred_method="analytical",
            expected_source="request_decomposition",
            expected_shadow_status="bound",
            expected_declaration_id="helper_basket_option_call",
            reference_price=_basket_reference,
            market_state_factory=_basket_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="basket_put",
            family_id="basket_option",
            description="European basket put on {SPX, NDX} strike 4300 expiring 2025-11-15",
            instrument_type="basket_option",
            preferred_method="analytical",
            expected_source="request_decomposition",
            expected_shadow_status="bound",
            expected_declaration_id="helper_basket_option_put",
            reference_price=_basket_reference,
            market_state_factory=_basket_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="variance_swap",
            family_id="variance_swap",
            description="Equity variance swap on SPX, variance strike 0.04, notional 10000, expiry 2025-11-15",
            instrument_type="variance_swap",
            preferred_method="analytical",
            expected_source="request_decomposition",
            expected_shadow_status="bound",
            expected_declaration_id="helper_equity_variance_swap",
            reference_price=_variance_reference,
            market_state_factory=_variance_market_state,
        ),
        ContractIRSolverParityCase(
            case_id="asian_call_blocked",
            family_id="asian_option",
            description="Arithmetic Asian call on SPX monthly average over 2025 strike 4500",
            instrument_type="asian_option",
            preferred_method="analytical",
            expected_source="request_decomposition",
            expected_shadow_status="no_match",
            expected_declaration_id="",
            reference_price=None,
            market_state_factory=_variance_market_state,
        ),
    )


def _structural_decision_for_case(case: ContractIRSolverParityCase, compiled, market_state: MarketState):
    contract_ir = decompose_to_contract_ir(
        case.description,
        instrument_type=case.instrument_type,
    )
    if contract_ir is None:
        return None, None
    valuation_context = (
        compiled.semantic_blueprint.valuation_context
        if getattr(compiled, "semantic_blueprint", None) is not None
        and getattr(compiled.semantic_blueprint, "valuation_context", None) is not None
        else build_valuation_context(
            market_snapshot=market_state,
            requested_outputs=compiled.request.requested_outputs,
            model_spec=compiled.request.model,
        )
    )
    term_environment = build_contract_ir_term_environment(getattr(compiled, "semantic_contract", None))
    decision = compile_contract_ir_solver(
        contract_ir,
        term_environment=term_environment,
        valuation_context=valuation_context,
        market_state=market_state,
        preferred_method=case.preferred_method,
        requested_outputs=compiled.request.requested_outputs,
    )
    return contract_ir, decision


def _run_case(case: ContractIRSolverParityCase) -> dict[str, object]:
    market_state = case.market_state_factory()
    compiled = compile_build_request(
        case.description,
        instrument_type=case.instrument_type,
        market_snapshot=market_state,
        preferred_method=case.preferred_method,
        metadata={"skip_semantic_extension_trace": True},
    )
    compiler_summary = dict(compiled.request.metadata.get("contract_ir_compiler") or {})
    route_authority = dict(compiled.request.metadata.get("route_binding_authority") or {})
    backend_binding = dict(route_authority.get("backend_binding") or {})
    exact_target_refs = tuple(backend_binding.get("exact_target_refs") or ())

    result: dict[str, object] = {
        "case_id": case.case_id,
        "family_id": case.family_id,
        "description": case.description,
        "instrument_type": case.instrument_type,
        "preferred_method": case.preferred_method,
        "expected_source": case.expected_source,
        "expected_shadow_status": case.expected_shadow_status,
        "expected_declaration_id": case.expected_declaration_id,
        "source": compiler_summary.get("source"),
        "shadow_status": compiler_summary.get("shadow_status"),
        "shadow_error": compiler_summary.get("shadow_error"),
        "legacy_route_id": route_authority.get("route_id", ""),
        "legacy_route_family": route_authority.get("route_family", ""),
        "legacy_exact_target_refs": list(exact_target_refs),
        "legacy_exact_target_contains_structural_callable": None,
        "semantic_contract_present": compiled.semantic_contract is not None,
        "semantic_blueprint_present": compiled.semantic_blueprint is not None,
        "product_ir_instrument": getattr(compiled.product_ir, "instrument", ""),
        "product_ir_payoff_family": getattr(compiled.product_ir, "payoff_family", ""),
        "passed": False,
    }

    shadow = dict(compiler_summary.get("contract_ir_solver_shadow") or {})
    if case.expected_shadow_status == "no_match":
        result["passed"] = (
            result["source"] == case.expected_source
            and result["shadow_status"] == "no_match"
            and result["shadow_error"] is not None
        )
        return result

    if result["source"] != case.expected_source or result["shadow_status"] != case.expected_shadow_status:
        return result
    if shadow.get("declaration_id") != case.expected_declaration_id:
        return result

    contract_ir, decision = _structural_decision_for_case(case, compiled, market_state)
    if contract_ir is None or decision is None:
        return result

    structural_price = float(execute_contract_ir_solver_decision(decision))
    reference_price = (
        None
        if case.reference_price is None
        else float(case.reference_price(decision, market_state))
    )
    abs_diff = None if reference_price is None else abs(structural_price - reference_price)
    rel_diff = None
    if reference_price not in {None, 0.0} and abs_diff is not None:
        rel_diff = abs_diff / abs(reference_price)

    exact_contains = decision.callable_ref in exact_target_refs if exact_target_refs else None
    result.update(
        {
            "structural_contract_ir": contract_ir,
            "declaration_id": decision.declaration_id,
            "callable_ref": decision.callable_ref,
            "structural_price": structural_price,
            "reference_price": reference_price,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "legacy_exact_target_contains_structural_callable": exact_contains,
            "passed": (
                decision.declaration_id == case.expected_declaration_id
                and (reference_price is None or abs_diff <= case.tolerance_abs or (rel_diff is not None and rel_diff <= case.tolerance_rel))
            ),
        }
    )
    return result


def _family_notes(case_results: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    if any(result["family_id"] == "asian_option" for result in case_results):
        notes.append(
            "Arithmetic Asians remain an explicit Phase 3 blocker: ContractIR decomposition exists, but the structural solver returns an intentional no-match until a checked arithmetic-Asian solver surface is admitted."
        )
    if any(
        result.get("shadow_status") == "bound" and not str(result.get("legacy_route_id") or "").strip()
        for result in case_results
    ):
        notes.append(
            "The incumbent request path still does not emit a legacy route-authority packet for every passing case in this family, so Phase 4 promotion would be under-evidenced even though structural binding itself succeeds."
        )
    if any(
        result.get("legacy_exact_target_contains_structural_callable") is False
        for result in case_results
    ):
        notes.append(
            "The incumbent request path does not yet advertise the structural callable as an exact target for every passing case in this family, so Phase 4 promotion would still be premature."
        )
    return notes


def build_contract_ir_solver_parity_report() -> dict[str, object]:
    """Build the checked Phase 3 parity / closure report."""

    cases = tuple(_parity_cases())
    case_results = [_run_case(case) for case in cases]
    grouped: dict[str, list[dict[str, object]]] = {}
    for case_result in case_results:
        grouped.setdefault(str(case_result["family_id"]), []).append(case_result)

    family_entries: list[dict[str, object]] = []
    for family_id, results in grouped.items():
        bound_results = [item for item in results if item["shadow_status"] == "bound"]
        no_match_results = [item for item in results if item["shadow_status"] == "no_match"]
        representation_closed = all(item.get("structural_contract_ir") is not None or item.get("shadow_status") == "no_match" for item in results)
        decomposition_closed = all(item.get("source") in {"semantic_blueprint", "request_decomposition"} for item in results)
        lowering_closed = bool(bound_results) and not no_match_results
        parity_closed = bool(bound_results) and all(item.get("passed") for item in bound_results)
        provenance_closed = bool(bound_results) and all(item.get("source") for item in bound_results)
        exact_authority_closed = bool(bound_results) and all(
            str(item.get("legacy_route_id") or "").strip()
            and item.get("legacy_exact_target_contains_structural_callable") in {True, None}
            for item in bound_results
        )
        phase4_candidate = (
            family_id != "asian_option"
            and representation_closed
            and decomposition_closed
            and lowering_closed
            and parity_closed
            and provenance_closed
            and exact_authority_closed
        )
        if family_id == "asian_option":
            lowering_closed = False
            parity_closed = False
            provenance_closed = False
            phase4_candidate = False
        family_entries.append(
            {
                "family_id": family_id,
                "case_count": len(results),
                "representation_closed": representation_closed,
                "decomposition_closed": decomposition_closed,
                "lowering_closed": lowering_closed,
                "parity_closed": parity_closed,
                "provenance_closed": provenance_closed,
                "exact_authority_closed": exact_authority_closed,
                "phase4_candidate": phase4_candidate,
                "blocked": bool(no_match_results),
                "notes": _family_notes(results),
                "cases": results,
            }
        )

    family_entries.sort(key=lambda item: item["family_id"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_revision": get_repo_revision(),
        "status": "completed",
        "families": family_entries,
        "totals": {
            "families": len(family_entries),
            "phase4_candidates": sum(1 for item in family_entries if item["phase4_candidate"]),
            "blocked_families": sum(1 for item in family_entries if item["blocked"]),
            "passed_cases": sum(1 for item in case_results if item["passed"]),
            "failed_cases": sum(1 for item in case_results if not item["passed"]),
        },
    }


def render_contract_ir_solver_parity_report(report: Mapping[str, object]) -> str:
    """Render the parity report as compact Markdown."""

    lines = [
        "# ContractIR Structural Compiler Parity",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Repo revision: `{report.get('repo_revision', '')}`",
        "",
        "## Family Summary",
        "",
        "| Family | Rep | Dec | Low | Parity | Prov | Exact authority | Phase 4 candidate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family in report.get("families", ()):
        lines.append(
            "| {family_id} | {representation_closed} | {decomposition_closed} | {lowering_closed} | {parity_closed} | {provenance_closed} | {exact_authority_closed} | {phase4_candidate} |".format(
                **family
            )
        )

    for family in report.get("families", ()):
        lines.extend(
            [
                "",
                f"## {family['family_id']}",
                "",
            ]
        )
        for note in family.get("notes") or ():
            lines.append(f"- {note}")
        lines.extend(
            [
                "",
                "| Case | Source | Shadow | Declaration | Route | Exact-target contains callable | Passed |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for case in family.get("cases") or ():
            lines.append(
                "| {case_id} | {source} | {shadow_status} | {declaration} | {route} | {contains} | {passed} |".format(
                    case_id=case.get("case_id", ""),
                    source=case.get("source", ""),
                    shadow_status=case.get("shadow_status", ""),
                    declaration=case.get("declaration_id", "") or case.get("expected_declaration_id", ""),
                    route=case.get("legacy_route_id", ""),
                    contains=case.get("legacy_exact_target_contains_structural_callable", ""),
                    passed=case.get("passed", False),
                )
            )
            if case.get("reference_price") is not None:
                lines.append(
                    f"- `{case['case_id']}` value parity: structural=`{case['structural_price']}` reference=`{case['reference_price']}` abs_diff=`{case['abs_diff']}`"
                )
            if case.get("shadow_error") is not None:
                lines.append(
                    f"- `{case['case_id']}` blocker: `{case['shadow_error']['error_type']}` — {case['shadow_error']['message']}"
                )
    return "\n".join(lines) + "\n"


def save_contract_ir_solver_parity_report(
    report: Mapping[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Persist JSON and Markdown parity artifacts."""

    json_path.write_text(json.dumps(report, indent=2, default=str))
    markdown_path.write_text(render_contract_ir_solver_parity_report(report))


def default_parity_artifact_paths() -> tuple[Path, Path]:
    """Return the checked-in artifact paths for the Phase 3 parity ledger."""

    return (
        ROOT / "docs" / "benchmarks" / "contract_ir_solver_parity.json",
        ROOT / "docs" / "benchmarks" / "contract_ir_solver_parity.md",
    )


__all__ = [
    "build_contract_ir_solver_parity_report",
    "default_parity_artifact_paths",
    "render_contract_ir_solver_parity_report",
    "save_contract_ir_solver_parity_report",
]
