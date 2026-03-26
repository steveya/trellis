"""Reusable helpers for task reruns and offline task benchmarking."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time
from typing import Any, Callable

from trellis.agent.executor import _make_test_payoff, _try_import_existing
from trellis.agent.knowledge.methods import is_known_method, normalize_method
from trellis.agent.planner import FieldDef, SpecSchema
from trellis.agent.planner import plan_build
from trellis.agent.quant import select_pricing_method


ROOT = Path(__file__).resolve().parents[2]
PRICING_TASKS_MANIFEST = "TASKS.yaml"
FRAMEWORK_TASKS_MANIFEST = "FRAMEWORK_TASKS.yaml"
DEFAULT_SETTLEMENT = date(2024, 11, 15)
_MARKET_SELECTION_KEYS = (
    "discount_curve",
    "forecast_curve",
    "vol_surface",
    "credit_curve",
    "fx_rate",
    "state_space",
    "underlier_spot",
    "local_vol_surface",
    "jump_parameters",
    "model_parameters",
)
_GENERIC_TITLE_STOPWORDS = {
    "a",
    "across",
    "and",
    "build",
    "for",
    "maturities",
    "option",
    "pricer",
    "puts",
    "the",
    "via",
    "vs",
}


@dataclass(frozen=True)
class PreparedTask:
    """Offline-ready task metadata backed by an existing generated module."""

    task_id: str
    title: str
    description: str
    instrument_type: str
    requirements: set[str]
    payoff_cls: type
    spec_schema: object


@dataclass(frozen=True)
class ComparisonBuildTarget:
    """One concrete build target within a comparison task."""

    target_id: str
    preferred_method: str
    is_reference: bool = False


@dataclass(frozen=True)
class TaskContractError(ValueError):
    """Structured error for tasks that do not fit the pricing-task runner."""

    code: str
    message: str
    explanation: str
    suggestion: str | None = None

    def __str__(self) -> str:
        parts = [self.message, self.explanation]
        if self.suggestion:
            parts.append(f"Suggested action: {self.suggestion}")
        return " ".join(part.strip() for part in parts if part and part.strip())


_FALLBACK_AGENT_MODULES: dict[str, tuple[str, str]] = {
    "american_option": ("trellis.instruments._agent.americanputpayoff", "AmericanOptionPayoff"),
    "american_put": ("trellis.instruments._agent.americanputpayoff", "AmericanOptionPayoff"),
}

_GENERIC_FALLBACK_AGENT_MODULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("fft vs cos", "gbm calls/puts"),
        "trellis.instruments._agent.fftvscos",
        "FFTvsCOSPricer",
    ),
)


def _load_task_manifest(
    manifest_name: str,
    *,
    root: Path = ROOT,
) -> list[dict]:
    import yaml

    with open(root / manifest_name) as f:
        tasks = yaml.safe_load(f)

    if not isinstance(tasks, list):
        return []
    return tasks


def _filter_loaded_tasks(
    tasks: list[dict],
    start_id: str | None = None,
    end_id: str | None = None,
    *,
    status: str | None = "pending",
) -> list[dict]:
    filtered = list(tasks)

    if status is not None:
        filtered = [task for task in filtered if task.get("status") == status]

    if start_id and end_id:
        start_num = int(start_id.lstrip("TE"))
        end_num = int(end_id.lstrip("TE"))
        prefix = start_id[0]
        filtered = [
            task for task in filtered
            if task["id"].startswith(prefix)
            and start_num <= int(task["id"].lstrip("TE")) <= end_num
        ]

    return filtered


def load_tasks(
    start_id: str | None = None,
    end_id: str | None = None,
    *,
    root: Path = ROOT,
    status: str | None = "pending",
) -> list[dict]:
    """Load pricing tasks from ``TASKS.yaml`` with optional status/range filters."""
    tasks = _load_task_manifest(PRICING_TASKS_MANIFEST, root=root)
    return _filter_loaded_tasks(tasks, start_id, end_id, status=status)


def load_framework_tasks(
    start_id: str | None = None,
    end_id: str | None = None,
    *,
    root: Path = ROOT,
    status: str | None = "pending",
) -> list[dict]:
    """Load framework/meta tasks from ``FRAMEWORK_TASKS.yaml``."""
    tasks = _load_task_manifest(FRAMEWORK_TASKS_MANIFEST, root=root)
    return _filter_loaded_tasks(tasks, start_id, end_id, status=status)


def build_market_state():
    """Create a standard market state with broad capability coverage."""
    try:
        from trellis.data.resolver import resolve_market_snapshot

        snapshot = resolve_market_snapshot(as_of=DEFAULT_SETTLEMENT, source="mock")
        return snapshot.to_market_state(settlement=DEFAULT_SETTLEMENT)
    except Exception:
        from trellis.core.market_state import MarketState
        from trellis.curves.yield_curve import YieldCurve
        from trellis.models.vol_surface import FlatVol

        settle = DEFAULT_SETTLEMENT
        return MarketState(
            as_of=settle,
            settlement=settle,
            discount=YieldCurve.flat(0.05, max_tenor=31.0),
            vol_surface=FlatVol(0.20),
        )


def build_market_snapshot_for_task(task: dict):
    """Resolve a market snapshot for a task-level market spec, if present."""
    market_spec = task.get("market") or {}
    if not market_spec:
        return None

    from trellis.data.resolver import resolve_market_snapshot

    return resolve_market_snapshot(
        as_of=market_spec.get("as_of", DEFAULT_SETTLEMENT),
        source=market_spec.get("source", "mock"),
    )


def build_market_state_for_task(task: dict, fallback_market_state=None):
    """Resolve a task-specific MarketState plus provenance/context metadata."""
    snapshot = build_market_snapshot_for_task(task)
    if snapshot is None:
        market_state = fallback_market_state if fallback_market_state is not None else build_market_state()
        return market_state, {
            "source": "default",
            "as_of": getattr(market_state, "as_of", None).isoformat() if getattr(market_state, "as_of", None) else None,
            "selected_components": {},
            "available_capabilities": sorted(getattr(market_state, "available_capabilities", ())),
            "metadata": {},
        }

    market_spec = task.get("market") or {}
    selected_components = {
        key: market_spec[key]
        for key in _MARKET_SELECTION_KEYS
        if market_spec.get(key) is not None
    }

    market_state = snapshot.to_market_state(
        settlement=DEFAULT_SETTLEMENT,
        discount_curve=selected_components.get("discount_curve"),
        forecast_curve=selected_components.get("forecast_curve"),
        vol_surface=selected_components.get("vol_surface"),
        credit_curve=selected_components.get("credit_curve"),
        fx_rate=selected_components.get("fx_rate"),
        state_space=selected_components.get("state_space"),
        underlier_spot=selected_components.get("underlier_spot"),
        local_vol_surface=selected_components.get("local_vol_surface"),
        jump_parameters=selected_components.get("jump_parameters"),
        model_parameters=selected_components.get("model_parameters"),
    )

    market_context = {
        "source": snapshot.source,
        "as_of": snapshot.as_of.isoformat(),
        "selected_components": selected_components,
        "available_capabilities": sorted(market_state.available_capabilities),
        "metadata": dict(snapshot.metadata),
    }
    _validate_task_market_assertions(task, market_context)
    return market_state, market_context


def task_to_description(task: dict) -> str:
    """Convert a ``TASKS.yaml`` entry into a pricing-build request string."""
    return f"Build a pricer for: {task['title']}"


def task_to_instrument_type(task: dict) -> str | None:
    """Heuristically resolve the most likely instrument type for a task."""
    title = task["title"].lower()
    mappings = [
        ("american put", "american_put"),
        ("american option", "american_option"),
        ("worst-of", "basket_option"),
        ("worst of", "basket_option"),
        ("best-of", "basket_option"),
        ("best of", "basket_option"),
        ("rainbow", "basket_option"),
        ("spread option", "basket_option"),
        ("basket", "basket_option"),
        ("european equity call", "european_option"),
        ("european equity put", "european_option"),
        ("european call", "european_option"),
        ("european put", "european_option"),
        ("european option", "european_option"),
        ("callable bond", "callable_bond"),
        ("puttable bond", "puttable_bond"),
        ("bermudan swaption", "bermudan_swaption"),
        ("barrier", "barrier_option"),
        ("asian option", "asian_option"),
        ("asian", "asian_option"),
        ("lookback", "barrier_option"),
        ("autocallable", "autocallable"),
        ("variance swap", "variance_swap"),
        ("heston", "heston_option"),
        ("cev", "european_option"),
        ("cdo", "cdo"),
        ("cds", "nth_to_default"),
        ("nth-to-default", "nth_to_default"),
        ("swaption", "swaption"),
        ("cap", "cap"),
        ("floor", "floor"),
        ("convertible", "callable_bond"),
        ("mbs", "mbs"),
        ("range accrual", "callable_bond"),
        ("digital", "european_option"),
        ("compound option", "european_option"),
        ("chooser", "european_option"),
        ("cliquet", "autocallable"),
        ("double barrier", "barrier_option"),
        ("quanto", "european_option"),
        ("forward start", "european_option"),
        ("fx", "european_option"),
        ("swap", "swap"),
        ("bond", "bond"),
    ]
    for pattern, instrument_type in mappings:
        if pattern in title:
            return instrument_type
    return None


def run_task(
    task: dict,
    market_state,
    *,
    model: str = "gpt-5-mini",
    force_rebuild: bool = True,
    validation: str = "standard",
    max_retries: int = 3,
    build_fn: Callable[..., Any] | None = None,
    timer: Callable[[], float] = time,
    now_fn: Callable[[], datetime] = datetime.now,
    payoff_factory: Callable[[type, object, date], Any] | None = None,
    price_fn: Callable[[Any, Any], float] | None = None,
) -> dict:
    """Execute one task through the knowledge-aware build pipeline."""
    from trellis.agent.task_run_store import persist_task_run_record

    if build_fn is None:
        from trellis.agent.knowledge.autonomous import build_with_knowledge

        build_fn = build_with_knowledge

    task_id = task["id"]
    description = task_to_description(task)
    instrument_type = task_to_instrument_type(task)
    construct_methods = _task_construct_methods(task)
    comparison_targets = _task_comparison_targets(task, construct_methods)
    comparison_task = len(comparison_targets) > 1

    print(f"\n{'=' * 60}")
    print(f"  {task_id}: {task['title']}")
    print(f"  instrument_type={instrument_type}")
    if construct_methods:
        print(f"  construct_methods={construct_methods}")
    print(f"{'=' * 60}")

    t0 = timer()
    base_request_metadata = {
        "task_id": task_id,
        "task_title": task["title"],
    }
    result_data = {
        "task_id": task_id,
        "title": task["title"],
        "instrument_type": instrument_type,
        "start_time": now_fn().isoformat(),
        "comparison_task": comparison_task,
        "construct_methods": construct_methods,
        "comparison_targets": [target.target_id for target in comparison_targets],
        "cross_validate": task.get("cross_validate"),
        "new_component": task.get("new_component"),
    }

    try:
        _validate_task_contract(task, instrument_type, construct_methods)
        market_state, market_context = build_market_state_for_task(task, market_state)
        result_data["market_context"] = market_context
        if comparison_task:
            method_results = {}
            live_results = {}
            for target in comparison_targets:
                build_kwargs = {
                    "description": _description_for_comparison_target(description, target),
                    "instrument_type": instrument_type,
                    "preferred_method": target.preferred_method,
                    "comparison_target": target.target_id,
                    "request_metadata": {
                        **base_request_metadata,
                        "comparison_target": target.target_id,
                        "preferred_method": target.preferred_method,
                    },
                    "model": model,
                    "market_state": market_state,
                    "max_retries": max_retries,
                    "validation": validation,
                    "force_rebuild": force_rebuild,
                }
                result = build_fn(**build_kwargs)
                live_results[target.target_id] = result
                method_results[target.target_id] = _build_result_payload(
                    result,
                    preferred_method=target.preferred_method,
                    reference_target=target.is_reference,
                )

            elapsed = timer() - t0
            successful_methods = [
                method for method, payload in method_results.items()
                if payload["success"]
            ]
            lesson_ids = [
                payload["reflection"].get("lesson_captured")
                for payload in method_results.values()
                if payload["reflection"].get("lesson_captured")
            ]
            all_gaps = sorted({
                gap
                for payload in method_results.values()
                for gap in payload["knowledge_gaps"]
            })
            cross_validation = _cross_validate_comparison_task(
                comparison_targets,
                live_results,
                market_state,
                configured_targets=task.get("cross_validate") or {},
                payoff_factory=payoff_factory,
                price_fn=price_fn,
            )
            result_data.update({
                "success": (
                    all(payload["success"] for payload in method_results.values())
                    and cross_validation["status"] == "passed"
                ),
                "attempts": sum(payload["attempts"] for payload in method_results.values()),
                "elapsed_seconds": round(elapsed, 1),
                "gap_confidence": round(mean(
                    payload["gap_confidence"] for payload in method_results.values()
                ), 2) if method_results else 0.0,
                "knowledge_gaps": all_gaps,
                "method_results": method_results,
                "artifacts": _aggregate_artifacts(method_results),
                "knowledge_summary": _aggregate_knowledge_summaries(method_results),
                "token_usage_summary": _aggregate_token_usage(method_results),
                "agent_observation_count": sum(
                    payload.get("agent_observation_count", 0)
                    for payload in method_results.values()
                ),
                "preferred_method": None,
                "reflection": {
                    "lesson_captured": lesson_ids,
                    "cookbook_enriched": any(
                        payload["reflection"].get("cookbook_enriched")
                        for payload in method_results.values()
                    ),
                    "method_reflections": {
                        method: payload["reflection"]
                        for method, payload in method_results.items()
                    },
                },
                "cross_validation": cross_validation,
            })
        else:
            preferred_method = construct_methods[0] if construct_methods else None
            build_kwargs = {
                "description": description,
                "instrument_type": instrument_type,
                "request_metadata": {
                    **base_request_metadata,
                    **(
                        {"preferred_method": preferred_method}
                        if preferred_method is not None
                        else {}
                    ),
                },
                "model": model,
                "market_state": market_state,
                "max_retries": max_retries,
                "validation": validation,
                "force_rebuild": force_rebuild,
            }
            if preferred_method is not None:
                build_kwargs["preferred_method"] = preferred_method
            if task.get("cross_validate") and comparison_targets:
                build_kwargs["comparison_target"] = comparison_targets[0].target_id
            result = build_fn(**build_kwargs)
            elapsed = timer() - t0
            result_data.update({
                **_build_result_payload(result, preferred_method=preferred_method),
                "elapsed_seconds": round(elapsed, 1),
                "preferred_method": preferred_method,
            })
        status = "OK" if result_data.get("success") else "FAIL"
        print(
            f"  [{status}] {elapsed:.1f}s, attempts={result_data.get('attempts', 0)}, "
            f"confidence={result_data.get('gap_confidence', 0):.0%}"
        )
    except TaskContractError as exc:
        elapsed = timer() - t0
        result_data.update({
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(exc),
            "failures": [exc.message],
            "task_contract_error": {
                "code": exc.code,
                "message": exc.message,
                "explanation": exc.explanation,
                "suggestion": exc.suggestion,
            },
        })
        print(f"  [INVALID TASK] {elapsed:.1f}s: {exc.message}")
    except Exception as exc:
        elapsed = timer() - t0
        result_data.update({
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(exc)[:200],
        })
        print(f"  [ERROR] {elapsed:.1f}s: {type(exc).__name__}: {str(exc)[:100]}")

    try:
        persisted = persist_task_run_record(task, result_data)
        result_data["task_run_history_path"] = persisted["history_path"]
        result_data["task_run_latest_path"] = persisted["latest_path"]
        result_data["task_run_latest_index_path"] = persisted["latest_index_path"]
    except Exception as exc:
        result_data["task_run_persist_error"] = str(exc)[:200]

    return result_data


def _validate_task_market_assertions(
    task: dict,
    market_context: dict[str, Any],
) -> None:
    """Validate task-level market assertions against the resolved market context."""
    assertions = task.get("market_assertions") or {}
    if not assertions:
        return

    errors: list[str] = []
    available_capabilities = set(market_context.get("available_capabilities", ()))
    required = assertions.get("requires") or ()
    for capability in required:
        if capability not in available_capabilities:
            errors.append(f"missing asserted capability: {capability}")

    selected = assertions.get("selected") or {}
    actual_selected = market_context.get("selected_components", {})
    for key, expected in selected.items():
        actual = actual_selected.get(key)
        if actual != expected:
            errors.append(
                f"selected component mismatch for {key}: expected {expected!r}, got {actual!r}"
            )

    if errors:
        raise ValueError("; ".join(errors))


def _validate_task_contract(
    task: dict,
    instrument_type: str | None,
    construct_methods: list[str],
) -> None:
    """Fail early when a task spec does not make sense for the pricing runner."""
    construct = task.get("construct")
    if isinstance(construct, str):
        normalized_construct = construct.strip().lower()
    else:
        normalized_construct = None

    if normalized_construct not in {"framework", "infrastructure", "experience"}:
        return

    market_spec = task.get("market") or {}
    selected_market_keys = sorted(key for key, value in market_spec.items() if value is not None)
    cross_validate = task.get("cross_validate") or {}
    external_targets = list(cross_validate.get("external") or ())
    internal_targets = list(cross_validate.get("internal") or ())

    explanation_parts = [
        "This task describes framework or meta-development work rather than pricing a concrete instrument build.",
        "The pricing task runner only supports concrete pricing/comparison tasks over a priceable instrument and method family.",
    ]
    if instrument_type is not None:
        explanation_parts.append(
            f"The current heuristic instrument hint `{instrument_type}` comes from the title text, but that does not make the task a concrete pricing request."
        )
    if selected_market_keys:
        explanation_parts.append(
            "The declared market context "
            f"({', '.join(selected_market_keys)}) is therefore not a coherent requirement for this runner."
        )
    if internal_targets or external_targets:
        explanation_parts.append(
            "Its cross-validation targets "
            f"(internal={internal_targets or []}, external={external_targets or []}) describe framework evaluation, not direct pricing output comparison."
        )

    suggestion = (
        "Route this through a dedicated framework-eval harness, or rewrite it as a concrete pricing/comparison task with a specific instrument, methods, and coherent market inputs."
    )
    raise TaskContractError(
        code="framework_task_not_priceable",
        message=f"Task {task.get('id', '<unknown>')} does not make sense as a pricing-task run.",
        explanation=" ".join(explanation_parts),
        suggestion=suggestion,
    )


def _task_construct_methods(task: dict) -> list[str]:
    """Normalize known construct method hints from TASKS metadata."""
    construct = task.get("construct")
    if construct is None:
        return []
    raw_methods = construct if isinstance(construct, (list, tuple)) else [construct]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_methods:
        method = normalize_method(str(item))
        if not is_known_method(method):
            continue
        if method in seen:
            continue
        normalized.append(method)
        seen.add(method)
    return normalized


def _task_comparison_targets(
    task: dict,
    construct_methods: list[str],
) -> list[ComparisonBuildTarget]:
    """Resolve concrete comparison targets from TASKS metadata."""
    from trellis.agent.assembly_tools import build_comparison_harness_plan

    harness_plan = build_comparison_harness_plan(task)
    if harness_plan.targets:
        return [
            ComparisonBuildTarget(
                target_id=target.target_id,
                preferred_method=target.preferred_method,
                is_reference=target.is_reference,
            )
            for target in harness_plan.targets
        ]

    return [
        ComparisonBuildTarget(target_id=method, preferred_method=method)
        for method in construct_methods
    ]


def _preferred_method_for_target(target_id: str, construct_methods: list[str]) -> str:
    """Map a concrete cross-validation target to a canonical method family."""
    normalized_target = normalize_method(target_id)
    if normalized_target in {"analytical", "rate_tree", "monte_carlo", "pde_solver", "fft_pricing"}:
        return normalized_target

    explicit_patterns = (
        ("tree", "rate_tree"),
        ("lattice", "rate_tree"),
        ("pde", "pde_solver"),
        ("psor", "pde_solver"),
        ("mc", "monte_carlo"),
        ("monte", "monte_carlo"),
        ("lsm", "monte_carlo"),
        ("dual", "monte_carlo"),
        ("mesh", "monte_carlo"),
        ("stochastic", "monte_carlo"),
        ("fft", "fft_pricing"),
        ("cos", "fft_pricing"),
        ("garman", "analytical"),
        ("gk", "analytical"),
        ("black", "analytical"),
        ("jamshidian", "analytical"),
        ("rubinstein", "analytical"),
    )
    lower = target_id.lower()
    for pattern, method in explicit_patterns:
        if pattern in lower:
            return method

    if len(construct_methods) == 1:
        return construct_methods[0]
    return "analytical"


def _description_for_comparison_target(
    base_description: str,
    target: ComparisonBuildTarget,
) -> str:
    """Augment a task description with a concrete comparison-target hint."""
    return (
        f"{base_description}\n\n"
        f"Implementation target: {target.target_id}\n"
        f"Preferred method family: {target.preferred_method}"
    )


def _build_result_payload(
    result: Any,
    *,
    preferred_method: str | None = None,
    reference_target: bool = False,
) -> dict[str, Any]:
    """Project a BuildResult-like object into a stable task result payload."""
    payload = {
        "success": result.success,
        "attempts": result.attempts,
        "gap_confidence": result.gap_confidence,
        "knowledge_gaps": result.knowledge_gaps,
        "payoff_class": result.payoff_cls.__name__ if result.payoff_cls else None,
        "preferred_method": preferred_method,
        "reference_target": reference_target,
        "failures": result.failures[:3],
        "agent_observation_count": len(getattr(result, "agent_observations", []) or []),
        "agent_observations": list(getattr(result, "agent_observations", []) or []),
        "knowledge_summary": dict(getattr(result, "knowledge_summary", {}) or {}),
        "token_usage_summary": dict(getattr(result, "token_usage_summary", {}) or {}),
        "platform_trace_path": getattr(result, "platform_trace_path", None),
        "platform_request_id": getattr(result, "platform_request_id", None),
        "blocker_details": getattr(result, "blocker_details", None),
        "reflection": {
            key: value for key, value in result.reflection.items()
            if key in (
                "lessons_attributed",
                "lesson_captured",
                "gaps_identified",
                "cookbook_enriched",
                "cookbook_candidate_saved",
                "knowledge_trace_saved",
                "knowledge_gap_log_saved",
                "decomposition_saved",
                "distill_run",
            )
        },
    }
    payload["artifacts"] = _artifacts_from_payload(payload)
    return payload


def _artifacts_from_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Collect stable artifact references from one build payload."""
    reflection = payload.get("reflection") or {}
    return {
        "platform_request_ids": _unique_strings([payload.get("platform_request_id")]),
        "platform_trace_paths": _unique_strings([payload.get("platform_trace_path")]),
        "knowledge_trace_paths": _unique_strings([reflection.get("knowledge_trace_saved")]),
        "cookbook_candidate_paths": _unique_strings([reflection.get("cookbook_candidate_saved")]),
        "knowledge_gap_log_paths": _unique_strings([reflection.get("knowledge_gap_log_saved")]),
    }


def _aggregate_artifacts(method_payloads: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Merge artifact references across comparison-target payloads."""
    return {
        "platform_request_ids": _unique_strings(
            payload.get("artifacts", {}).get("platform_request_ids", ())
            for payload in method_payloads.values()
        ),
        "platform_trace_paths": _unique_strings(
            payload.get("artifacts", {}).get("platform_trace_paths", ())
            for payload in method_payloads.values()
        ),
        "knowledge_trace_paths": _unique_strings(
            payload.get("artifacts", {}).get("knowledge_trace_paths", ())
            for payload in method_payloads.values()
        ),
        "cookbook_candidate_paths": _unique_strings(
            payload.get("artifacts", {}).get("cookbook_candidate_paths", ())
            for payload in method_payloads.values()
        ),
        "knowledge_gap_log_paths": _unique_strings(
            payload.get("artifacts", {}).get("knowledge_gap_log_paths", ())
            for payload in method_payloads.values()
        ),
    }


def _aggregate_knowledge_summaries(method_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge compact shared-knowledge summaries across comparison builds."""
    summaries = [
        payload.get("knowledge_summary") or {}
        for payload in method_payloads.values()
    ]
    if not summaries:
        return {}

    return {
        "principle_ids": _unique_strings(summary.get("principle_ids", ()) for summary in summaries),
        "lesson_ids": _unique_strings(summary.get("lesson_ids", ()) for summary in summaries),
        "lesson_titles": _unique_strings(summary.get("lesson_titles", ()) for summary in summaries),
        "cookbook_methods": _unique_strings(
            [summary.get("cookbook_method")] for summary in summaries
        ),
        "data_contracts": _unique_strings(summary.get("data_contracts", ()) for summary in summaries),
        "unresolved_primitives": _unique_strings(
            summary.get("unresolved_primitives", ()) for summary in summaries
        ),
    }


def _aggregate_token_usage(method_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge token-usage summaries across method-level build payloads."""
    aggregate = {
        "call_count": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {},
        "by_provider": {},
    }

    def _merge(bucket: dict[str, Any], incoming: dict[str, Any]) -> None:
        for key in (
            "call_count",
            "calls_with_usage",
            "calls_without_usage",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            bucket[key] += int(incoming.get(key, 0) or 0)

    for payload in method_payloads.values():
        summary = payload.get("token_usage_summary") or {}
        if not summary:
            continue
        _merge(aggregate, summary)
        for group_name in ("by_stage", "by_provider"):
            for group_key, group_summary in (summary.get(group_name) or {}).items():
                bucket = aggregate[group_name].setdefault(
                    group_key,
                    {
                        "call_count": 0,
                        "calls_with_usage": 0,
                        "calls_without_usage": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                _merge(bucket, group_summary)
    return aggregate


def _unique_strings(values) -> list[str]:
    """Flatten nested iterables/scalars into a sorted unique string list."""
    flattened: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value:
                flattened.append(value)
            continue
        if isinstance(value, (list, tuple, set)):
            flattened.extend(item for item in value if isinstance(item, str) and item)
    return sorted(set(flattened))


def _cross_validate_comparison_task(
    comparison_targets: list[ComparisonBuildTarget],
    live_results: dict[str, Any],
    market_state,
    *,
    configured_targets: dict[str, Any],
    payoff_factory: Callable[[type, object, date], Any] | None = None,
    price_fn: Callable[[Any, Any], float] | None = None,
) -> dict[str, Any]:
    """Instantiate and compare prices across successful comparison builds."""
    from trellis.engine.payoff_pricer import price_payoff

    custom_payoff_factory = payoff_factory is not None
    payoff_factory = payoff_factory or _make_test_payoff
    price_fn = price_fn or price_payoff
    settle = getattr(market_state, "settlement", DEFAULT_SETTLEMENT)

    priced: dict[str, float] = {}
    price_errors: dict[str, str] = {}
    for target in comparison_targets:
        result = live_results.get(target.target_id)
        if result is None or not result.success or result.payoff_cls is None:
            continue

        try:
            spec_schema = None
            if not custom_payoff_factory:
                spec_schema = _infer_cached_spec_schema(result.payoff_cls, target.target_id)
                if spec_schema is None:
                    spec_schema = _infer_spec_schema_from_module(
                        import_module(result.payoff_cls.__module__),
                        result.payoff_cls,
                    )
                if spec_schema is None:
                    raise ValueError("Could not infer spec schema for comparison build")
            payoff = payoff_factory(result.payoff_cls, spec_schema, settle)
            priced[target.target_id] = float(price_fn(payoff, market_state))
        except Exception as exc:
            price_errors[target.target_id] = str(exc)

    reference_target = next(
        (target.target_id for target in comparison_targets if target.is_reference and target.target_id in priced),
        None,
    )
    reference_price = priced.get(reference_target) if reference_target is not None else None

    comparable_targets = [
        target.target_id
        for target in comparison_targets
        if not target.is_reference and target.target_id in priced
    ]
    if reference_target is None and len(comparable_targets) >= 2:
        reference_price = median(priced[target_id] for target_id in comparable_targets)
        reference_target = "median_internal"

    tolerance_pct = float(configured_targets.get("tolerance_pct", 5.0))
    deviations: dict[str, float] = {}
    passed_targets: list[str] = []
    failed_targets: list[str] = []

    if reference_price is not None:
        denominator = max(abs(reference_price), 1e-12)
        for target_id in comparable_targets:
            deviation_pct = abs(priced[target_id] - reference_price) / denominator * 100.0
            deviations[target_id] = round(deviation_pct, 4)
            if deviation_pct <= tolerance_pct:
                passed_targets.append(target_id)
            else:
                failed_targets.append(target_id)

    if reference_price is None and len(priced) < 2:
        status = "insufficient_results"
    elif price_errors:
        status = "pricing_error"
    elif failed_targets:
        status = "failed"
    else:
        status = "passed"

    return {
        "status": status,
        "configured_targets": configured_targets,
        "prices": priced,
        "price_errors": price_errors,
        "reference_target": reference_target,
        "reference_price": round(reference_price, 10) if reference_price is not None else None,
        "tolerance_pct": tolerance_pct,
        "deviations_pct": deviations,
        "passed_targets": passed_targets,
        "failed_targets": failed_targets,
        "successful_targets": [target_id for target_id in [target.target_id for target in comparison_targets] if target_id in priced],
    }


def prepare_existing_task(task: dict, *, model: str = "gpt-5-mini") -> PreparedTask:
    """Resolve a task to an existing generated payoff class without rebuilding."""
    description = task_to_description(task)
    instrument_type = task_to_instrument_type(task)

    if instrument_type is None:
        generic_plan = plan_build(description, set(), model=model)
        generic_payoff_cls = _try_import_existing(generic_plan)
        if generic_payoff_cls is None:
            generic_payoff_cls = _load_payoff_class_from_plan(generic_plan)
        generic_schema = None
        if generic_payoff_cls is not None:
            generic_schema = _infer_cached_spec_schema(generic_payoff_cls, task["title"])

        if generic_payoff_cls is None or generic_schema is None:
            generic_payoff_cls, generic_schema = _load_generic_fallback_cached_agent(task["title"])

        if generic_payoff_cls is None or generic_schema is None:
            raise ValueError(
                f"Offline benchmark requires a deterministic instrument type for {task['id']}"
            )

        return PreparedTask(
            task_id=task["id"],
            title=task["title"],
            description=description,
            instrument_type="generic",
            requirements=set(),
            payoff_cls=generic_payoff_cls,
            spec_schema=generic_schema,
        )

    pricing_plan = select_pricing_method(
        description,
        instrument_type=instrument_type,
        model=model,
    )
    preferred_method = getattr(pricing_plan, "method", None)
    try:
        plan = plan_build(
            description,
            pricing_plan.required_market_data,
            model=model,
            instrument_type=instrument_type,
            preferred_method=preferred_method,
        )
    except TypeError as exc:
        if "preferred_method" not in str(exc):
            raise
        plan = plan_build(
            description,
            pricing_plan.required_market_data,
            model=model,
            instrument_type=instrument_type,
        )
    payoff_cls = _try_import_existing(plan)
    spec_schema = plan.spec_schema

    if payoff_cls is not None and spec_schema is None:
        inferred_schema = _infer_cached_spec_schema(payoff_cls, task["title"])
        if inferred_schema is not None:
            spec_schema = inferred_schema

    if payoff_cls is None or spec_schema is None:
        payoff_cls, spec_schema = _load_fallback_cached_agent(instrument_type, task["title"])

    if payoff_cls is None or spec_schema is None:
        raise FileNotFoundError(
            f"No cached agent module found for {task['id']} ({instrument_type})"
        )

    return PreparedTask(
        task_id=task["id"],
        title=task["title"],
        description=description,
        instrument_type=instrument_type,
        requirements=set(pricing_plan.required_market_data),
        payoff_cls=payoff_cls,
        spec_schema=spec_schema,
    )


def benchmark_existing_task(
    task: dict,
    *,
    market_state,
    repeats: int = 5,
    warmups: int = 1,
    model: str = "gpt-5-mini",
    timer: Callable[[], float] = perf_counter,
    price_fn: Callable[[Any, Any], float] | None = None,
) -> dict:
    """Benchmark pricing runtime for an existing generated payoff class."""
    from trellis.engine.payoff_pricer import price_payoff

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmups < 0:
        raise ValueError("warmups must be >= 0")

    price_fn = price_fn or price_payoff
    prepared = prepare_existing_task(task, model=model)
    settle = getattr(market_state, "settlement", DEFAULT_SETTLEMENT)

    t0 = timer()
    payoff = _make_test_payoff(prepared.payoff_cls, prepared.spec_schema, settle)
    instantiate_seconds = timer() - t0

    for _ in range(warmups):
        price_fn(payoff, market_state)

    durations: list[float] = []
    last_price: float | None = None
    for _ in range(repeats):
        start = timer()
        last_price = price_fn(payoff, market_state)
        durations.append(timer() - start)

    return {
        "task_id": prepared.task_id,
        "title": prepared.title,
        "instrument_type": prepared.instrument_type,
        "payoff_class": prepared.payoff_cls.__name__,
        "warmups": warmups,
        "repeats": repeats,
        "instantiate_seconds": round(instantiate_seconds, 6),
        "mean_seconds": round(mean(durations), 6),
        "min_seconds": round(min(durations), 6),
        "max_seconds": round(max(durations), 6),
        "last_price": last_price,
    }


def _load_fallback_cached_agent(
    instrument_type: str,
    task_title: str,
) -> tuple[type | None, SpecSchema | None]:
    """Load an explicitly mapped cached agent payoff when planner metadata is too weak."""
    fallback = _FALLBACK_AGENT_MODULES.get(instrument_type)
    if fallback is None:
        return None, None

    module_name, payoff_class_name = fallback
    try:
        module = import_module(module_name)
    except Exception:
        return None, None

    payoff_cls = getattr(module, payoff_class_name, None)
    if payoff_cls is None:
        return None, None

    return payoff_cls, _infer_spec_schema_from_module(module, payoff_cls)


def _load_generic_fallback_cached_agent(task_title: str) -> tuple[type | None, SpecSchema | None]:
    """Load a checked-in cached generic payoff for known benchmark tasks."""
    title_lower = task_title.lower()
    for tokens, module_name, payoff_class_name in _GENERIC_FALLBACK_AGENT_MODULES:
        if not all(token in title_lower for token in tokens):
            continue
        try:
            module = import_module(module_name)
        except Exception:
            return None, None
        payoff_cls = getattr(module, payoff_class_name, None)
        if payoff_cls is None:
            return None, None
        return payoff_cls, _infer_spec_schema_from_module(module, payoff_cls)
    return None, None


def _load_payoff_class_from_plan(plan) -> type | None:
    """Import a cached agent module directly from the planner path and find its payoff class."""
    step = plan.steps[0] if getattr(plan, "steps", None) else None
    if step is None:
        return None

    module_path = getattr(step, "module_path", "")
    if not module_path:
        return None

    module_name = f"trellis.{module_path.replace('/', '.').replace('.py', '')}"
    try:
        module = import_module(module_name)
    except Exception:
        return None

    return _find_cached_payoff_class(module)


def _infer_cached_spec_schema(payoff_cls: type, task_title: str) -> SpecSchema | None:
    """Infer a spec schema from a cached generic module if it matches the task."""
    try:
        module = import_module(payoff_cls.__module__)
    except Exception:
        return None

    if not _module_matches_task(module, task_title):
        return None

    return _infer_spec_schema_from_module(module, payoff_cls)


def _module_matches_task(module, task_title: str) -> bool:
    """Check whether a cached generic module appears to correspond to the task."""
    doc = (getattr(module, "__doc__", "") or "").lower()
    title = task_title.lower()
    if title in doc:
        return True

    tokens = [
        token.strip("():,-")
        for token in title.split()
        if token.strip("():,-") and token.strip("():,-") not in _GENERIC_TITLE_STOPWORDS
    ]
    shared = sum(1 for token in tokens if token in doc)
    return shared >= min(3, len(tokens)) if tokens else False


def _infer_spec_schema_from_module(module, payoff_cls: type) -> SpecSchema | None:
    """Infer a planner-compatible spec schema from a cached module dataclass."""
    spec_cls = None
    for value in module.__dict__.values():
        if isinstance(value, type) and is_dataclass(value) and value.__name__.endswith("Spec"):
            spec_cls = value
            break

    if spec_cls is None:
        return None

    field_defs = []
    for field in fields(spec_cls):
        field_defs.append(
            FieldDef(
                name=field.name,
                type=_annotation_to_field_type(field.type),
                description="",
                default=None if field.default is MISSING and field.default_factory is MISSING else repr(
                    field.default if field.default is not MISSING else None
                ),
            )
        )

    return SpecSchema(
        class_name=payoff_cls.__name__,
        spec_name=spec_cls.__name__,
        requirements=[],
        fields=field_defs,
    )


def _find_cached_payoff_class(module) -> type | None:
    """Find the first non-dataclass payoff-like class in a cached module."""
    for value in module.__dict__.values():
        if not isinstance(value, type):
            continue
        if is_dataclass(value):
            continue
        if callable(getattr(value, "evaluate", None)):
            return value
    return None


def _annotation_to_field_type(annotation) -> str:
    """Map Python/dataclass annotations to planner field type strings."""
    if isinstance(annotation, str):
        return annotation if annotation in {
            "float",
            "int",
            "str",
            "bool",
            "date",
            "str | None",
            "Frequency",
            "DayCountConvention",
        } else "str"

    if annotation is float:
        return "float"
    if annotation is int:
        return "int"
    if annotation is str:
        return "str"
    if annotation is bool:
        return "bool"
    if annotation is date:
        return "date"

    annotation_name = getattr(annotation, "__name__", None)
    if annotation_name in {"Frequency", "DayCountConvention"}:
        return annotation_name

    return "str"
