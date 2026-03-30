"""Helpers for running pricing tasks without an LLM and benchmarking results.

Loads task definitions from TASKS.yaml, resolves their generated payoff
modules, and runs them against market data to measure correctness and timing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time
from typing import Any, Callable, Mapping

_log = logging.getLogger(__name__)

from trellis.agent.executor import _make_test_payoff, _try_import_existing
from trellis.agent.knowledge.methods import is_known_method, normalize_method
from trellis.agent.planner import FieldDef, SpecSchema
from trellis.agent.planner import plan_build
from trellis.agent.platform_requests import compile_build_request
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


def _selected_curve_names_from_market_state(market_state) -> dict[str, str]:
    """Return resolved curve-name provenance from a compiled market state."""
    names = getattr(market_state, "selected_curve_names", None)
    if not isinstance(names, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in names.items()
        if value is not None
    }


def _task_requires_credit_curve(task: dict) -> bool:
    """Return True when the task contract needs a credit curve to be usable."""
    construct_methods = set(_task_construct_methods(task))
    if "credit" in construct_methods:
        return True
    if task_to_instrument_type(task) == "nth_to_default":
        return True
    title = " ".join(
        part for part in (
            str(task.get("title") or ""),
            str(task.get("description") or ""),
        )
        if part
    ).lower()
    return "cds" in title


def _inject_default_credit_curve_for_task(task: dict, market_state):
    """Attach a default credit curve when the task contract requires one."""
    if getattr(market_state, "credit_curve", None) is not None:
        return market_state
    if not _task_requires_credit_curve(task):
        return market_state

    from trellis.curves.credit_curve import CreditCurve

    selected_curve_names = dict(getattr(market_state, "selected_curve_names", None) or {})
    selected_curve_names["credit_curve"] = "default_flat_credit_curve_2pct"
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    market_provenance["credit_curve"] = {
        "source": "runtime_default",
        "hazard_rate": 0.02,
        "reason": "credit task without explicit market spec",
    }
    return replace(
        market_state,
        credit_curve=CreditCurve.flat(0.02),
        selected_curve_names=selected_curve_names,
        market_provenance=market_provenance,
    )


@dataclass(frozen=True)
class PreparedTask:
    """A task that can be priced without an LLM because its payoff module already exists."""

    task_id: str
    title: str
    description: str
    instrument_type: str
    requirements: set[str]
    payoff_cls: type
    spec_schema: object
    compiled_request: object | None = None


@dataclass(frozen=True)
class ComparisonBuildTarget:
    """One pricing method to build and evaluate when a task compares multiple methods side by side."""

    target_id: str
    preferred_method: str
    is_reference: bool = False


@dataclass(frozen=True)
class TaskContractError(ValueError):
    """Raised when a task definition is incompatible with the automated pricing runner."""

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
        market_state = _inject_default_credit_curve_for_task(task, market_state)
        selected_curve_names = _selected_curve_names_from_market_state(market_state)
        market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
        return market_state, {
            "source": "default",
            "as_of": getattr(market_state, "as_of", None).isoformat() if getattr(market_state, "as_of", None) else None,
            "selected_components": {},
            "selected_curve_names": selected_curve_names,
            "available_capabilities": sorted(getattr(market_state, "available_capabilities", ())),
            "metadata": {},
            "provenance": market_provenance,
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
    had_credit_curve_before = getattr(market_state, "credit_curve", None) is not None
    market_state = _inject_default_credit_curve_for_task(task, market_state)
    selected_curve_names = _selected_curve_names_from_market_state(market_state)
    # Add credit_curve to selected_components only when it was runtime-injected for
    # this task (i.e. the task is a credit task and the snapshot had no credit_curve).
    # When the snapshot already had a credit_curve it is part of the snapshot's
    # ambient capabilities, not a task-specific selection.
    if not had_credit_curve_before and getattr(market_state, "credit_curve", None) is not None:
        selected_components["credit_curve"] = market_state.credit_curve
    market_provenance = dict(
        getattr(market_state, "market_provenance", None) or getattr(snapshot, "provenance", None) or {}
    )

    market_context = {
        "source": snapshot.source,
        "as_of": snapshot.as_of.isoformat(),
        "selected_components": selected_components,
        "selected_curve_names": selected_curve_names,
        "available_capabilities": sorted(market_state.available_capabilities),
        "metadata": dict(snapshot.metadata),
        "provenance": market_provenance,
    }
    _validate_task_market_assertions(task, market_context)
    return market_state, market_context


def task_to_description(task: dict) -> str:
    """Convert a ``TASKS.yaml`` entry into a pricing-build request string."""
    description = f"Build a pricer for: {task['title']}"
    extra = str(task.get("description") or "").strip()
    if extra:
        return f"{description}\n\n{extra}"
    return description


def _bootstrap_ranked_observation_basket_description(task: dict) -> str | None:
    """Return the canonical proving-case prompt for sparse basket tasks."""
    if str(task.get("description") or "").strip():
        return None

    title = " ".join(
        part for part in (
            str(task.get("title") or ""),
            str(task.get("description") or ""),
        )
        if part
    ).lower()
    if not any(
        cue in title
        for cue in (
            "himalaya",
            "ranked observation basket",
            "ranked observation",
            "remaining constituents",
        )
    ):
        return None

    return (
        "Build a pricer for: Himalaya ranked observation basket\n\n"
        "AAPL, MSFT, and NVDA with observation dates 2025-01-15, "
        "2025-02-15, 2025-03-15. At each observation choose the best "
        "performer among remaining constituents, remove it, lock the "
        "simple return, and settle the average locked returns at maturity."
    )


def _effective_task_description(task: dict) -> str:
    """Return the task description after applying any canonical bootstrap prompt."""
    description = _bootstrap_ranked_observation_basket_description(task) or task_to_description(task)
    context_lines: list[str] = []

    construct_methods = _task_construct_methods(task)
    if construct_methods:
        context_lines.append(f"Construct methods: {', '.join(construct_methods)}")

    comparison_targets = _task_comparison_targets(task, construct_methods)
    if comparison_targets:
        target_lines: list[str] = []
        for target in comparison_targets:
            if target.preferred_method:
                target_lines.append(f"{target.target_id} ({target.preferred_method})")
            else:
                target_lines.append(target.target_id)
        context_lines.append(f"Comparison targets: {', '.join(target_lines)}")

    cross_validate = task.get("cross_validate") or {}
    internal_targets = [str(target) for target in (cross_validate.get("internal") or ())]
    analytical_target = cross_validate.get("analytical")
    external_targets = [str(target) for target in (cross_validate.get("external") or ())]
    if internal_targets or analytical_target or external_targets:
        context_lines.append("Cross-validation harness:")
        if internal_targets:
            context_lines.append(f"  internal targets: {', '.join(internal_targets)}")
        if analytical_target:
            context_lines.append(f"  analytical benchmark: {analytical_target}")
        if external_targets:
            context_lines.append(f"  external targets: {', '.join(external_targets)}")

    new_component = str(task.get("new_component") or "").strip()
    if new_component:
        context_lines.append(f"New component: {new_component}")

    if not context_lines:
        return description
    return f"{description}\n\n" + "\n".join(context_lines)


def task_to_instrument_type(task: dict) -> str | None:
    """Heuristically resolve the most likely instrument type for a task."""
    title = " ".join(
        part for part in (
            str(task.get("title") or ""),
            str(task.get("description") or ""),
        )
        if part
    ).lower()
    mappings = [
        ("himalaya", "basket_option"),
        ("ranked observation", "basket_option"),
        ("remaining constituents", "basket_option"),
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
        ("cds", "credit_default_swap"),
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
        ("quanto", "quanto_option"),
        ("forward start", "european_option"),
        ("vanilla option", "european_option"),
        ("vanilla", "european_option"),
        ("american", "american_option"),
        ("european", "european_option"),
        ("fx", "european_option"),
        ("swap", "swap"),
        ("bond", "bond"),
    ]
    for pattern, instrument_type in mappings:
        if pattern in title:
            return instrument_type
    return None


def task_to_semantic_contract(task: dict):
    """Draft the canonical semantic contract for a semantic basket request, if any."""
    from trellis.agent.semantic_contracts import draft_semantic_contract

    try:
        return draft_semantic_contract(
            _effective_task_description(task),
            instrument_type=task_to_instrument_type(task),
        )
    except ValueError:
        return None


def _runtime_snapshot_reference(market_context: dict[str, Any]) -> dict[str, Any]:
    """Return a stable snapshot reference for runtime replay metadata."""
    return {
        "source": market_context.get("source"),
        "as_of": market_context.get("as_of"),
        "selected_components": dict(market_context.get("selected_components") or {}),
        "selected_curve_names": dict(market_context.get("selected_curve_names") or {}),
        "available_capabilities": list(market_context.get("available_capabilities") or ()),
        "metadata": dict(market_context.get("metadata") or {}),
    }


def _explicit_simulation_seed(task: dict, market_context: dict[str, Any]) -> tuple[int | None, str]:
    """Return an explicit simulation seed if the task or market metadata provided one."""
    metadata = dict(market_context.get("metadata") or {})
    seed_candidates = (
        ("task.simulation_seed", task.get("simulation_seed")),
        ("task.seed", task.get("seed")),
        ("market.metadata.simulation_seed", metadata.get("simulation_seed")),
        ("market.metadata.seed", metadata.get("seed")),
    )
    for source, value in seed_candidates:
        if value is None or value == "":
            continue
        try:
            return int(value), source
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid simulation seed from {source}: {value!r}") from exc
    return None, ""


def _stable_seed(payload: dict[str, Any]) -> int:
    """Derive a deterministic non-negative simulation seed from stable metadata."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(encoded, digest_size=8).digest(), "big")


def _runtime_simulation_identity(
    task: dict,
    *,
    semantic_contract,
    market_context: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic Monte Carlo replay metadata for the current task slice."""
    snapshot_reference = _runtime_snapshot_reference(market_context)
    evaluation_tags = _runtime_evaluation_tags(
        task,
        semantic_contract=semantic_contract,
        market_context=market_context,
    )
    semantic_id = getattr(getattr(semantic_contract, "product", None), "semantic_id", None)
    base_payload = {
        "task_id": task["id"],
        "task_title": task["title"],
        "description": task_to_description(task),
        "instrument_type": task_to_instrument_type(task),
        "semantic_contract_id": semantic_id,
        "snapshot_reference": snapshot_reference,
        "evaluation_tags": evaluation_tags,
    }
    explicit_seed, seed_source = _explicit_simulation_seed(task, market_context)
    if explicit_seed is None:
        explicit_seed = _stable_seed(base_payload)
        seed_source = "derived_from_request_and_snapshot"
    stream_digest = hashlib.blake2s(
        json.dumps(
            {**base_payload, "seed": explicit_seed},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    sample_source = {
        "kind": "market_snapshot",
        "source": snapshot_reference.get("source"),
        "as_of": snapshot_reference.get("as_of"),
        "snapshot_reference": snapshot_reference,
    }
    sample_indexing = {
        "kind": "path_index",
        "ordering": "simulation_generation_order",
        "start": 0,
    }
    return {
        "seed": explicit_seed,
        "seed_source": seed_source,
        "sample_source": sample_source,
        "sample_indexing": sample_indexing,
        "simulation_stream_id": f"{task['id']}:{stream_digest}",
        "replay_key": f"{task['id']}:{stream_digest}",
    }


def _runtime_evaluation_tags(
    task: dict,
    *,
    semantic_contract,
    market_context: dict[str, Any],
) -> tuple[str, ...]:
    """Return deterministic tags that identify one runtime/eval slice."""
    tags: list[str] = ["task_runtime"]
    construct = task.get("construct")
    if isinstance(construct, (list, tuple)):
        construct_text = ",".join(
            str(item).strip().lower()
            for item in construct
            if str(item).strip()
        )
    else:
        construct_text = str(construct or "").strip().lower()
    if construct_text:
        tags.append(f"construct:{construct_text}")
    source = str(market_context.get("source") or "").strip().lower()
    if source:
        tags.append(f"market:{source}")
    if task.get("cross_validate"):
        tags.append("comparison")
    if semantic_contract is not None:
        tags.append("semantic_contract")
        semantic_id = getattr(getattr(semantic_contract, "product", None), "semantic_id", None)
        if semantic_id:
            tags.append(f"semantic:{semantic_id}")
    return tuple(dict.fromkeys(tags))


def _runtime_contract_metadata(
    task: dict,
    *,
    description: str,
    instrument_type: str | None,
    semantic_contract,
    market_context: dict[str, Any],
    trace_identifier: str | None = None,
    trace_path: str | None = None,
) -> dict[str, Any]:
    """Build the runtime contract payload carried through request and result metadata."""
    from trellis.agent.semantic_contracts import semantic_contract_summary

    summary = semantic_contract_summary(semantic_contract) if semantic_contract is not None else None
    simulation_identity = _runtime_simulation_identity(
        task,
        semantic_contract=semantic_contract,
        market_context=market_context,
    )
    runtime_contract: dict[str, Any] = {
        "task_id": task["id"],
        "task_title": task["title"],
        "description": description,
        "instrument_type": instrument_type,
        "semantic_contract": summary,
        "semantic_contract_id": getattr(getattr(semantic_contract, "product", None), "semantic_id", None),
        "snapshot_reference": _runtime_snapshot_reference(market_context),
        "selected_curve_names": dict(market_context.get("selected_curve_names") or {}),
        "market_provenance": dict(market_context.get("provenance") or {}),
        "evaluation_tags": _runtime_evaluation_tags(
            task,
            semantic_contract=semantic_contract,
            market_context=market_context,
        ),
        "simulation_identity": simulation_identity,
        "simulation_seed": simulation_identity["seed"],
        "sample_source": simulation_identity["sample_source"],
        "sample_indexing": simulation_identity["sample_indexing"],
        "simulation_stream_id": simulation_identity["simulation_stream_id"],
        "replay_key": simulation_identity["replay_key"],
    }
    if trace_identifier is not None:
        runtime_contract["trace_identifier"] = trace_identifier
    if trace_path is not None:
        runtime_contract["trace_path"] = trace_path
    return runtime_contract


def _trace_observability(trace_path: str | None) -> dict[str, Any]:
    """Extract replay-relevant guardrail status from a persisted platform trace."""
    if not trace_path:
        return {}

    path = Path(trace_path)
    if not path.exists():
        return {"trace_path": str(path)}

    try:
        import yaml

        trace = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {"trace_path": str(path)}

    observability: dict[str, Any] = {
        "trace_path": str(path),
        "trace_status": trace.get("status"),
        "trace_outcome": trace.get("outcome"),
    }
    details = dict(trace.get("details") or {})
    candidates: list[dict[str, Any]] = [details]
    for item in trace.get("events", []):
        if not isinstance(item, dict):
            continue
        candidates.append(item)
        nested_details = item.get("details")
        if isinstance(nested_details, dict):
            candidates.append(nested_details)
    for candidate in candidates:
        source_sanitization = candidate.get("source_sanitization")
        if isinstance(source_sanitization, dict):
            observability["source_sanitization"] = source_sanitization
            observability["source_status"] = source_sanitization.get("source_status")
        parse_status = candidate.get("parse_status")
        if parse_status:
            observability["parse_status"] = parse_status
        correlation_preflight = candidate.get("correlation_preflight")
        if isinstance(correlation_preflight, dict):
            observability["correlation_preflight"] = correlation_preflight
            observability["correlation_status"] = correlation_preflight.get(
                "correlation_status"
            )
    return observability


def run_task(
    task: dict,
    market_state,
    *,
    model: str = "gpt-5-mini",
    force_rebuild: bool = True,
    fresh_build: bool = False,
    validation: str = "standard",
    max_retries: int = 3,
    build_fn: Callable[..., Any] | None = None,
    timer: Callable[[], float] = time,
    now_fn: Callable[[], datetime] = datetime.now,
    payoff_factory: Callable[[type, object, date], Any] | None = None,
    price_fn: Callable[[Any, Any], float] | None = None,
) -> dict:
    """Execute one task through the knowledge-aware build pipeline."""
    from trellis.agent.task_run_store import persist_task_run_record, summarize_task_learning

    if build_fn is None:
        from trellis.agent.knowledge.autonomous import build_with_knowledge

        build_fn = build_with_knowledge

    task_id = task["id"]
    description = _effective_task_description(task)
    instrument_type = task_to_instrument_type(task)
    semantic_contract = task_to_semantic_contract(task)
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
        "semantic_contract_id": getattr(getattr(semantic_contract, "product", None), "semantic_id", None),
    }

    try:
        _validate_task_contract(task, instrument_type, construct_methods)
        market_state, market_context = build_market_state_for_task(task, market_state)
        runtime_contract = _runtime_contract_metadata(
            task,
            description=description,
            instrument_type=instrument_type,
            semantic_contract=semantic_contract,
            market_context=market_context,
        )
        base_request_metadata = {
            "task_id": task_id,
            "task_title": task["title"],
            "runtime_contract": runtime_contract,
        }
        if semantic_contract is not None:
            from trellis.agent.semantic_contracts import semantic_contract_summary

            base_request_metadata["semantic_contract"] = semantic_contract_summary(semantic_contract)
        result_data["market_context"] = market_context
        result_data["runtime_contract"] = runtime_contract
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
                    "fresh_build": fresh_build,
                }
                result = build_fn(**build_kwargs)
                live_results[target.target_id] = result
                method_results[target.target_id] = _build_result_payload(
                    result,
                    preferred_method=target.preferred_method,
                    reference_target=target.is_reference,
                    task_kind="pricing",
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
            lesson_contracts = [
                payload["reflection"].get("lesson_contract")
                for payload in method_results.values()
                if payload["reflection"].get("lesson_contract") is not None
            ]
            lesson_promotion_outcomes = [
                payload["reflection"].get("lesson_promotion_outcome")
                for payload in method_results.values()
                if payload["reflection"].get("lesson_promotion_outcome") is not None
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
            _write_benchmark_sidecars(method_results, cross_validation)
            promotion_candidates: dict[str, str] = {}
            if (
                fresh_build
                and all(payload["success"] for payload in method_results.values())
                and cross_validation["status"] == "passed"
            ):
                promotion_candidates = _record_promotion_candidates(
                    task=task,
                    instrument_type=instrument_type,
                    market_context=market_context,
                    live_results=live_results,
                    method_results=method_results,
                    cross_validation=cross_validation,
                )
                for target_id, candidate_path in promotion_candidates.items():
                    payload = method_results.get(target_id)
                    if not payload or not candidate_path:
                        continue
                    payload.setdefault("reflection", {})["promotion_candidate_saved"] = candidate_path
                    payload["artifacts"] = _artifacts_from_payload(payload)

            reflection_payload = {
                "lesson_captured": lesson_ids,
                "lesson_contract": lesson_contracts,
                "lesson_promotion_outcome": lesson_promotion_outcomes,
                "cookbook_enriched": any(
                    payload["reflection"].get("cookbook_enriched")
                    for payload in method_results.values()
                ),
                "method_reflections": {
                    method: payload["reflection"]
                    for method, payload in method_results.items()
                },
            }
            if promotion_candidates:
                reflection_payload["promotion_candidate_saved"] = _unique_strings(
                    promotion_candidates.values()
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
                "reflection": reflection_payload,
                "cross_validation": cross_validation,
            })
            result_data["learning"] = summarize_task_learning(
                result_data,
                task_kind="pricing",
            )
            result_data["failures"] = _aggregate_failures(result_data)
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
                "fresh_build": fresh_build,
            }
            if preferred_method is not None:
                build_kwargs["preferred_method"] = preferred_method
            if task.get("cross_validate") and comparison_targets:
                build_kwargs["comparison_target"] = comparison_targets[0].target_id
            result = build_fn(**build_kwargs)
            elapsed = timer() - t0
            runtime_contract_result = dict(runtime_contract)
            runtime_contract_result["trace_identifier"] = getattr(result, "platform_request_id", None)
            runtime_contract_result["trace_path"] = getattr(result, "platform_trace_path", None)
            runtime_contract_result["analytical_trace_path"] = getattr(result, "analytical_trace_path", None)
            runtime_contract_result["analytical_trace_text_path"] = getattr(result, "analytical_trace_text_path", None)
            result_data.update({
                **_build_result_payload(result, preferred_method=preferred_method),
                "elapsed_seconds": round(elapsed, 1),
                "preferred_method": preferred_method,
                "runtime_contract": runtime_contract_result,
            })
            result_data["learning"] = summarize_task_learning(
                result_data,
                task_kind="pricing",
            )
            result_data["failures"] = _aggregate_failures(result_data)
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
        result_data["task_diagnosis_packet_path"] = persisted.get("diagnosis_packet_path")
        result_data["task_diagnosis_dossier_path"] = persisted.get("diagnosis_dossier_path")
        result_data["task_diagnosis_latest_packet_path"] = persisted.get("latest_diagnosis_packet_path")
        result_data["task_diagnosis_latest_dossier_path"] = persisted.get("latest_diagnosis_dossier_path")
        result_data["task_diagnosis_headline"] = persisted.get("diagnosis_headline")
        result_data["task_diagnosis_failure_bucket"] = persisted.get("diagnosis_failure_bucket")
        result_data["task_diagnosis_decision_stage"] = persisted.get("diagnosis_decision_stage")
        result_data["task_diagnosis_next_action"] = persisted.get("diagnosis_next_action")
        result_data["task_diagnosis_persist_error"] = persisted.get("diagnosis_persist_error")
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

    selected_curve_names = assertions.get("selected_curve_names") or {}
    actual_curve_names = market_context.get("selected_curve_names", {})
    for key, expected in selected_curve_names.items():
        actual = actual_curve_names.get(key)
        if actual != expected:
            errors.append(
                f"selected curve name mismatch for {key}: expected {expected!r}, got {actual!r}"
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
    task_kind: str = "pricing",
) -> dict[str, Any]:
    """Project a BuildResult-like object into a stable task result payload."""
    from trellis.agent.task_run_store import summarize_task_learning

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
        "analytical_trace_path": getattr(result, "analytical_trace_path", None),
        "analytical_trace_text_path": getattr(result, "analytical_trace_text_path", None),
        "build_observability": _trace_observability(
            getattr(result, "platform_trace_path", None)
        ),
        "blocker_details": getattr(result, "blocker_details", None),
                "reflection": {
                    key: value for key, value in result.reflection.items()
                    if key in (
                        "lessons_attributed",
                        "lesson_captured",
                        "lesson_contract",
                        "lesson_promotion_outcome",
                        "gaps_identified",
                        "cookbook_enriched",
                        "cookbook_candidate_saved",
                        "promotion_candidate_saved",
                        "knowledge_trace_saved",
                "knowledge_gap_log_saved",
                "decomposition_saved",
                "distill_run",
            )
        },
    }
    payload["artifacts"] = _artifacts_from_payload(payload)
    payload["learning"] = summarize_task_learning(payload, task_kind=task_kind)
    return payload


def _aggregate_failures(result: Mapping[str, Any]) -> list[str]:
    """Collect failure messages from the top-level result and nested method runs."""
    failures: list[str] = []

    def _append(value: Any, *, prefix: str | None = None) -> None:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return
            failures.append(f"{prefix}: {text}" if prefix else text)
            return
        if isinstance(value, Mapping):
            text = json.dumps(value, default=str, sort_keys=True)
            if text and text != "{}":
                failures.append(f"{prefix}: {text}" if prefix else text)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _append(item, prefix=prefix)

    _append(result.get("error"))
    _append(result.get("failures"))
    _append(result.get("blocker_details"))

    cross_validation = result.get("cross_validation")
    if isinstance(cross_validation, Mapping):
        status = str(cross_validation.get("status") or "").strip()
        if status and status != "passed":
            failures.append(f"cross_validation status: {status}")

    method_results = result.get("method_results")
    if isinstance(method_results, Mapping):
        for method_id, payload in method_results.items():
            if not isinstance(payload, Mapping):
                continue
            prefix = str(method_id)
            _append(payload.get("error"), prefix=prefix)
            _append(payload.get("failures"), prefix=prefix)
            _append(payload.get("blocker_details"), prefix=prefix)
            method_cross_validation = payload.get("cross_validation")
            if isinstance(method_cross_validation, Mapping):
                status = str(method_cross_validation.get("status") or "").strip()
                if status and status != "passed":
                    failures.append(f"{prefix}: cross_validation status: {status}")

    unique = _unique_strings(failures)
    if not unique and not bool(result.get("success")):
        return ["failure details unavailable"]
    return unique


def _artifacts_from_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Collect stable artifact references from one build payload."""
    reflection = payload.get("reflection") or {}
    return {
        "platform_request_ids": _unique_strings([payload.get("platform_request_id")]),
        "platform_trace_paths": _unique_strings([payload.get("platform_trace_path")]),
        "analytical_trace_paths": _unique_strings([payload.get("analytical_trace_path")]),
        "analytical_trace_text_paths": _unique_strings(
            [payload.get("analytical_trace_text_path")]
        ),
        "audit_record_paths": _unique_strings([payload.get("audit_record_path")]),
        "knowledge_trace_paths": _unique_strings([reflection.get("knowledge_trace_saved")]),
        "cookbook_candidate_paths": _unique_strings([reflection.get("cookbook_candidate_saved")]),
        "promotion_candidate_paths": _unique_strings([reflection.get("promotion_candidate_saved")]),
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
        "analytical_trace_paths": _unique_strings(
            payload.get("artifacts", {}).get("analytical_trace_paths", ())
            for payload in method_payloads.values()
        ),
        "analytical_trace_text_paths": _unique_strings(
            payload.get("artifacts", {}).get("analytical_trace_text_paths", ())
            for payload in method_payloads.values()
        ),
        "audit_record_paths": _unique_strings(
            payload.get("artifacts", {}).get("audit_record_paths", ())
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
        "promotion_candidate_paths": _unique_strings(
            payload.get("artifacts", {}).get("promotion_candidate_paths", ())
            for payload in method_payloads.values()
        ),
        "knowledge_gap_log_paths": _unique_strings(
            payload.get("artifacts", {}).get("knowledge_gap_log_paths", ())
            for payload in method_payloads.values()
        ),
    }


def _write_benchmark_sidecars(
    method_results: dict[str, dict[str, Any]],
    cross_validation: dict[str, Any],
) -> None:
    """Write benchmark sidecars alongside audit records after cross-validation. Non-blocking."""
    try:
        from pathlib import Path
        from trellis.agent.model_audit import write_benchmark_sidecar

        for payload in method_results.values():
            artifacts = dict(payload.get("artifacts") or {})
            for audit_path_str in artifacts.get("audit_record_paths") or []:
                if not audit_path_str:
                    continue
                audit_path = Path(audit_path_str)
                if not audit_path.exists():
                    continue
                try:
                    write_benchmark_sidecar(
                        audit_path,
                        comparison_status=str(cross_validation.get("status") or "unknown"),
                        prices=dict(cross_validation.get("prices") or {}),
                        deviations_pct=dict(cross_validation.get("deviations_pct") or {}),
                        reference_targets=list(
                            filter(None, [cross_validation.get("reference_target")])
                        ),
                    )
                except Exception:
                    pass
    except Exception:
        pass


def _record_promotion_candidates(
    *,
    task: dict[str, Any],
    instrument_type: str | None,
    market_context: dict[str, Any],
    live_results: dict[str, Any],
    method_results: dict[str, dict[str, Any]],
    cross_validation: dict[str, Any],
) -> dict[str, str]:
    """Persist successful fresh-build comparison routes as promotion candidates."""
    from trellis.agent.knowledge.promotion import record_promotion_candidate

    candidate_paths: dict[str, str] = {}
    for target_id, payload in method_results.items():
        live_result = live_results.get(target_id)
        if live_result is None or not payload.get("success"):
            continue
        code = str(getattr(live_result, "code", "") or "").strip()
        if not code:
            continue
        payoff_cls = getattr(live_result, "payoff_cls", None)
        try:
            path = record_promotion_candidate(
                task_id=str(task.get("id") or ""),
                task_title=str(task.get("title") or ""),
                instrument_type=instrument_type,
                comparison_target=target_id,
                preferred_method=payload.get("preferred_method"),
                payoff_class=getattr(payoff_cls, "__name__", None),
                module_path=getattr(payoff_cls, "__module__", None),
                code=code,
                attempts=int(getattr(live_result, "attempts", payload.get("attempts", 0)) or 0),
                platform_request_id=payload.get("platform_request_id"),
                platform_trace_path=payload.get("platform_trace_path"),
                market_context=dict(market_context or {}),
                cross_validation=dict(cross_validation or {}),
                reference_target=bool(payload.get("reference_target")),
            )
        except Exception:
            continue
        if path:
            candidate_paths[target_id] = path
    return candidate_paths


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


def _check_compiled_for_ambiguity(compiled: object | None, task: dict) -> None:
    """Log a WARNING if semantic analysis flagged this description as ambiguous.

    Does not raise — the cached payoff is still valid.  The task description
    should be clarified so future builds produce the correct instrument.
    """
    if compiled is None:
        return
    try:
        request = getattr(compiled, "request", None)
        metadata = getattr(request, "metadata", {}) or {}
        gap = metadata.get("semantic_gap") or {}
        if gap.get("requires_clarification"):
            summary = gap.get("summary", "ambiguous instrument type")
            _log.warning(
                "Task %s (%s) has an ambiguous description: %s. "
                "The cached payoff will be used, but the task description should be clarified.",
                task.get("id"),
                task.get("title"),
                summary,
            )
    except Exception as exc:
        _log.debug("_check_compiled_for_ambiguity: could not read metadata: %s", exc)


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
            compiled_request=None,
        )

    # Use compile_build_request for all instrument types so semantic
    # analysis (concept resolution, gap classification, ambiguity
    # detection) runs on the batch path — not just the NLP path.
    # Falls back to select_pricing_method only on expected registry/plan
    # misses; unexpected errors (RuntimeError, ImportError, …) propagate.
    try:
        compiled = compile_build_request(
            description,
            instrument_type=instrument_type,
            model=model,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        _log.warning(
            "compile_build_request degraded for task %s (%s): %s — "
            "falling back to select_pricing_method",
            task.get("id"),
            instrument_type,
            exc,
        )
        compiled = None

    if compiled is not None:
        pricing_plan = compiled.pricing_plan  # AttributeError here is a bug — propagate
        _check_compiled_for_ambiguity(compiled, task)
    else:
        pricing_plan = select_pricing_method(
            description,
            instrument_type=instrument_type,
            model=model,
        )
    if pricing_plan is None:
        raise ValueError(
            f"Could not determine a pricing plan for {task['id']} ({instrument_type})"
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
        compiled_request=compiled,
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
