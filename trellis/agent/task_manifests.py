"""Task-manifest loading helpers for benchmark, extension, negative, legacy, and canary corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]

FRAMEWORK_TASKS_MANIFEST = "FRAMEWORK_TASKS.yaml"
CANARY_TASKS_MANIFEST = "CANARY_TASKS.yaml"
NEGATIVE_TASKS_MANIFEST = "TASKS_NEGATIVE.yaml"
MARKET_SCENARIOS_MANIFEST = "MARKET_SCENARIOS.yaml"
FINANCEPY_BINDINGS_MANIFEST = "FINANCEPY_BINDINGS.yaml"

PRICING_TASK_CORPORA: tuple[str, ...] = (
    "TASKS_BENCHMARK_FINANCEPY.yaml",
    "TASKS_EXTENSION.yaml",
    "TASKS_MARKET_CONSTRUCTION.yaml",
    "TASKS_PROOF_LEGACY.yaml",
)


def load_pricing_tasks(
    *,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    """Load the aggregated priceable-task surface across the new corpora."""
    loaded: list[dict[str, Any]] = []
    for manifest_name in PRICING_TASK_CORPORA:
        loaded.extend(load_task_manifest(manifest_name, root=root))
    return loaded


def load_negative_tasks(*, root: Path = ROOT) -> list[dict[str, Any]]:
    """Load the clarification / honest-block task corpus."""
    return load_task_manifest(NEGATIVE_TASKS_MANIFEST, root=root)


def load_framework_tasks(*, root: Path = ROOT) -> list[dict[str, Any]]:
    """Load framework/meta tasks."""
    return load_task_manifest(FRAMEWORK_TASKS_MANIFEST, root=root)


def load_active_task_lookup(*, root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Load all active task corpora keyed by task id."""
    task_lookup: dict[str, dict[str, Any]] = {}
    for task in load_pricing_tasks(root=root):
        task_id = str(task.get("id") or "").strip()
        if task_id:
            task_lookup[task_id] = dict(task)
    for task in load_negative_tasks(root=root):
        task_id = str(task.get("id") or "").strip()
        if task_id:
            task_lookup[task_id] = dict(task)
    return task_lookup


def load_canary_manifest(
    *,
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the curated canary manifest and normalize its metadata."""
    payload = _load_yaml_mapping(root / CANARY_TASKS_MANIFEST)
    canary_set = payload.get("canary_set") or ()
    canaries = [dict(entry) for entry in canary_set if isinstance(entry, Mapping)]
    meta = {
        "version": int(payload.get("version") or 1),
        "total_budget_usd": float(payload.get("total_budget_usd") or 0.0),
        "refresh_cadence": str(payload.get("refresh_cadence") or ""),
        "source_corpora": tuple(
            str(item).strip()
            for item in (payload.get("source_corpora") or ())
            if str(item).strip()
        ),
    }
    return canaries, meta


def load_canary_task_lookup(*, root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Materialize the live task payload for each canary id."""
    task_lookup = load_active_task_lookup(root=root)
    canaries, _ = load_canary_manifest(root=root)
    return {
        canary["id"]: dict(task_lookup[canary["id"]])
        for canary in canaries
        if str(canary.get("id") or "").strip() in task_lookup
    }


def load_market_scenarios(*, root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Load the canonical market-scenario registry keyed by scenario id."""
    payload = _load_yaml_mapping(root / MARKET_SCENARIOS_MANIFEST)
    scenarios = payload.get("scenarios") or {}
    return {
        str(scenario_id): dict(data)
        for scenario_id, data in dict(scenarios).items()
        if str(scenario_id).strip() and isinstance(data, Mapping)
    }


def load_financepy_bindings(*, root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Load the canonical FinancePy binding registry keyed by binding id."""
    payload = _load_yaml_mapping(root / FINANCEPY_BINDINGS_MANIFEST)
    bindings = payload.get("bindings") or {}
    return {
        str(binding_id): dict(data)
        for binding_id, data in dict(bindings).items()
        if str(binding_id).strip() and isinstance(data, Mapping)
    }


def load_task_manifest(
    manifest_name: str,
    *,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    """Load one task manifest and normalize its task-level metadata."""
    path = root / manifest_name
    if not path.exists():
        return []

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if isinstance(raw, list):
        tasks = raw
        version = 1
    elif isinstance(raw, Mapping):
        tasks = list(raw.get("tasks") or ())
        version = int(raw.get("version") or 1)
    else:
        return []

    corpus_name = _manifest_to_corpus_name(manifest_name)
    scenarios = _load_market_scenarios_mapping(root)
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        payload = dict(task)
        payload.setdefault("task_corpus", corpus_name)
        payload.setdefault("task_definition_version", version)
        payload.setdefault("task_definition_manifest", manifest_name)
        payload.setdefault("market", _materialize_market_from_scenario(payload, scenarios))
        normalized.append(payload)
    return normalized


def filter_loaded_tasks(
    tasks: Sequence[dict[str, Any]],
    start_id: str | None = None,
    end_id: str | None = None,
    *,
    status: str | None = "pending",
) -> list[dict[str, Any]]:
    """Apply the legacy id-range/status filter semantics across a normalized task list."""
    filtered = list(tasks)

    if status is not None:
        filtered = [task for task in filtered if task.get("status") == status]

    if start_id and end_id:
        start_num = int(start_id.lstrip("TEFPN"))
        end_num = int(end_id.lstrip("TEFPN"))
        prefix = start_id[0]
        filtered = [
            task
            for task in filtered
            if str(task.get("id") or "").startswith(prefix)
            and start_num <= int(str(task["id"]).lstrip("TEFPN")) <= end_num
        ]

    return filtered


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def _manifest_to_corpus_name(manifest_name: str) -> str:
    stem = Path(manifest_name).stem.lower()
    return stem.replace("tasks_", "").replace("task_", "")


def _load_market_scenarios_mapping(root: Path) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_mapping(root / MARKET_SCENARIOS_MANIFEST)
    scenarios = payload.get("scenarios") or {}
    return {
        str(scenario_id): dict(data)
        for scenario_id, data in dict(scenarios).items()
        if str(scenario_id).strip() and isinstance(data, Mapping)
    }


def _materialize_market_from_scenario(
    task: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(task.get("market"), Mapping):
        return dict(task.get("market") or {})
    scenario_id = str(task.get("market_scenario_id") or "").strip()
    if not scenario_id:
        return {}
    scenario = dict(scenarios.get(scenario_id) or {})
    selected = dict(scenario.get("selected_components") or {})
    if not scenario:
        return {}
    market = {
        "source": scenario.get("source"),
        "as_of": scenario.get("as_of"),
        **selected,
    }
    benchmark_inputs = scenario.get("benchmark_inputs")
    if isinstance(benchmark_inputs, Mapping) and benchmark_inputs:
        market["benchmark_inputs"] = dict(benchmark_inputs)
    description = str(scenario.get("description") or "").strip()
    if description:
        market["scenario_description"] = description
    return market
