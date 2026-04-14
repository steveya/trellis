"""Shared market-scenario contracts, constructors, and coverage helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from trellis.core.market_state import MarketState


ROOT = Path(__file__).resolve().parents[2]
MARKET_SCENARIOS_MANIFEST = "MARKET_SCENARIOS.yaml"


def _float_or_none(value: object | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _date_or_default(value: object | None, *, default: date) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    return date.fromisoformat(text)


def _string_mapping(payload: Mapping[str, object] | None) -> dict[str, str]:
    return {
        str(key).strip(): str(value).strip()
        for key, value in dict(payload or {}).items()
        if str(key).strip() and str(value).strip()
    }


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _flat_yield_curve(rate: float | None, *, value_date: date | None = None):
    if rate is None:
        return None
    if value_date is not None:
        from trellis.curves.date_aware_flat_curve import DateAwareFlatYieldCurve

        return DateAwareFlatYieldCurve(value_date=value_date, flat_rate=float(rate), max_tenor=31.0)
    from trellis.curves.yield_curve import YieldCurve

    return YieldCurve.flat(float(rate), max_tenor=31.0)


def _flat_credit_curve(hazard_rate: float | None):
    if hazard_rate is None:
        return None
    from trellis.curves.credit_curve import CreditCurve

    return CreditCurve.flat(float(hazard_rate), max_tenor=31.0)


@dataclass(frozen=True)
class ScenarioUnderlier:
    """One underlier declaration inside a benchmark/task market scenario."""

    name: str
    spot: float
    volatility: float | None = None
    carry_rate: float = 0.0
    carry_curve_name: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "spot": float(self.spot),
            "carry_rate": float(self.carry_rate),
        }
        if self.volatility is not None:
            payload["volatility"] = float(self.volatility)
        if self.carry_curve_name:
            payload["carry_curve_name"] = self.carry_curve_name
        return payload


@dataclass(frozen=True)
class MarketScenarioContract:
    """Normalized task-facing market scenario contract."""

    scenario_id: str
    schema_version: int
    source: str
    as_of: date
    description: str
    selected_components: Mapping[str, str] = field(default_factory=dict)
    constructor_kind: str = "request_only"
    valuation_date: date | None = None
    domestic_rate: float | None = None
    forecast_rate: float | None = None
    forecast_curve_name: str | None = None
    foreign_rate: float | None = None
    foreign_curve_name: str | None = None
    fx_pair: str | None = None
    hazard_rate: float | None = None
    recovery_rate: float | None = None
    black_vol: float | None = None
    shifted_black_vol: float | None = None
    shift: float | None = None
    sabr: Mapping[str, float] = field(default_factory=dict)
    underliers: tuple[ScenarioUnderlier, ...] = ()
    correlation_source: Mapping[str, object] | None = None
    scenario_digest: str = ""

    def with_digest(self) -> "MarketScenarioContract":
        digest = hashlib.sha1(_stable_json(self._digest_payload()).encode("utf-8")).hexdigest()[:16]
        return replace(self, scenario_digest=digest)

    def _digest_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "as_of": self.as_of.isoformat(),
            "description": self.description,
            "selected_components": dict(self.selected_components),
            "constructor_kind": self.constructor_kind,
            "valuation_date": None if self.valuation_date is None else self.valuation_date.isoformat(),
            "domestic_rate": self.domestic_rate,
            "forecast_rate": self.forecast_rate,
            "forecast_curve_name": self.forecast_curve_name,
            "foreign_rate": self.foreign_rate,
            "foreign_curve_name": self.foreign_curve_name,
            "fx_pair": self.fx_pair,
            "hazard_rate": self.hazard_rate,
            "recovery_rate": self.recovery_rate,
            "black_vol": self.black_vol,
            "shifted_black_vol": self.shifted_black_vol,
            "shift": self.shift,
            "sabr": dict(self.sabr),
            "underliers": [underlier.to_payload() for underlier in self.underliers],
            "correlation_source": dict(self.correlation_source or {}),
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._digest_payload()
        payload["scenario_digest"] = self.scenario_digest
        return payload

    def to_market_spec(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "as_of": self.as_of.isoformat(),
            "scenario_description": self.description,
            "selected_components": dict(self.selected_components),
            "scenario_schema_version": self.schema_version,
            "scenario_digest": self.scenario_digest,
            "scenario_constructor_kind": self.constructor_kind,
            "scenario_contract": self.to_payload(),
        }
        payload.update(dict(self.selected_components))
        financepy_inputs = self.financepy_inputs()
        if financepy_inputs:
            payload["benchmark_inputs"] = financepy_inputs
        return payload

    def financepy_inputs(self) -> dict[str, object]:
        if self.constructor_kind == "request_only":
            return {}

        valuation_date = (
            self.valuation_date.isoformat()
            if self.valuation_date is not None
            else self.as_of.isoformat()
        )
        payload: dict[str, object] = {"valuation_date": valuation_date}
        if self.domestic_rate is not None:
            payload["domestic_rate"] = float(self.domestic_rate)
            payload["flat_discount_rate"] = float(self.domestic_rate)
        if self.forecast_rate is not None:
            payload["flat_forward_rate"] = float(self.forecast_rate)
        if self.foreign_rate is not None:
            payload["foreign_rate"] = float(self.foreign_rate)
        if self.black_vol is not None:
            payload["black_vol"] = float(self.black_vol)
        if self.shifted_black_vol is not None:
            payload["shifted_black_vol"] = float(self.shifted_black_vol)
        if self.shift is not None:
            payload["shift"] = float(self.shift)
        if self.sabr:
            payload["sabr"] = dict(self.sabr)
        if self.hazard_rate is not None:
            payload["issuer_hazard_rate"] = float(self.hazard_rate)
        if self.recovery_rate is not None:
            payload["recovery_rate"] = float(self.recovery_rate)
        if self.fx_pair:
            payload["currency_pair"] = self.fx_pair

        if len(self.underliers) == 1:
            underlier = self.underliers[0]
            payload["stock_price"] = float(underlier.spot)
            payload["spot_fx"] = float(underlier.spot)
            payload["volatility"] = (
                float(underlier.volatility)
                if underlier.volatility is not None
                else payload.get("black_vol")
            )
            payload["dividend_rate"] = float(underlier.carry_rate)
        elif self.underliers:
            payload["stock_prices"] = [float(underlier.spot) for underlier in self.underliers]
            payload["volatilities"] = [
                None if underlier.volatility is None else float(underlier.volatility)
                for underlier in self.underliers
            ]
            payload["dividend_rates"] = [float(underlier.carry_rate) for underlier in self.underliers]

        if self.correlation_source:
            matrix = self.correlation_source.get("matrix")
            value = self.correlation_source.get("value")
            if matrix is not None:
                payload["correlation"] = matrix
            elif value is not None:
                payload["correlation"] = value
        return payload


def load_market_scenario_contracts(*, root: Path = ROOT) -> dict[str, MarketScenarioContract]:
    """Load normalized market-scenario contracts from the canonical manifest."""
    path = root / MARKET_SCENARIOS_MANIFEST
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(raw, Mapping):
        return {}
    version = int(raw.get("version") or 1)
    scenarios = dict(raw.get("scenarios") or {})
    contracts: dict[str, MarketScenarioContract] = {}
    for scenario_id, payload in scenarios.items():
        if not str(scenario_id).strip() or not isinstance(payload, Mapping):
            continue
        contract = _parse_market_scenario_contract(
            str(scenario_id).strip(),
            dict(payload),
            schema_version=version,
        )
        contracts[contract.scenario_id] = contract.with_digest()
    return contracts


def market_scenario_contract_from_task(
    task: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> MarketScenarioContract | None:
    """Return the embedded or canonical market-scenario contract for one task."""
    market_spec = dict(task.get("market") or {})
    embedded = market_spec.get("scenario_contract")
    if isinstance(embedded, Mapping):
        nested_contract = embedded.get("scenario_contract")
        if isinstance(nested_contract, Mapping):
            embedded = nested_contract
        contract = _parse_market_scenario_contract(
            str(task.get("market_scenario_id") or embedded.get("scenario_id") or "").strip(),
            {
                "source": market_spec.get("source") or embedded.get("source"),
                "as_of": market_spec.get("as_of") or embedded.get("as_of"),
                "description": market_spec.get("scenario_description") or embedded.get("description"),
                "selected_components": dict(embedded.get("selected_components") or {}),
                "constructor": {
                    "kind": embedded.get("constructor_kind"),
                    "valuation_date": embedded.get("valuation_date"),
                    "domestic_rate": embedded.get("domestic_rate"),
                    "forecast_rate": embedded.get("forecast_rate"),
                    "forecast_curve_name": embedded.get("forecast_curve_name"),
                    "foreign_rate": embedded.get("foreign_rate"),
                    "foreign_curve_name": embedded.get("foreign_curve_name"),
                    "fx_pair": embedded.get("fx_pair"),
                    "hazard_rate": embedded.get("hazard_rate"),
                    "recovery_rate": embedded.get("recovery_rate"),
                    "black_vol": embedded.get("black_vol"),
                    "shifted_black_vol": embedded.get("shifted_black_vol"),
                    "shift": embedded.get("shift"),
                    "sabr": dict(embedded.get("sabr") or {}),
                    "underliers": list(embedded.get("underliers") or ()),
                    "correlation_source": dict(embedded.get("correlation_source") or {}),
                },
            },
            schema_version=int(market_spec.get("scenario_schema_version") or embedded.get("schema_version") or 2),
        )
        embedded_digest = str(
            market_spec.get("scenario_digest")
            or embedded.get("scenario_digest")
            or ""
        ).strip()
        if embedded_digest:
            return replace(contract, scenario_digest=embedded_digest)
        return contract.with_digest()

    scenario_id = str(task.get("market_scenario_id") or "").strip()
    if not scenario_id:
        return None
    return load_market_scenario_contracts(root=root).get(scenario_id)


def construct_market_state_for_scenario(
    contract: MarketScenarioContract,
    market_state: MarketState,
    *,
    task_id: str | None = None,
) -> tuple[MarketState, dict[str, object]]:
    """Apply one normalized scenario contract onto a runtime MarketState."""
    if contract.constructor_kind == "request_only":
        return market_state, {
            "scenario_digest": contract.scenario_digest,
            "scenario_schema_version": contract.schema_version,
            "scenario_construction_kind": contract.constructor_kind,
        }

    from trellis.instruments.fx import FXRate
    from trellis.models.vol_surface import FlatVol

    curve_value_date = contract.valuation_date or contract.as_of
    discount = _flat_yield_curve(contract.domestic_rate, value_date=curve_value_date) or market_state.discount
    credit_curve = _flat_credit_curve(contract.hazard_rate) or market_state.credit_curve
    forecast_curves = dict(getattr(market_state, "forecast_curves", None) or {})
    fx_rates = dict(getattr(market_state, "fx_rates", None) or {})
    underlier_spots = dict(getattr(market_state, "underlier_spots", None) or {})
    model_parameters = dict(getattr(market_state, "model_parameters", None) or {})
    spot = market_state.spot
    vol_surface = market_state.vol_surface

    applied_inputs: dict[str, object] = {}

    if contract.black_vol is not None:
        vol_surface = FlatVol(float(contract.black_vol))
        applied_inputs["black_vol"] = float(contract.black_vol)

    if contract.constructor_kind in {"single_asset_equity", "multi_asset_equity"}:
        underlier_carry_rates: dict[str, float] = {}
        for underlier in contract.underliers:
            underlier_spots[underlier.name] = float(underlier.spot)
            curve_name = underlier.carry_curve_name or f"{underlier.name}-DISC"
            forecast_curves[curve_name] = _flat_yield_curve(float(underlier.carry_rate), value_date=curve_value_date)
            underlier_carry_rates[underlier.name] = float(underlier.carry_rate)
        if contract.underliers:
            spot = float(contract.underliers[0].spot)
        per_underlier_vols = {
            underlier.name: float(underlier.volatility)
            for underlier in contract.underliers
            if underlier.volatility is not None
        }
        if per_underlier_vols:
            model_parameters["underlier_vols"] = per_underlier_vols
            if vol_surface is None and len(set(per_underlier_vols.values())) == 1:
                vol_surface = FlatVol(next(iter(per_underlier_vols.values())))
            applied_inputs["underlier_vols"] = sorted(per_underlier_vols)
        if underlier_carry_rates:
            model_parameters["underlier_carry_rates"] = underlier_carry_rates

    if contract.constructor_kind == "single_asset_fx":
        if contract.underliers:
            spot = float(contract.underliers[0].spot)
        if contract.fx_pair and spot is not None and len(contract.fx_pair) == 6:
            fx_rates[contract.fx_pair] = FXRate(
                spot=float(spot),
                domestic=contract.fx_pair[3:],
                foreign=contract.fx_pair[:3],
            )
            underlier_spots[contract.fx_pair] = float(spot)
            applied_inputs["fx_pair"] = contract.fx_pair
        foreign_curve_name = contract.foreign_curve_name or contract.forecast_curve_name
        foreign_rate = contract.foreign_rate if contract.foreign_rate is not None else contract.forecast_rate
        if foreign_curve_name and foreign_rate is not None:
            forecast_curves[foreign_curve_name] = _flat_yield_curve(float(foreign_rate), value_date=curve_value_date)
            applied_inputs["foreign_curve_name"] = foreign_curve_name

    if contract.constructor_kind == "flat_rates":
        if contract.forecast_curve_name and contract.forecast_rate is not None:
            forecast_curves[contract.forecast_curve_name] = _flat_yield_curve(
                float(contract.forecast_rate),
                value_date=curve_value_date,
            )
            applied_inputs["forecast_curve_name"] = contract.forecast_curve_name
        if contract.shifted_black_vol is not None:
            model_parameters["shifted_black_vol"] = float(contract.shifted_black_vol)
        if contract.shift is not None:
            model_parameters["shift"] = float(contract.shift)
        if contract.sabr:
            model_parameters["sabr"] = dict(contract.sabr)

    if contract.constructor_kind == "flat_credit":
        if contract.recovery_rate is not None:
            model_parameters["recovery_rate"] = float(contract.recovery_rate)

    if contract.correlation_source:
        model_parameters["correlation_source"] = dict(contract.correlation_source)
        applied_inputs["correlation_source"] = contract.correlation_source.get("kind") or "explicit"

    if contract.domestic_rate is not None:
        applied_inputs["domestic_rate"] = float(contract.domestic_rate)
    if contract.forecast_rate is not None:
        applied_inputs["forecast_rate"] = float(contract.forecast_rate)
    if contract.foreign_rate is not None:
        applied_inputs["foreign_rate"] = float(contract.foreign_rate)
    if contract.hazard_rate is not None:
        applied_inputs["hazard_rate"] = float(contract.hazard_rate)
    if contract.underliers:
        applied_inputs["underliers"] = [underlier.name for underlier in contract.underliers]

    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    market_provenance["market_scenario"] = {
        "task_id": str(task_id or "").strip(),
        "scenario_id": contract.scenario_id,
        "scenario_digest": contract.scenario_digest,
        "schema_version": contract.schema_version,
        "constructor_kind": contract.constructor_kind,
        "selected_components": dict(contract.selected_components),
        "applied_inputs": applied_inputs,
    }
    if contract.recovery_rate is not None:
        market_provenance["market_scenario"]["recovery_rate"] = float(contract.recovery_rate)

    constructed_state = replace(
        market_state,
        discount=discount,
        credit_curve=credit_curve,
        forecast_curves=forecast_curves or None,
        vol_surface=vol_surface,
        fx_rates=fx_rates or None,
        spot=spot,
        underlier_spots=underlier_spots or None,
        model_parameters=model_parameters or None,
        market_provenance=market_provenance,
    )
    return constructed_state, {
        "market_scenario_construction": True,
        "scenario_digest": contract.scenario_digest,
        "scenario_schema_version": contract.schema_version,
        "scenario_construction_kind": contract.constructor_kind,
        "scenario_applied_inputs": applied_inputs,
    }


def build_market_scenario_coverage_report(
    *,
    pricing_tasks: list[dict[str, Any]],
    negative_tasks: list[dict[str, Any]],
    canaries: list[dict[str, Any]],
    scenario_contracts: Mapping[str, MarketScenarioContract],
    required_task_corpora: tuple[str, ...] = ("benchmark_financepy", "extension", "negative"),
) -> dict[str, object]:
    """Return a stable market-scenario coverage audit across task corpora."""
    task_lookup = {
        str(task.get("id") or "").strip(): dict(task)
        for task in [*pricing_tasks, *negative_tasks]
        if str(task.get("id") or "").strip()
    }
    usage_by_scenario: dict[str, dict[str, int]] = {
        scenario_id: {"pricing": 0, "negative": 0, "canary": 0}
        for scenario_id in scenario_contracts
    }
    missing_task_scenarios: list[dict[str, str]] = []
    unknown_scenario_refs: list[dict[str, str]] = []
    required_corpora = {str(item).strip() for item in required_task_corpora if str(item).strip()}
    corpus_counts: dict[str, int] = {}

    def _record_task(task: Mapping[str, Any], bucket: str) -> None:
        task_id = str(task.get("id") or "").strip()
        task_corpus = str(task.get("task_corpus") or "").strip()
        if task_corpus:
            corpus_counts[task_corpus] = corpus_counts.get(task_corpus, 0) + 1
        scenario_id = str(task.get("market_scenario_id") or "").strip()
        requires_scenario = task_corpus in required_corpora
        if not scenario_id:
            if requires_scenario:
                missing_task_scenarios.append({"task_id": task_id, "task_corpus": task_corpus or bucket})
            return
        if scenario_id not in usage_by_scenario:
            if requires_scenario:
                unknown_scenario_refs.append(
                    {"task_id": task_id, "task_corpus": task_corpus or bucket, "scenario_id": scenario_id}
                )
            return
        usage_by_scenario[scenario_id][bucket] += 1

    for task in pricing_tasks:
        _record_task(task, "pricing")
    for task in negative_tasks:
        _record_task(task, "negative")
    for canary in canaries:
        task_id = str(canary.get("id") or "").strip()
        task = task_lookup.get(task_id)
        if task is None:
            unknown_scenario_refs.append({"task_id": task_id, "scenario_id": "__missing_task_payload__"})
            continue
        task_corpus = str(task.get("task_corpus") or "").strip()
        requires_scenario = task_corpus in required_corpora
        scenario_id = str(task.get("market_scenario_id") or "").strip()
        if not scenario_id:
            if requires_scenario:
                missing_task_scenarios.append({"task_id": task_id, "task_corpus": task_corpus or "canary"})
            continue
        if scenario_id not in usage_by_scenario:
            if requires_scenario:
                unknown_scenario_refs.append(
                    {"task_id": task_id, "task_corpus": task_corpus or "canary", "scenario_id": scenario_id}
                )
            continue
        usage_by_scenario[scenario_id]["canary"] += 1

    constructor_counts: dict[str, int] = {}
    for contract in scenario_contracts.values():
        constructor_counts[contract.constructor_kind] = constructor_counts.get(contract.constructor_kind, 0) + 1

    return {
        "scenario_count": len(scenario_contracts),
        "pricing_task_count": len(pricing_tasks),
        "negative_task_count": len(negative_tasks),
        "canary_count": len(canaries),
        "required_task_corpora": sorted(required_corpora),
        "task_counts_by_corpus": dict(sorted(corpus_counts.items())),
        "constructor_counts": constructor_counts,
        "usage_by_scenario": usage_by_scenario,
        "missing_task_scenarios": sorted(
            (
                {"task_id": item["task_id"], "task_corpus": item["task_corpus"]}
                for item in missing_task_scenarios
                if item.get("task_id")
            ),
            key=lambda item: (item["task_corpus"], item["task_id"]),
        ),
        "unknown_scenario_refs": sorted(
            unknown_scenario_refs,
            key=lambda item: (
                item.get("task_corpus", ""),
                item["task_id"],
                item["scenario_id"],
            ),
        ),
    }


def render_market_scenario_coverage_report(report: Mapping[str, object]) -> str:
    """Render the coverage audit as Markdown."""
    lines = [
        "# Market Scenario Coverage Audit",
        f"- Scenario count: `{report.get('scenario_count', 0)}`",
        f"- Pricing task count: `{report.get('pricing_task_count', 0)}`",
        f"- Negative task count: `{report.get('negative_task_count', 0)}`",
        f"- Canary count: `{report.get('canary_count', 0)}`",
        f"- Required task corpora: `{', '.join(report.get('required_task_corpora') or [])}`",
        "",
        "## Task Counts By Corpus",
    ]
    for corpus, count in sorted(dict(report.get("task_counts_by_corpus") or {}).items()):
        lines.append(f"- `{corpus}`: `{count}`")
    lines.extend([
        "",
        "## Constructor Counts",
    ])
    for kind, count in sorted(dict(report.get("constructor_counts") or {}).items()):
        lines.append(f"- `{kind}`: `{count}`")
    lines.extend(["", "## Scenario Usage"])
    for scenario_id, usage in sorted(dict(report.get("usage_by_scenario") or {}).items()):
        lines.append(
            f"- `{scenario_id}`: pricing={usage.get('pricing', 0)}, "
            f"negative={usage.get('negative', 0)}, canary={usage.get('canary', 0)}"
        )
    missing = list(report.get("missing_task_scenarios") or ())
    if missing:
        lines.extend(["", "## Tasks Missing Market Scenarios"])
        lines.extend(
            f"- `{item['task_id']}` (`{item['task_corpus']}`)"
            for item in missing
            if isinstance(item, Mapping)
        )
    unknown = list(report.get("unknown_scenario_refs") or ())
    if unknown:
        lines.extend(["", "## Unknown Scenario References"])
        lines.extend(
            f"- `{item['task_id']}` (`{item.get('task_corpus', 'unknown')}`) -> `{item['scenario_id']}`"
            for item in unknown
            if isinstance(item, Mapping)
        )
    return "\n".join(lines) + "\n"


def _parse_market_scenario_contract(
    scenario_id: str,
    raw: Mapping[str, Any],
    *,
    schema_version: int,
) -> MarketScenarioContract:
    as_of = _date_or_default(raw.get("as_of"), default=date(2024, 11, 15))
    constructor = dict(raw.get("constructor") or {})
    legacy_inputs = dict(raw.get("benchmark_inputs") or {})
    kind = str(
        constructor.get("kind")
        or _infer_legacy_constructor_kind(legacy_inputs)
        or "request_only"
    ).strip()

    valuation_date = _date_or_default(
        constructor.get("valuation_date") or legacy_inputs.get("valuation_date"),
        default=as_of,
    )
    underliers = _parse_underliers(
        constructor=constructor,
        legacy_inputs=legacy_inputs,
    )
    correlation_source = _parse_correlation_source(
        constructor=constructor,
        legacy_inputs=legacy_inputs,
    )
    contract = MarketScenarioContract(
        scenario_id=scenario_id,
        schema_version=schema_version,
        source=str(raw.get("source") or "mock"),
        as_of=as_of,
        description=str(raw.get("description") or "").strip(),
        selected_components=_string_mapping(raw.get("selected_components")),
        constructor_kind=kind,
        valuation_date=valuation_date,
        domestic_rate=_float_or_none(
            constructor.get("domestic_rate")
            if constructor
            else legacy_inputs.get("domestic_rate") or legacy_inputs.get("flat_discount_rate")
        ),
        forecast_rate=_float_or_none(
            constructor.get("forecast_rate")
            if constructor
            else legacy_inputs.get("flat_forward_rate")
        ),
        forecast_curve_name=(
            str(constructor.get("forecast_curve_name") or "").strip()
            or _string_mapping(raw.get("selected_components")).get("forecast_curve")
            or None
        ),
        foreign_rate=_float_or_none(
            constructor.get("foreign_rate")
            if constructor
            else legacy_inputs.get("foreign_rate")
        ),
        foreign_curve_name=(
            str(constructor.get("foreign_curve_name") or "").strip()
            or (
                _string_mapping(raw.get("selected_components")).get("forecast_curve")
                if kind == "single_asset_fx"
                else None
            )
            or None
        ),
        fx_pair=str(constructor.get("fx_pair") or legacy_inputs.get("currency_pair") or _string_mapping(raw.get("selected_components")).get("fx_rate") or "").strip() or None,
        hazard_rate=_float_or_none(
            constructor.get("hazard_rate")
            if constructor
            else legacy_inputs.get("issuer_hazard_rate")
        ),
        recovery_rate=_float_or_none(
            constructor.get("recovery_rate")
            if constructor
            else legacy_inputs.get("recovery_rate")
        ),
        black_vol=_float_or_none(
            constructor.get("black_vol")
            if constructor
            else legacy_inputs.get("black_vol") or legacy_inputs.get("volatility")
        ),
        shifted_black_vol=_float_or_none(
            constructor.get("shifted_black_vol")
            if constructor
            else legacy_inputs.get("shifted_black_vol")
        ),
        shift=_float_or_none(
            constructor.get("shift") if constructor else legacy_inputs.get("shift")
        ),
        sabr={
            str(key): float(value)
            for key, value in dict(constructor.get("sabr") or legacy_inputs.get("sabr") or {}).items()
            if str(key).strip()
        },
        underliers=underliers,
        correlation_source=correlation_source,
    )
    return contract


def _infer_legacy_constructor_kind(legacy_inputs: Mapping[str, Any]) -> str:
    if not legacy_inputs:
        return "request_only"
    if "issuer_hazard_rate" in legacy_inputs:
        return "flat_credit"
    if "spot_fx" in legacy_inputs:
        return "single_asset_fx"
    if "stock_prices" in legacy_inputs:
        return "multi_asset_equity"
    if "flat_forward_rate" in legacy_inputs:
        return "flat_rates"
    if "stock_price" in legacy_inputs:
        return "single_asset_equity"
    return "request_only"


def _parse_underliers(
    *,
    constructor: Mapping[str, Any],
    legacy_inputs: Mapping[str, Any],
) -> tuple[ScenarioUnderlier, ...]:
    raw_underliers = list(constructor.get("underliers") or ())
    if not raw_underliers and constructor.get("underlier"):
        raw_underliers = [dict(constructor.get("underlier") or {})]
    if raw_underliers:
        parsed: list[ScenarioUnderlier] = []
        for raw in raw_underliers:
            if not isinstance(raw, Mapping):
                continue
            carry = dict(raw.get("carry") or {})
            parsed.append(
                ScenarioUnderlier(
                    name=str(raw.get("name") or "").strip(),
                    spot=float(raw.get("spot")),
                    volatility=_float_or_none(raw.get("volatility")),
                    carry_rate=float(carry.get("rate") or raw.get("carry_rate") or 0.0),
                    carry_curve_name=str(carry.get("curve_name") or raw.get("carry_curve_name") or "").strip() or None,
                )
            )
        return tuple(item for item in parsed if item.name)

    stock_prices = list(legacy_inputs.get("stock_prices") or ())
    volatilities = list(legacy_inputs.get("volatilities") or ())
    dividend_rates = list(legacy_inputs.get("dividend_rates") or ())
    if stock_prices:
        names = [f"ASSET_{index + 1}" for index in range(len(stock_prices))]
        return tuple(
            ScenarioUnderlier(
                name=names[index],
                spot=float(stock_prices[index]),
                volatility=None if index >= len(volatilities) or volatilities[index] in {None, ""} else float(volatilities[index]),
                carry_rate=0.0 if index >= len(dividend_rates) or dividend_rates[index] in {None, ""} else float(dividend_rates[index]),
                carry_curve_name=f"{names[index]}-DISC",
            )
            for index in range(len(stock_prices))
        )
    if legacy_inputs.get("stock_price") not in {None, ""}:
        return (
            ScenarioUnderlier(
                name="SPX",
                spot=float(legacy_inputs["stock_price"]),
                volatility=_float_or_none(legacy_inputs.get("volatility")),
                carry_rate=float(legacy_inputs.get("dividend_rate") or 0.0),
                carry_curve_name="SPX-DISC",
            ),
        )
    if legacy_inputs.get("spot_fx") not in {None, ""}:
        pair = str(legacy_inputs.get("currency_pair") or "EURUSD").strip() or "EURUSD"
        return (
            ScenarioUnderlier(
                name=pair,
                spot=float(legacy_inputs["spot_fx"]),
                volatility=_float_or_none(legacy_inputs.get("volatility")),
                carry_rate=float(legacy_inputs.get("foreign_rate") or 0.0),
                carry_curve_name=f"{pair[:3]}-DISC",
            ),
        )
    return ()


def _parse_correlation_source(
    *,
    constructor: Mapping[str, Any],
    legacy_inputs: Mapping[str, Any],
) -> Mapping[str, object] | None:
    raw = constructor.get("correlation_source")
    if isinstance(raw, Mapping):
        return dict(raw)
    if legacy_inputs.get("correlation") is not None:
        return {
            "kind": "explicit",
            "matrix": legacy_inputs["correlation"],
        }
    return None
