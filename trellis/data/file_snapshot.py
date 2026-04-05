"""File-based market-snapshot import contracts and rehydration helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Mapping

import yaml

from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol, GridVolSurface


FILE_IMPORT_PROVIDER_ID = "market_data.file_import"
FILE_IMPORT_SOURCE = "file_import"
FILE_IMPORT_BUNDLE_TYPE = "imported_market_snapshot"
FILE_IMPORT_SCHEMA_VERSION = 1


def _normalize_date(value, *, field: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required for file-based market snapshots.")
    return date.fromisoformat(text)


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip()


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _read_structured_file(path: Path):
    if not path.exists():
        raise ValueError(f"Referenced snapshot component file does not exist: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml", ".json"}:
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported snapshot component file type: {path.suffix}")


def _read_fixing_history_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"Referenced fixing-history file does not exist: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Fixing-history CSV must include a header row: {path}")
        required = {"date", "value"}
        if not required.issubset({field.strip() for field in reader.fieldnames}):
            raise ValueError(
                f"Fixing-history CSV must include date,value columns: {path}"
            )
        for row in reader:
            rows.append(
                {
                    "date": str(row.get("date", "")).strip(),
                    "value": float(row.get("value", 0.0)),
                }
            )
    return rows


def _resolve_path(base_dir: Path, reference: str) -> Path:
    path = Path(str(reference).strip()).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalize_mapping_payload(
    payload,
    *,
    base_dir: Path,
    family: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if payload is None or payload == "":
        return {}, {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{family} must be a mapping of named snapshot components.")
    file_key = _normalize_token(payload.get("file")) if set(payload) == {"file"} else ""
    if file_key:
        source_path = _resolve_path(base_dir, file_key)
        loaded = _read_structured_file(source_path)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{family} file must resolve to a mapping: {source_path}")
        return (
            {str(name): _json_safe(value) for name, value in loaded.items()},
            {"file": str(source_path)},
        )

    normalized: dict[str, object] = {}
    source_files: dict[str, object] = {}
    for name, value in payload.items():
        normalized_name = str(name).strip()
        if not normalized_name:
            continue
        if isinstance(value, str) and value.strip():
            source_path = _resolve_path(base_dir, value)
            if family == "fixing_histories" and source_path.suffix.lower() == ".csv":
                normalized[normalized_name] = _read_fixing_history_csv(source_path)
            else:
                normalized[normalized_name] = _json_safe(_read_structured_file(source_path))
            source_files[normalized_name] = str(source_path)
            continue
        normalized[normalized_name] = _json_safe(value)
    return normalized, source_files


def _normalize_underlier_spots(payload, *, base_dir: Path) -> tuple[dict[str, float], dict[str, object]]:
    normalized, source_files = _normalize_mapping_payload(
        payload,
        base_dir=base_dir,
        family="underlier_spots",
    )
    result: dict[str, float] = {}
    for name, value in normalized.items():
        if isinstance(value, Mapping):
            if "spot" in value:
                result[name] = float(value["spot"])
                continue
            raise ValueError(f"Underlier spot payload for {name!r} must include `spot`.")
        result[name] = float(value)
    return result, source_files


def _normalize_fixing_histories(payload, *, base_dir: Path) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    normalized, source_files = _normalize_mapping_payload(
        payload,
        base_dir=base_dir,
        family="fixing_histories",
    )
    result: dict[str, list[dict[str, object]]] = {}
    for name, value in normalized.items():
        if isinstance(value, Mapping):
            observations = [
                {"date": str(obs_date).strip(), "value": float(obs_value)}
                for obs_date, obs_value in value.items()
            ]
        elif isinstance(value, list):
            observations = [
                {
                    "date": str(item.get("date", "")).strip(),
                    "value": float(item.get("value", 0.0)),
                }
                for item in value
            ]
        else:
            raise ValueError(
                f"Fixing history payload for {name!r} must be a list of observations or date/value mapping."
            )
        result[name] = observations
    return result, source_files


def _default_selection(
    *,
    family: str,
    available: Mapping[str, object],
    explicit_default: str | None,
    warning_label: str,
    warnings: list[str],
) -> str | None:
    token = _normalize_token(explicit_default)
    if token:
        if token not in available:
            raise ValueError(
                f"Snapshot default {warning_label!r} is not present in {family}."
            )
        return token
    if len(available) == 1:
        selected = next(iter(available))
        warnings.append(
            f"Defaulted {warning_label} to {selected!r} because only one {family} component was provided."
        )
        return selected
    return None


def _yield_curve_from_spec(spec: Mapping[str, object]) -> YieldCurve:
    kind = _normalize_token(spec.get("kind")).lower() or "points"
    if kind == "flat":
        return YieldCurve.flat(
            float(spec.get("rate", 0.0)),
            max_tenor=float(spec.get("max_tenor", 30.0)),
        )
    if kind == "treasury_yields":
        quotes = {float(key): float(value) for key, value in dict(spec.get("quotes") or {}).items()}
        return YieldCurve.from_treasury_yields(quotes)
    if kind in {"points", "zero_rates"}:
        tenors = tuple(float(item) for item in spec.get("tenors") or ())
        rates = tuple(float(item) for item in spec.get("rates") or ())
        if len(tenors) != len(rates) or not tenors:
            raise ValueError("Yield-curve point specs require matching non-empty tenors and rates.")
        return YieldCurve(tenors, rates)
    raise ValueError(f"Unsupported yield-curve spec kind: {kind!r}")


def _credit_curve_from_spec(spec: Mapping[str, object]) -> CreditCurve:
    kind = _normalize_token(spec.get("kind")).lower() or "flat"
    if kind == "flat":
        return CreditCurve.flat(
            float(spec.get("hazard_rate", 0.0)),
            max_tenor=float(spec.get("max_tenor", 30.0)),
        )
    if kind == "spreads":
        spreads = {float(key): float(value) for key, value in dict(spec.get("spreads") or {}).items()}
        return CreditCurve.from_spreads(
            spreads,
            recovery=float(spec.get("recovery", 0.4)),
        )
    if kind in {"hazard_points", "points"}:
        tenors = tuple(float(item) for item in spec.get("tenors") or ())
        hazard_rates = tuple(float(item) for item in spec.get("hazard_rates") or ())
        if len(tenors) != len(hazard_rates) or not tenors:
            raise ValueError(
                "Credit-curve point specs require matching non-empty tenors and hazard_rates."
            )
        return CreditCurve(tenors, hazard_rates)
    raise ValueError(f"Unsupported credit-curve spec kind: {kind!r}")


def _vol_surface_from_spec(spec: Mapping[str, object]):
    kind = _normalize_token(spec.get("kind")).lower() or "flat"
    if kind == "flat":
        return FlatVol(float(spec.get("vol", 0.0)))
    if kind == "grid":
        expiries = tuple(float(item) for item in spec.get("expiries") or ())
        strikes = tuple(float(item) for item in spec.get("strikes") or ())
        vols = tuple(
            tuple(float(value) for value in row)
            for row in spec.get("vols") or ()
        )
        return GridVolSurface(
            expiries=expiries,
            strikes=strikes,
            vols=vols,
        )
    raise ValueError(f"Unsupported vol-surface spec kind: {kind!r}")


def _fx_rate_from_spec(spec: Mapping[str, object]) -> FXRate:
    return FXRate(
        spot=float(spec.get("spot", 0.0)),
        domestic=_normalize_token(spec.get("domestic")),
        foreign=_normalize_token(spec.get("foreign")),
    )


def _component_source_kind(spec) -> str:
    if isinstance(spec, Mapping):
        source_kind = _normalize_token(spec.get("source_kind"))
        if source_kind:
            return source_kind
        provenance = spec.get("provenance") or {}
        if isinstance(provenance, Mapping):
            return _normalize_token(provenance.get("source_kind"))
    return ""


def _intrinsic_warnings(manifest_contract: Mapping[str, object]) -> tuple[str, ...]:
    warnings: list[str] = []
    expected_families = tuple(
        str(item).strip()
        for item in manifest_contract.get("expected_component_families") or ()
        if str(item).strip()
    )
    for family in expected_families:
        if not dict(manifest_contract.get(family) or {}):
            warnings.append(f"Expected market family {family!r} is missing from the imported snapshot.")
    for family in ("discount_curves", "forecast_curves", "vol_surfaces", "credit_curves", "fx_rates", "underlier_spots", "fixing_histories"):
        for name, spec in dict(manifest_contract.get(family) or {}).items():
            if _component_source_kind(spec).startswith("synthetic"):
                warnings.append(f"Imported {family} component {name!r} is marked synthetic.")
    defaults = dict(manifest_contract.get("defaults") or {})
    default_map = {
        "discount_curves": "discount_curve",
        "vol_surfaces": "vol_surface",
        "credit_curves": "credit_curve",
        "underlier_spots": "underlier_spot",
        "fixing_histories": "fixing_history",
    }
    for family, warning_label in default_map.items():
        if _normalize_token(defaults.get(warning_label)):
            continue
        available = dict(manifest_contract.get(family) or {})
        if len(available) == 1:
            selected = next(iter(available))
            warnings.append(
                f"Defaulted {warning_label} to {selected!r} because only one {family} component was provided."
            )
    return tuple(dict.fromkeys(warnings))


def manifest_warnings(
    manifest_contract: Mapping[str, object],
    *,
    reference_date: date | str | None = None,
) -> tuple[str, ...]:
    warnings = [
        str(item)
        for item in manifest_contract.get("intrinsic_warnings") or ()
        if str(item).strip()
    ]
    warnings.extend(_intrinsic_warnings(manifest_contract))
    stale_after_days = manifest_contract.get("stale_after_days")
    if stale_after_days not in {None, ""} and reference_date is not None:
        as_of = _normalize_date(manifest_contract.get("as_of"), field="as_of")
        reference = _normalize_date(reference_date, field="reference_date")
        age_days = (reference - as_of).days
        threshold = int(stale_after_days)
        if age_days > threshold:
            warnings.append(
                f"Imported snapshot as_of {as_of.isoformat()} is stale versus {reference.isoformat()} (stale_after_days={threshold})."
            )
    return tuple(dict.fromkeys(warnings))


def _snapshot_id(manifest_contract: Mapping[str, object]) -> str:
    payload = {
        "provider_id": FILE_IMPORT_PROVIDER_ID,
        "manifest": _json_safe(manifest_contract),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"snapshot_{digest}"


def build_market_snapshot(
    manifest_contract: Mapping[str, object],
    *,
    snapshot_id: str | None = None,
) -> MarketSnapshot:
    defaults = dict(manifest_contract.get("defaults") or {})
    discount_curves = {
        name: _yield_curve_from_spec(spec)
        for name, spec in dict(manifest_contract.get("discount_curves") or {}).items()
    }
    forecast_curves = {
        name: _yield_curve_from_spec(spec)
        for name, spec in dict(manifest_contract.get("forecast_curves") or {}).items()
    }
    vol_surfaces = {
        name: _vol_surface_from_spec(spec)
        for name, spec in dict(manifest_contract.get("vol_surfaces") or {}).items()
    }
    credit_curves = {
        name: _credit_curve_from_spec(spec)
        for name, spec in dict(manifest_contract.get("credit_curves") or {}).items()
    }
    fx_rates = {
        name: _fx_rate_from_spec(spec)
        for name, spec in dict(manifest_contract.get("fx_rates") or {}).items()
    }
    underlier_spots = {
        name: float(value)
        for name, value in dict(manifest_contract.get("underlier_spots") or {}).items()
    }
    fixing_histories = {
        name: {
            _normalize_date(item.get("date"), field=f"{name}.date"): float(item.get("value", 0.0))
            for item in history
        }
        for name, history in dict(manifest_contract.get("fixing_histories") or {}).items()
    }
    metadata = dict(manifest_contract.get("metadata") or {})
    default_fixing_history = _normalize_token(defaults.get("fixing_history"))
    default_forecast_curve = _normalize_token(defaults.get("forecast_curve"))
    if default_forecast_curve:
        metadata["default_forecast_curve"] = default_forecast_curve

    resolved_snapshot_id = str(snapshot_id or _snapshot_id(manifest_contract)).strip()
    source = _normalize_token(manifest_contract.get("source")) or FILE_IMPORT_SOURCE
    return MarketSnapshot(
        as_of=_normalize_date(manifest_contract.get("as_of"), field="as_of"),
        source=source,
        discount_curves=discount_curves,
        forecast_curves=forecast_curves,
        vol_surfaces=vol_surfaces,
        credit_curves=credit_curves,
        fixing_histories=fixing_histories,
        fx_rates=fx_rates,
        underlier_spots=underlier_spots,
        metadata=metadata,
        default_discount_curve=_normalize_token(defaults.get("discount_curve")) or None,
        default_vol_surface=_normalize_token(defaults.get("vol_surface")) or None,
        default_credit_curve=_normalize_token(defaults.get("credit_curve")) or None,
        default_fixing_history=default_fixing_history or None,
        default_underlier_spot=_normalize_token(defaults.get("underlier_spot")) or None,
        provenance={
            "provider_id": FILE_IMPORT_PROVIDER_ID,
            "snapshot_id": resolved_snapshot_id,
            "source": source,
            "source_kind": "explicit_input",
            "source_ref": _normalize_token(manifest_contract.get("manifest_path")),
            "import_schema_version": FILE_IMPORT_SCHEMA_VERSION,
            "bundle_type": FILE_IMPORT_BUNDLE_TYPE,
        },
    )


def manifest_summary(manifest_contract: Mapping[str, object]) -> dict[str, object]:
    defaults = dict(manifest_contract.get("defaults") or {})
    return {
        "discount_curves": sorted(dict(manifest_contract.get("discount_curves") or {})),
        "forecast_curves": sorted(dict(manifest_contract.get("forecast_curves") or {})),
        "vol_surfaces": sorted(dict(manifest_contract.get("vol_surfaces") or {})),
        "credit_curves": sorted(dict(manifest_contract.get("credit_curves") or {})),
        "fx_rates": sorted(dict(manifest_contract.get("fx_rates") or {})),
        "underlier_spots": sorted(dict(manifest_contract.get("underlier_spots") or {})),
        "fixing_histories": sorted(dict(manifest_contract.get("fixing_histories") or {})),
        "defaults": {
            key: value
            for key, value in sorted(defaults.items())
            if _normalize_token(value)
        },
    }


def load_manifest_contract(manifest_path: Path | str) -> dict[str, object]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _read_structured_file(path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Snapshot manifest must resolve to a mapping: {path}")
    base_dir = path.parent
    defaults = dict(manifest.get("defaults") or {})
    warnings: list[str] = []

    discount_curves, discount_files = _normalize_mapping_payload(
        manifest.get("discount_curves"),
        base_dir=base_dir,
        family="discount_curves",
    )
    forecast_curves, forecast_files = _normalize_mapping_payload(
        manifest.get("forecast_curves"),
        base_dir=base_dir,
        family="forecast_curves",
    )
    vol_surfaces, vol_files = _normalize_mapping_payload(
        manifest.get("vol_surfaces"),
        base_dir=base_dir,
        family="vol_surfaces",
    )
    credit_curves, credit_files = _normalize_mapping_payload(
        manifest.get("credit_curves"),
        base_dir=base_dir,
        family="credit_curves",
    )
    fx_rates, fx_files = _normalize_mapping_payload(
        manifest.get("fx_rates"),
        base_dir=base_dir,
        family="fx_rates",
    )
    underlier_spots, spot_files = _normalize_underlier_spots(
        manifest.get("underlier_spots"),
        base_dir=base_dir,
    )
    fixing_histories, fixing_files = _normalize_fixing_histories(
        manifest.get("fixing_histories"),
        base_dir=base_dir,
    )

    defaults["discount_curve"] = _default_selection(
        family="discount_curves",
        available=discount_curves,
        explicit_default=defaults.get("discount_curve"),
        warning_label="discount_curve",
        warnings=warnings,
    )
    defaults["vol_surface"] = _default_selection(
        family="vol_surfaces",
        available=vol_surfaces,
        explicit_default=defaults.get("vol_surface"),
        warning_label="vol_surface",
        warnings=warnings,
    )
    defaults["credit_curve"] = _default_selection(
        family="credit_curves",
        available=credit_curves,
        explicit_default=defaults.get("credit_curve"),
        warning_label="credit_curve",
        warnings=warnings,
    )
    defaults["underlier_spot"] = _default_selection(
        family="underlier_spots",
        available=underlier_spots,
        explicit_default=defaults.get("underlier_spot"),
        warning_label="underlier_spot",
        warnings=warnings,
    )
    defaults["fixing_history"] = _default_selection(
        family="fixing_histories",
        available=fixing_histories,
        explicit_default=defaults.get("fixing_history"),
        warning_label="fixing_history",
        warnings=warnings,
    )
    forecast_default = _normalize_token(defaults.get("forecast_curve"))
    if forecast_default and forecast_default not in forecast_curves:
        raise ValueError(
            f"Snapshot default 'forecast_curve' is not present in forecast_curves."
        )

    manifest_contract = {
        "schema_version": FILE_IMPORT_SCHEMA_VERSION,
        "manifest_path": str(path),
        "as_of": _normalize_date(manifest.get("as_of"), field="as_of").isoformat(),
        "source": _normalize_token(manifest.get("source")) or FILE_IMPORT_SOURCE,
        "stale_after_days": (
            None
            if manifest.get("stale_after_days") in {None, ""}
            else int(manifest.get("stale_after_days"))
        ),
        "expected_component_families": [
            str(item).strip()
            for item in manifest.get("expected_component_families") or ()
            if str(item).strip()
        ],
        "defaults": {
            key: value
            for key, value in defaults.items()
            if _normalize_token(value)
        },
        "metadata": _json_safe(manifest.get("metadata") or {}),
        "discount_curves": discount_curves,
        "forecast_curves": forecast_curves,
        "vol_surfaces": vol_surfaces,
        "credit_curves": credit_curves,
        "fx_rates": fx_rates,
        "underlier_spots": underlier_spots,
        "fixing_histories": fixing_histories,
        "source_files": {
            "manifest": str(path),
            "discount_curves": discount_files,
            "forecast_curves": forecast_files,
            "vol_surfaces": vol_files,
            "credit_curves": credit_files,
            "fx_rates": fx_files,
            "underlier_spots": spot_files,
            "fixing_histories": fixing_files,
        },
    }
    intrinsic = tuple(dict.fromkeys([*warnings, *_intrinsic_warnings(manifest_contract)]))
    if intrinsic:
        manifest_contract["intrinsic_warnings"] = list(intrinsic)
    return manifest_contract


def import_snapshot_manifest(
    manifest_path: Path | str,
    *,
    reference_date: date | str | None = None,
) -> tuple[MarketSnapshot, dict[str, object], dict[str, object], tuple[str, ...]]:
    manifest_contract = load_manifest_contract(manifest_path)
    snapshot_id = _snapshot_id(manifest_contract)
    snapshot = build_market_snapshot(
        manifest_contract,
        snapshot_id=snapshot_id,
    )
    return (
        snapshot,
        manifest_contract,
        manifest_summary(manifest_contract),
        manifest_warnings(manifest_contract, reference_date=reference_date),
    )


def load_snapshot_from_record(record) -> MarketSnapshot:
    manifest = dict(getattr(record, "payload", {}).get("manifest") or {})
    if not manifest:
        raise ValueError(
            f"Snapshot record {getattr(record, 'snapshot_id', '')!r} has no import manifest payload."
        )
    return build_market_snapshot(manifest, snapshot_id=getattr(record, "snapshot_id", None))


__all__ = [
    "FILE_IMPORT_BUNDLE_TYPE",
    "FILE_IMPORT_PROVIDER_ID",
    "FILE_IMPORT_SCHEMA_VERSION",
    "FILE_IMPORT_SOURCE",
    "build_market_snapshot",
    "import_snapshot_manifest",
    "load_manifest_contract",
    "load_snapshot_from_record",
    "manifest_summary",
    "manifest_warnings",
]
