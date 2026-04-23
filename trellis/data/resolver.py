"""Market-data resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from trellis.core.differentiable import get_numpy
from trellis.curves.bootstrap import (
    BootstrapCurveInputBundle,
    BootstrapInstrument,
    DatedBootstrapCurveInputBundle,
    MultiCurveBootstrapProgram,
    bootstrap_multi_curve_program,
    bootstrap_named_curve_results,
)
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot

np = get_numpy()


def _normalize_token(value: object | None) -> str:
    """Return a trimmed string token."""
    return str(value or "").strip()


def _normalize_as_of(as_of: date | str | None = None) -> date:
    """Normalize market-data date inputs."""
    if as_of is None or as_of == "latest":
        return date.today()
    if isinstance(as_of, str):
        return date.fromisoformat(as_of)
    return as_of


def _provider_for_source(source: str):
    """Instantiate the provider for a supported market-data source."""
    if source == "mock":
        from trellis.data.mock import MockDataProvider

        return MockDataProvider()
    if source == "fred":
        try:
            from trellis.data.fred import FredDataProvider

            return FredDataProvider()
        except ImportError:
            import warnings

            warnings.warn(
                "fredapi not installed; falling back to mock data",
                stacklevel=2,
            )
            from trellis.data.mock import MockDataProvider

            return MockDataProvider()
    if source == "treasury_gov":
        try:
            from trellis.data.treasury_gov import TreasuryGovDataProvider

            return TreasuryGovDataProvider()
        except ImportError:
            import warnings

            warnings.warn(
                "requests not installed; falling back to mock data",
                stacklevel=2,
                )
            from trellis.data.mock import MockDataProvider

            return MockDataProvider()
    raise ValueError(f"Unknown data source: {source!r}")


def _bootstrap_input_summary(
    curve_inputs: list[BootstrapInstrument] | BootstrapCurveInputBundle | DatedBootstrapCurveInputBundle,
    *,
    curve_name: str,
) -> dict[str, object]:
    """Return a deterministic, JSON-friendly summary of bootstrap inputs."""
    if isinstance(curve_inputs, (BootstrapCurveInputBundle, DatedBootstrapCurveInputBundle)):
        bundle = curve_inputs.with_curve_name(curve_name)
    else:
        bundle = BootstrapCurveInputBundle(
            instruments=tuple(curve_inputs),
            curve_name=curve_name,
        )
    return bundle.to_payload()


def _market_provenance(
    source: str,
    as_of: date,
    *,
    source_kind: str,
    source_ref: str,
) -> dict[str, object]:
    """Build the canonical provenance payload for a resolved market snapshot."""
    return {
        "source": source,
        "as_of": as_of.isoformat(),
        "source_kind": source_kind,
        "source_ref": source_ref,
    }


def _required_float(
    payload: Mapping[str, object],
    *,
    field: str,
    source_name: str,
    parameter_name: str,
) -> float:
    """Return one required float field from a bootstrap model-parameter entry."""
    raw_value = payload.get(field)
    if raw_value is None or _normalize_token(raw_value) == "":
        raise ValueError(
            f"Bootstrap model-parameter source {source_name!r} entry {parameter_name!r} requires {field!r}."
        )
    return float(raw_value)


def _normalize_named_string_mapping(
    payload: Mapping[str, object] | None,
) -> dict[str, str]:
    """Return a stable string-to-string mapping."""
    normalized: dict[str, str] = {}
    for raw_key, raw_value in dict(payload or {}).items():
        key = _normalize_token(raw_key)
        value = _normalize_token(raw_value)
        if key and value:
            normalized[key] = value
    return normalized


def _normalize_empirical_window(
    empirical_inputs: Mapping[str, object],
    *,
    source_name: str,
) -> dict[str, object]:
    """Return a shallow normalized window payload for empirical sources."""
    raw_window = empirical_inputs.get("window")
    if raw_window is None:
        return {}
    if not isinstance(raw_window, Mapping):
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} requires 'window' to be a mapping when provided."
        )
    return {
        _normalize_token(key): value
        for key, value in dict(raw_window).items()
        if _normalize_token(key)
    }


def _normalize_empirical_series_names(value) -> tuple[str, ...]:
    """Return one stable ordered tuple of series names."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = _normalize_token(value)
        return (text,) if text else ()
    names: list[str] = []
    for raw_item in value:
        item = _normalize_token(raw_item)
        if item:
            names.append(item)
    return tuple(names)


def _lookup_observation_series(
    observations: Mapping[str, object],
    *,
    series_name: str,
    source_name: str,
    parameter_name: str,
) -> tuple[str, object]:
    """Resolve one named empirical series from an observation mapping."""
    key_candidates = (
        series_name,
        series_name.upper(),
        series_name.lower(),
        series_name.replace(" ", "_"),
    )
    for candidate in key_candidates:
        if candidate in observations:
            return candidate, observations[candidate]
    raise ValueError(
        f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} is missing observations for {series_name!r}."
    )


def _coerce_empirical_observation_matrix(
    observations: object,
    *,
    source_name: str,
    parameter_name: str,
    series_names: tuple[str, ...] | None = None,
) -> tuple[object, tuple[str, ...]]:
    """Normalize empirical observations onto a samples-by-series array."""
    requested_names = tuple(series_names or ())
    if isinstance(observations, Mapping):
        normalized_names = requested_names or tuple(
            _normalize_token(name)
            for name in observations
            if _normalize_token(name)
        )
        if not normalized_names:
            raise ValueError(
                f"Empirical model-parameter source {source_name!r} requires at least one observed series."
            )
        series: list[object] = []
        sample_size: int | None = None
        for name in normalized_names:
            _, raw_series = _lookup_observation_series(
                observations,
                series_name=name,
                source_name=source_name,
                parameter_name=parameter_name,
            )
            arr = np.asarray(raw_series, dtype=float).reshape(-1)
            if arr.size < 2:
                raise ValueError(
                    f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires at least two samples for {name!r}."
                )
            if sample_size is None:
                sample_size = int(arr.size)
            elif sample_size != int(arr.size):
                raise ValueError(
                    f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires all observed series to share a common sample size."
                )
            series.append(arr)
        return np.column_stack(series), normalized_names

    data = np.asarray(observations, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        if requested_names and len(requested_names) != 1:
            raise ValueError(
                f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires exactly one series name for one-dimensional observations."
            )
        normalized_names = requested_names or ("series_0",)
    elif data.ndim == 2:
        if requested_names:
            if data.shape[1] == len(requested_names):
                normalized_names = requested_names
            elif data.shape[0] == len(requested_names):
                data = data.T
                normalized_names = requested_names
            else:
                raise ValueError(
                    f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} observations must align with {len(requested_names)} named series; got {data.shape}."
                )
        else:
            normalized_names = tuple(f"series_{index}" for index in range(int(data.shape[1])))
    else:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} observations must be one- or two-dimensional."
        )

    if data.shape[0] < 2:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires at least two samples."
        )
    return data, normalized_names


def _nearest_pd_correlation_matrix(
    matrix,
    *,
    floor: float,
):
    """Project a symmetric matrix onto the nearest positive-definite correlation matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    clipped = np.maximum(eigvals, floor)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    repaired = 0.5 * (repaired + repaired.T)
    scale = np.sqrt(np.clip(np.diag(repaired), floor, None))
    repaired = repaired / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _build_empirical_descriptor(
    value: object,
    *,
    estimator: str,
    sample_size: int,
    source_ref: str,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Wrap one empirical estimate in a portable runtime descriptor."""
    payload: dict[str, object] = {
        "kind": "empirical",
        "estimator": estimator,
        "sample_size": int(sample_size),
        "source_ref": source_ref,
        "parameters": dict(parameters),
    }
    if np.asarray(value, dtype=float).ndim == 0:
        payload["value"] = float(value)
    else:
        payload["correlation_matrix"] = [
            [float(cell) for cell in row]
            for row in np.asarray(value, dtype=float)
        ]
    return payload


def _resolve_empirical_parameter_entry(
    entry: Mapping[str, object],
    *,
    source_name: str,
    source_ref: str,
    empirical_inputs: Mapping[str, object],
) -> tuple[str, object, dict[str, object]]:
    """Resolve one empirical model-parameter entry onto a runtime value."""
    parameter_name = _normalize_token(entry.get("parameter"))
    if not parameter_name:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} has an entry missing 'parameter'."
        )
    measure = _normalize_token(entry.get("measure")).lower()
    if not measure:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires 'measure'."
        )
    observations = empirical_inputs.get("observations")
    if observations is None:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} requires empirical_inputs.observations."
        )
    window = _normalize_empirical_window(
        empirical_inputs,
        source_name=source_name,
    )
    source_paths = _normalize_named_string_mapping(empirical_inputs.get("source_paths"))
    series_names = _normalize_empirical_series_names(
        entry.get("series_names", entry.get("series_name"))
    )
    descriptor = bool(entry.get("descriptor", False))

    if measure == "pairwise_correlation":
        if len(series_names) != 2:
            raise ValueError(
                f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires exactly two series_names for pairwise_correlation."
            )
        estimator = _normalize_token(entry.get("estimator") or "sample_pearson").lower()
        if estimator != "sample_pearson":
            raise ValueError(
                f"Unsupported empirical estimator {estimator!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
            )
        data, normalized_names = _coerce_empirical_observation_matrix(
            observations,
            source_name=source_name,
            parameter_name=parameter_name,
            series_names=series_names,
        )
        sample_size = int(data.shape[0])
        value = float(np.corrcoef(data, rowvar=False)[0, 1])
        metadata = {
            "measure": "pairwise_correlation",
            "estimator": estimator,
            "sample_size": sample_size,
            "series_names": list(normalized_names),
            "observation_shape": [int(value) for value in data.shape],
        }
        filtered_paths = {
            name: source_paths[name]
            for name in normalized_names
            if name in source_paths
        }
        if filtered_paths:
            metadata["source_paths"] = filtered_paths
        if window:
            metadata["window"] = dict(window)
        resolved_value: object = value
        if descriptor:
            resolved_value = _build_empirical_descriptor(
                value,
                estimator=estimator,
                sample_size=sample_size,
                source_ref=source_ref,
                parameters=metadata,
            )
        return parameter_name, resolved_value, metadata

    if measure == "realized_vol":
        if len(series_names) not in {0, 1}:
            raise ValueError(
                f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires at most one series name for realized_vol."
            )
        estimator = _normalize_token(entry.get("estimator") or "sample_std").lower()
        if estimator != "sample_std":
            raise ValueError(
                f"Unsupported empirical estimator {estimator!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
            )
        data, normalized_names = _coerce_empirical_observation_matrix(
            observations,
            source_name=source_name,
            parameter_name=parameter_name,
            series_names=series_names if series_names else None,
        )
        if data.shape[1] != 1:
            raise ValueError(
                f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} realized_vol requires a single observed series."
            )
        annualization_factor = float(entry.get("annualization_factor", 252.0))
        sample_size = int(data.shape[0])
        value = float(np.std(data[:, 0], ddof=1) * np.sqrt(max(annualization_factor, 0.0)))
        metadata = {
            "measure": "realized_vol",
            "estimator": estimator,
            "sample_size": sample_size,
            "series_names": list(normalized_names),
            "annualization_factor": annualization_factor,
            "observation_shape": [int(value) for value in data.shape],
        }
        filtered_paths = {
            name: source_paths[name]
            for name in normalized_names
            if name in source_paths
        }
        if filtered_paths:
            metadata["source_paths"] = filtered_paths
        if window:
            metadata["window"] = dict(window)
        resolved_value = value
        if descriptor:
            resolved_value = _build_empirical_descriptor(
                value,
                estimator=estimator,
                sample_size=sample_size,
                source_ref=source_ref,
                parameters=metadata,
            )
        return parameter_name, resolved_value, metadata

    if measure == "correlation_matrix":
        estimator = _normalize_token(entry.get("estimator") or "sample_pearson").lower()
        if estimator != "sample_pearson":
            raise ValueError(
                f"Unsupported empirical estimator {estimator!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
            )
        data, normalized_names = _coerce_empirical_observation_matrix(
            observations,
            source_name=source_name,
            parameter_name=parameter_name,
            series_names=series_names if series_names else None,
        )
        sample_size = int(data.shape[0])
        matrix = np.corrcoef(data, rowvar=False)
        regularization_meta: dict[str, object] = {}
        regularization = entry.get("regularization")
        if regularization is not None:
            if not isinstance(regularization, Mapping):
                raise ValueError(
                    f"Empirical model-parameter source {source_name!r} entry {parameter_name!r} requires 'regularization' to be a mapping when provided."
                )
            regularization_kind = _normalize_token(regularization.get("kind") or "nearest_pd").lower()
            if regularization_kind != "nearest_pd":
                raise ValueError(
                    f"Unsupported empirical regularization {regularization_kind!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
                )
            floor = float(regularization.get("floor", 1e-12))
            min_before = float(np.min(np.linalg.eigvalsh(matrix))) if matrix.size else 1.0
            matrix = _nearest_pd_correlation_matrix(matrix, floor=floor)
            min_after = float(np.min(np.linalg.eigvalsh(matrix))) if matrix.size else 1.0
            regularization_meta = {
                "kind": regularization_kind,
                "floor": floor,
                "min_eigenvalue_before": min_before,
                "min_eigenvalue_after": min_after,
            }
        value = tuple(
            tuple(float(cell) for cell in row)
            for row in np.asarray(matrix, dtype=float)
        )
        metadata = {
            "measure": "correlation_matrix",
            "estimator": estimator,
            "sample_size": sample_size,
            "series_names": list(normalized_names),
            "observation_shape": [int(value) for value in data.shape],
        }
        filtered_paths = {
            name: source_paths[name]
            for name in normalized_names
            if name in source_paths
        }
        if filtered_paths:
            metadata["source_paths"] = filtered_paths
        if window:
            metadata["window"] = dict(window)
        if regularization_meta:
            metadata["regularization"] = regularization_meta
        resolved_value = value
        if descriptor:
            resolved_value = _build_empirical_descriptor(
                value,
                estimator=estimator,
                sample_size=sample_size,
                source_ref=source_ref,
                parameters=metadata,
            )
        return parameter_name, resolved_value, metadata

    raise ValueError(
        f"Unsupported empirical measure {measure!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
    )


def _resolve_empirical_parameter_source(
    *,
    source_name: str,
    source_spec: Mapping[str, object],
    source_ref: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve one empirical model-parameter source onto runtime parameters and provenance."""
    if source_spec.get("parameters") is not None:
        raise ValueError(
            f"Unsupported source combination for model-parameter source {source_name!r}: empirical sources cannot declare direct parameters."
        )
    if source_spec.get("bootstrap_inputs") is not None:
        raise ValueError(
            f"Unsupported source combination for model-parameter source {source_name!r}: empirical sources cannot declare bootstrap_inputs."
        )
    empirical_inputs = source_spec.get("empirical_inputs")
    if not isinstance(empirical_inputs, Mapping):
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} requires a mapping payload under 'empirical_inputs'."
        )
    raw_entries = source_spec.get("entries")
    if raw_entries is None:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} requires non-empty entries."
        )
    if isinstance(raw_entries, Mapping):
        entry_items = (raw_entries,)
    else:
        entry_items = tuple(raw_entries)
    if not entry_items:
        raise ValueError(
            f"Empirical model-parameter source {source_name!r} requires non-empty entries."
        )

    parameters: dict[str, object] = {}
    empirical_outputs: dict[str, object] = {}
    for raw_entry in entry_items:
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"Empirical model-parameter source {source_name!r} entries must be mapping payloads."
            )
        parameter_name, value, metadata = _resolve_empirical_parameter_entry(
            raw_entry,
            source_name=source_name,
            source_ref=source_ref,
            empirical_inputs=empirical_inputs,
        )
        if parameter_name in parameters:
            raise ValueError(
                f"Duplicate empirical parameter {parameter_name!r} in model-parameter source {source_name!r}."
            )
        parameters[parameter_name] = value
        empirical_outputs[parameter_name] = dict(metadata)

    source_provenance = {
        "source_kind": "empirical",
        "source_ref": source_ref,
        "empirical_inputs": {
            "input_key": "observations",
            "window": _normalize_empirical_window(empirical_inputs, source_name=source_name),
            "source_paths": _normalize_named_string_mapping(empirical_inputs.get("source_paths")),
        },
        "empirical_outputs": empirical_outputs,
        "parameters": dict(parameters),
    }
    return dict(parameters), source_provenance


def _coerce_optional_mapping(
    payload: object | None,
    *,
    field: str,
    source_name: str,
) -> Mapping[str, object]:
    """Return one optional mapping field or raise with a clear source-specific error."""
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Calibration model-parameter source {source_name!r} requires {field!r} to be a mapping when provided."
        )
    return payload


def _resolve_heston_calibration_source(
    *,
    source_name: str,
    source_ref: str,
    calibration_inputs: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve one Heston smile calibration source onto runtime parameters and provenance."""
    from trellis.models.calibration.heston_fit import calibrate_heston_smile_workflow

    surface = calibration_inputs.get("surface")
    if not isinstance(surface, Mapping):
        raise ValueError(
            f"Calibration model-parameter source {source_name!r} requires a mapping payload under calibration_inputs.surface."
        )
    options = _coerce_optional_mapping(
        calibration_inputs.get("options"),
        field="calibration_inputs.options",
        source_name=source_name,
    )
    merged_metadata: dict[str, object] = {}
    for metadata_payload in (
        calibration_inputs.get("metadata"),
        surface.get("metadata"),
    ):
        if metadata_payload is None:
            continue
        if not isinstance(metadata_payload, Mapping):
            raise ValueError(
                f"Calibration model-parameter source {source_name!r} requires metadata fields to be mappings when provided."
            )
        merged_metadata.update({
            _normalize_token(key): value
            for key, value in dict(metadata_payload).items()
            if _normalize_token(key)
        })

    parameter_set_name = _normalize_token(options.get("parameter_set_name")) or source_name
    initial_guess = options.get("initial_guess")
    warm_start = options.get("warm_start")
    result = calibrate_heston_smile_workflow(
        float(surface.get("spot")),
        float(surface.get("expiry_years")),
        surface.get("strikes") or (),
        surface.get("market_vols") or (),
        rate=float(surface.get("rate")),
        dividend_yield=float(surface.get("dividend_yield", 0.0)),
        labels=surface.get("labels"),
        weights=surface.get("weights"),
        surface_name=_normalize_token(surface.get("surface_name")),
        parameter_set_name=parameter_set_name,
        initial_guess=tuple(initial_guess) if initial_guess is not None else None,
        warm_start=tuple(warm_start) if warm_start is not None else None,
        metadata=merged_metadata or None,
    )
    parameters = dict(result.model_parameters)
    source_provenance = {
        "source_kind": "calibration",
        "source_ref": source_ref,
        "calibration_source": {
            "workflow": "heston_smile",
            "surface": result.surface.to_payload(),
            "options": {
                "parameter_set_name": parameter_set_name,
                "initial_guess": list(tuple(initial_guess)) if initial_guess is not None else None,
                "warm_start": list(tuple(warm_start)) if warm_start is not None else None,
            },
        },
        "calibration_result": {
            "source_kind": str(result.provenance.get("source_kind", "calibrated_surface")),
            "source_ref": str(result.provenance.get("source_ref", "calibrate_heston_smile_workflow")),
            "calibration_target": dict(result.provenance.get("calibration_target", {})),
            "solve_request": result.solve_request.to_payload(),
            "solve_result": result.solve_result.to_payload(),
            "solver_provenance": result.solver_provenance.to_payload(),
            "solver_replay_artifact": result.solver_replay_artifact.to_payload(),
            "fit_diagnostics": result.diagnostics.to_payload(),
            "summary": dict(result.summary),
            "warnings": list(result.warnings),
        },
        "parameters": dict(parameters),
    }
    return parameters, source_provenance


def _resolve_calibration_parameter_source(
    *,
    source_name: str,
    source_spec: Mapping[str, object],
    source_ref: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve one calibration-backed model-parameter source."""
    if source_spec.get("parameters") is not None:
        raise ValueError(
            f"Unsupported source combination for model-parameter source {source_name!r}: calibration sources cannot declare direct parameters."
        )
    if source_spec.get("bootstrap_inputs") is not None:
        raise ValueError(
            f"Unsupported source combination for model-parameter source {source_name!r}: calibration sources cannot declare bootstrap_inputs."
        )
    if source_spec.get("empirical_inputs") is not None:
        raise ValueError(
            f"Unsupported source combination for model-parameter source {source_name!r}: calibration sources cannot declare empirical_inputs."
        )
    calibration_inputs = source_spec.get("calibration_inputs")
    if not isinstance(calibration_inputs, Mapping):
        raise ValueError(
            f"Calibration model-parameter source {source_name!r} requires a mapping payload under 'calibration_inputs'."
        )
    workflow = _normalize_token(calibration_inputs.get("workflow")).lower()
    if workflow == "heston_smile":
        return _resolve_heston_calibration_source(
            source_name=source_name,
            source_ref=source_ref,
            calibration_inputs=calibration_inputs,
        )
    raise ValueError(
        f"Unsupported calibration workflow {workflow!r} for model-parameter source {source_name!r}."
    )


def _curve_family_map(
    family: str,
    *,
    discount_curves: Mapping[str, object],
    forecast_curves: Mapping[str, object],
    source_name: str,
    parameter_name: str,
) -> Mapping[str, object]:
    """Return the selected curve family mapping for one bootstrap source entry."""
    if family == "discount_curves":
        return discount_curves
    if family == "forecast_curves":
        return forecast_curves
    raise ValueError(
        f"Unsupported bootstrap curve family {family!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
    )


def _resolve_bootstrap_parameter_entry(
    entry: Mapping[str, object],
    *,
    source_name: str,
    discount_curves: Mapping[str, object],
    forecast_curves: Mapping[str, object],
) -> tuple[str, float, dict[str, object]]:
    """Resolve one bootstrap model-parameter entry onto a numeric parameter value."""
    parameter_name = _normalize_token(entry.get("parameter"))
    if not parameter_name:
        raise ValueError(
            f"Bootstrap model-parameter source {source_name!r} has an entry missing 'parameter'."
        )
    family = _normalize_token(entry.get("curve_family"))
    curve_name = _normalize_token(entry.get("curve_name"))
    measure = _normalize_token(entry.get("measure")).lower()
    if not family or not curve_name or not measure:
        raise ValueError(
            f"Bootstrap model-parameter source {source_name!r} entry {parameter_name!r} requires curve_family, curve_name, and measure."
        )
    curve_map = _curve_family_map(
        family,
        discount_curves=discount_curves,
        forecast_curves=forecast_curves,
        source_name=source_name,
        parameter_name=parameter_name,
    )
    if curve_name not in curve_map:
        raise ValueError(
            f"Unknown curve_name {curve_name!r} for model-parameter source {source_name!r} entry {parameter_name!r}."
        )
    curve = curve_map[curve_name]
    if measure == "zero_rate":
        tenor = _required_float(
            entry,
            field="tenor",
            source_name=source_name,
            parameter_name=parameter_name,
        )
        value = float(curve.zero_rate(tenor))
        normalized_entry = {
            "parameter": parameter_name,
            "curve_family": family,
            "curve_name": curve_name,
            "measure": "zero_rate",
            "tenor": tenor,
        }
    elif measure == "discount_factor":
        tenor = _required_float(
            entry,
            field="tenor",
            source_name=source_name,
            parameter_name=parameter_name,
        )
        value = float(curve.discount(tenor))
        normalized_entry = {
            "parameter": parameter_name,
            "curve_family": family,
            "curve_name": curve_name,
            "measure": "discount_factor",
            "tenor": tenor,
        }
    elif measure == "forward_rate":
        start_tenor = _required_float(
            entry,
            field="start_tenor",
            source_name=source_name,
            parameter_name=parameter_name,
        )
        end_tenor = _required_float(
            entry,
            field="end_tenor",
            source_name=source_name,
            parameter_name=parameter_name,
        )
        if end_tenor <= start_tenor:
            raise ValueError(
                f"Bootstrap model-parameter source {source_name!r} entry {parameter_name!r} requires end_tenor > start_tenor."
            )
        value = float(curve.forward_rate(start_tenor, end_tenor))
        normalized_entry = {
            "parameter": parameter_name,
            "curve_family": family,
            "curve_name": curve_name,
            "measure": "forward_rate",
            "start_tenor": start_tenor,
            "end_tenor": end_tenor,
        }
    else:
        raise ValueError(
            f"Unsupported bootstrap model-parameter measure {measure!r} for source {source_name!r} entry {parameter_name!r}."
        )
    return parameter_name, value, normalized_entry


def _resolve_model_parameter_source(
    *,
    source_name: str,
    source_spec: Mapping[str, object],
    discount_curves: Mapping[str, object],
    forecast_curves: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None]:
    """Resolve one model-parameter source spec onto runtime parameters and provenance."""
    source_kind = _normalize_token(source_spec.get("source_kind")).lower()
    if source_kind not in {"direct_quote", "bootstrap", "empirical", "calibration"}:
        raise ValueError(
            f"Unsupported model-parameter source kind {source_kind!r} for source {source_name!r}."
        )
    source_ref = _normalize_token(source_spec.get("source_ref")) or f"resolver.model_parameter_sources.{source_name}"
    if source_kind == "direct_quote":
        if source_spec.get("bootstrap_inputs") is not None:
            raise ValueError(
                f"Unsupported source combination for model-parameter source {source_name!r}: direct_quote cannot declare bootstrap_inputs."
            )
        direct_parameters = source_spec.get("parameters")
        if not isinstance(direct_parameters, Mapping):
            raise ValueError(
                f"Direct-quote model-parameter source {source_name!r} requires a mapping payload under 'parameters'."
            )
        parameters = {
            _normalize_token(key): value
            for key, value in dict(direct_parameters).items()
            if _normalize_token(key)
        }
        if not parameters:
            raise ValueError(
                f"Direct-quote model-parameter source {source_name!r} must contain at least one parameter."
            )
        return (
            dict(parameters),
            {
                "source_kind": "direct_quote",
                "source_ref": source_ref,
                "parameters": dict(parameters),
            },
            None,
        )

    if source_kind == "empirical":
        parameters, source_provenance = _resolve_empirical_parameter_source(
            source_name=source_name,
            source_spec=source_spec,
            source_ref=source_ref,
        )
        return parameters, source_provenance, None

    if source_kind == "calibration":
        parameters, source_provenance = _resolve_calibration_parameter_source(
            source_name=source_name,
            source_spec=source_spec,
            source_ref=source_ref,
        )
        return parameters, source_provenance, None

    if source_spec.get("parameters") is not None:
        raise ValueError(
            f"Unsupported source combination for model-parameter source {source_name!r}: bootstrap sources cannot declare direct parameters."
        )
    bootstrap_inputs = source_spec.get("bootstrap_inputs")
    if not isinstance(bootstrap_inputs, Mapping):
        raise ValueError(
            f"Bootstrap model-parameter source {source_name!r} requires a mapping payload under 'bootstrap_inputs'."
        )
    raw_entries = bootstrap_inputs.get("entries")
    if raw_entries is None:
        raise ValueError(
            f"Bootstrap model-parameter source {source_name!r} requires non-empty bootstrap_inputs.entries."
        )
    if isinstance(raw_entries, Mapping):
        entry_items = (raw_entries,)
    else:
        entry_items = tuple(raw_entries)
    if not entry_items:
        raise ValueError(
            f"Bootstrap model-parameter source {source_name!r} requires non-empty bootstrap_inputs.entries."
        )
    parameters: dict[str, object] = {}
    normalized_entries: list[dict[str, object]] = []
    for raw_entry in entry_items:
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"Bootstrap model-parameter source {source_name!r} entries must be mapping payloads."
            )
        parameter_name, value, normalized_entry = _resolve_bootstrap_parameter_entry(
            raw_entry,
            source_name=source_name,
            discount_curves=discount_curves,
            forecast_curves=forecast_curves,
        )
        if parameter_name in parameters:
            raise ValueError(
                f"Duplicate bootstrap parameter {parameter_name!r} in model-parameter source {source_name!r}."
            )
        parameters[parameter_name] = value
        normalized_entries.append(normalized_entry)
    normalized_bootstrap_inputs = {
        "entries": list(normalized_entries),
    }
    return (
        dict(parameters),
        {
            "source_kind": "bootstrap",
            "source_ref": source_ref,
            "bootstrap_inputs": dict(normalized_bootstrap_inputs),
            "parameters": dict(parameters),
            },
            normalized_bootstrap_inputs,
        )


def resolve_market_snapshot(
    as_of: date | str | None = None,
    source: str = "treasury_gov",
    *,
    provider=None,
    vol_surface=None,
    vol_surfaces: dict | None = None,
    default_vol_surface: str | None = None,
    forecast_curves: dict | None = None,
    fixing_histories: dict | None = None,
    default_fixing_history: str | None = None,
    discount_curve_bootstraps: dict | None = None,
    forecast_curve_bootstraps: dict | None = None,
    multi_curve_bootstrap_program: MultiCurveBootstrapProgram | None = None,
    credit_curve=None,
    fx_rates: dict | None = None,
    state_space=None,
    state_spaces: dict | None = None,
    default_state_space: str | None = None,
    underlier_spots: dict | None = None,
    default_underlier_spot: str | None = None,
    local_vol_surface=None,
    local_vol_surfaces: dict | None = None,
    default_local_vol_surface: str | None = None,
    jump_parameters: dict | None = None,
    jump_parameter_sets: dict | None = None,
    default_jump_parameters: str | None = None,
    model_parameters: dict | None = None,
    model_parameter_sets: dict | None = None,
    model_parameter_sources: dict | None = None,
    default_model_parameters: str | None = None,
    metadata: dict | None = None,
) -> MarketSnapshot:
    """Resolve a canonical market snapshot.

    Parameters
    ----------
    as_of : date, str, or None
        ``"latest"`` / ``None`` → today; string → parsed as ISO date.
    source : str
        ``"treasury_gov"``, ``"fred"``, or ``"mock"``.
    """
    resolved_date = _normalize_as_of(as_of)
    provider = provider or _provider_for_source(source)

    if vol_surface is not None and vol_surfaces is not None:
        raise ValueError("Pass either vol_surface= or vol_surfaces=, not both")
    if state_space is not None and state_spaces is not None:
        raise ValueError("Pass either state_space= or state_spaces=, not both")
    if local_vol_surface is not None and local_vol_surfaces is not None:
        raise ValueError("Pass either local_vol_surface= or local_vol_surfaces=, not both")
    if jump_parameters is not None and jump_parameter_sets is not None:
        raise ValueError("Pass either jump_parameters= or jump_parameter_sets=, not both")
    if model_parameters is not None and model_parameter_sets is not None:
        raise ValueError("Pass either model_parameters= or model_parameter_sets=, not both")
    if multi_curve_bootstrap_program is not None and (
        discount_curve_bootstraps is not None or forecast_curve_bootstraps is not None
    ):
        raise ValueError(
            "Pass either multi_curve_bootstrap_program= or discount_curve_bootstraps=/forecast_curve_bootstraps=, not both"
        )
    if model_parameter_sources is not None and (
        model_parameters is not None or model_parameter_sets is not None
    ):
        raise ValueError(
            "Pass either model_parameter_sources= or model_parameters=/model_parameter_sets=, not both"
        )

    used_provider_snapshot = False
    try:
        base_snapshot = provider.fetch_market_snapshot(resolved_date)
    except NotImplementedError:
        base_snapshot = None
    else:
        used_provider_snapshot = isinstance(base_snapshot, MarketSnapshot)
    if not isinstance(base_snapshot, MarketSnapshot):
        base_snapshot = None

    if base_snapshot is None:
        yields = provider.fetch_yields(resolved_date)
        if not yields:
            raise RuntimeError(
                f"No yield data returned from {source!r} for as_of={resolved_date}"
            )
        discount_curve = YieldCurve.from_treasury_yields(yields)
        base_snapshot = MarketSnapshot(
            as_of=resolved_date,
            source=source,
            discount_curves={"discount": discount_curve},
            default_discount_curve="discount",
            provenance=_market_provenance(
                source,
                resolved_date,
                source_kind="synthetic_snapshot" if source == "mock" else "direct_quote",
                source_ref="fetch_yields",
            ),
        )

    if not base_snapshot.discount_curves:
        raise RuntimeError(
            f"No yield data returned from {source!r} for as_of={resolved_date}"
        )

    resolved_forecast_curves = dict(base_snapshot.forecast_curves)
    resolved_fx_rates = dict(base_snapshot.fx_rates)
    resolved_fixing_histories = {
        key: dict(value) for key, value in base_snapshot.fixing_histories.items()
    }
    resolved_state_spaces = dict(base_snapshot.state_spaces)
    resolved_metadata = dict(base_snapshot.metadata)
    resolved_vol_surfaces = dict(base_snapshot.vol_surfaces)
    resolved_credit_curves = dict(base_snapshot.credit_curves)
    resolved_underlier_spots = dict(base_snapshot.underlier_spots)
    resolved_local_vol_surfaces = dict(base_snapshot.local_vol_surfaces)
    resolved_jump_parameter_sets = {
        key: dict(value) for key, value in base_snapshot.jump_parameter_sets.items()
    }
    resolved_model_parameter_sets = {
        key: dict(value) for key, value in base_snapshot.model_parameter_sets.items()
    }
    resolved_default_vol_surface = base_snapshot.default_vol_surface
    resolved_default_credit_curve = base_snapshot.default_credit_curve
    resolved_default_fixing_history = base_snapshot.default_fixing_history
    resolved_default_state_space = base_snapshot.default_state_space
    resolved_default_underlier_spot = base_snapshot.default_underlier_spot
    resolved_default_local_vol_surface = base_snapshot.default_local_vol_surface
    resolved_default_jump_parameters = base_snapshot.default_jump_parameters
    resolved_default_model_parameters = base_snapshot.default_model_parameters
    resolved_provenance = dict(base_snapshot.provenance or {})
    if not resolved_provenance:
        resolved_provenance = _market_provenance(
            base_snapshot.source,
            base_snapshot.as_of,
            source_kind=(
                "synthetic_snapshot"
                if source == "mock"
                else ("provider_snapshot" if used_provider_snapshot else "direct_quote")
            ),
            source_ref=(
                "fetch_market_snapshot" if used_provider_snapshot else "fetch_yields"
            ),
        )
    else:
        resolved_provenance.setdefault("source", base_snapshot.source)
        resolved_provenance.setdefault("as_of", base_snapshot.as_of.isoformat())
        resolved_provenance.setdefault(
            "source_ref",
            "fetch_market_snapshot" if used_provider_snapshot else "fetch_yields",
        )
        if "source_kind" not in resolved_provenance:
            resolved_provenance["source_kind"] = (
                "synthetic_snapshot"
                if source == "mock"
                else ("provider_snapshot" if used_provider_snapshot else "direct_quote")
            )

    bootstrap_inputs = dict(resolved_provenance.get("bootstrap_inputs") or {})
    bootstrap_runs = dict(resolved_provenance.get("bootstrap_runs") or {})
    market_parameter_sources = dict(resolved_provenance.get("market_parameter_sources") or {})
    explicit_inputs_present = any(
        value is not None
        for value in (
            forecast_curves,
            fixing_histories,
            multi_curve_bootstrap_program,
            credit_curve,
            fx_rates,
            state_space,
            state_spaces,
            underlier_spots,
            vol_surface,
            vol_surfaces,
            local_vol_surface,
            local_vol_surfaces,
            jump_parameters,
            jump_parameter_sets,
            model_parameters,
            model_parameter_sets,
            model_parameter_sources,
        )
    )

    if forecast_curves:
        resolved_forecast_curves.update(forecast_curves)
    if multi_curve_bootstrap_program is not None:
        multi_curve_result = bootstrap_multi_curve_program(multi_curve_bootstrap_program)
        bootstrapped_discount_curves = {
            name: result.curve
            for name, result in multi_curve_result.node_results.items()
            if getattr(result.input_bundle, "curve_role", "") == "discount_curve"
        }
        bootstrapped_forecast_curves = {
            name: result.curve
            for name, result in multi_curve_result.node_results.items()
            if getattr(result.input_bundle, "curve_role", "") == "forecast_curve"
        }
        overlap_discount = set(bootstrapped_discount_curves) & set(base_snapshot.discount_curves)
        if overlap_discount:
            raise ValueError(f"Duplicate discount curve sources: {sorted(overlap_discount)}")
        overlap_forecast = set(bootstrapped_forecast_curves) & set(resolved_forecast_curves)
        if overlap_forecast:
            raise ValueError(f"Duplicate forecast curve sources: {sorted(overlap_forecast)}")
        resolved_discount_curves = dict(base_snapshot.discount_curves)
        resolved_discount_curves.update(bootstrapped_discount_curves)
        resolved_forecast_curves.update(bootstrapped_forecast_curves)
        bootstrap_inputs["multi_curve_program"] = multi_curve_bootstrap_program.to_payload()
        bootstrap_runs["multi_curve_program"] = multi_curve_result.to_payload()
    elif discount_curve_bootstraps:
        discount_bootstrap_results = bootstrap_named_curve_results(discount_curve_bootstraps)
        bootstrapped_discount_curves = {
            name: result.curve
            for name, result in discount_bootstrap_results.items()
        }
        overlap = set(bootstrapped_discount_curves) & set(base_snapshot.discount_curves)
        if overlap:
            raise ValueError(
                f"Duplicate discount curve sources: {sorted(overlap)}"
            )
        resolved_discount_curves = dict(base_snapshot.discount_curves)
        resolved_discount_curves.update(bootstrapped_discount_curves)
        bootstrap_inputs.setdefault("discount_curves", {})
        bootstrap_runs.setdefault("discount_curves", {})
        for name, result in discount_bootstrap_results.items():
            bootstrap_inputs["discount_curves"][name] = _bootstrap_input_summary(
                discount_curve_bootstraps[name],
                curve_name=name,
            )
            bootstrap_runs["discount_curves"][name] = result.to_payload()
    else:
        resolved_discount_curves = dict(base_snapshot.discount_curves)

    if multi_curve_bootstrap_program is not None:
        pass
    elif forecast_curve_bootstraps:
        forecast_bootstrap_results = bootstrap_named_curve_results(forecast_curve_bootstraps)
        bootstrapped_forecast_curves = {
            name: result.curve
            for name, result in forecast_bootstrap_results.items()
        }
        overlap = set(bootstrapped_forecast_curves) & set(resolved_forecast_curves)
        if overlap:
            raise ValueError(
                f"Duplicate forecast curve sources: {sorted(overlap)}"
            )
        resolved_forecast_curves.update(bootstrapped_forecast_curves)
        bootstrap_inputs.setdefault("forecast_curves", {})
        bootstrap_runs.setdefault("forecast_curves", {})
        for name, result in forecast_bootstrap_results.items():
            bootstrap_inputs["forecast_curves"][name] = _bootstrap_input_summary(
                forecast_curve_bootstraps[name],
                curve_name=name,
            )
            bootstrap_runs["forecast_curves"][name] = result.to_payload()

    if fx_rates:
        resolved_fx_rates.update(fx_rates)
    if fixing_histories:
        resolved_fixing_histories.update({
            key: dict(value) for key, value in fixing_histories.items()
        })
    if default_fixing_history is not None:
        resolved_default_fixing_history = default_fixing_history
    elif resolved_default_fixing_history is None and len(resolved_fixing_histories) == 1:
        resolved_default_fixing_history = next(iter(resolved_fixing_histories))
    if state_spaces is not None:
        resolved_state_spaces.update(state_spaces)
        resolved_default_state_space = default_state_space or resolved_default_state_space
        if resolved_default_state_space is None and len(resolved_state_spaces) == 1:
            resolved_default_state_space = next(iter(resolved_state_spaces))
    elif state_space is not None:
        resolved_default_state_space = (
            default_state_space or resolved_default_state_space or "default"
        )
        resolved_state_spaces[resolved_default_state_space] = state_space
    if underlier_spots:
        resolved_underlier_spots.update(underlier_spots)
    if metadata:
        resolved_metadata.update(metadata)

    if vol_surfaces is not None:
        resolved_vol_surfaces.update(vol_surfaces)
        resolved_default_vol_surface = default_vol_surface or resolved_default_vol_surface
        if resolved_default_vol_surface is None and len(resolved_vol_surfaces) == 1:
            resolved_default_vol_surface = next(iter(resolved_vol_surfaces))
    elif vol_surface is not None:
        resolved_default_vol_surface = (
            default_vol_surface or resolved_default_vol_surface or "default"
        )
        resolved_vol_surfaces[resolved_default_vol_surface] = vol_surface

    if local_vol_surfaces is not None:
        resolved_local_vol_surfaces.update(local_vol_surfaces)
        resolved_default_local_vol_surface = (
            default_local_vol_surface or resolved_default_local_vol_surface
        )
        if (
            resolved_default_local_vol_surface is None
            and len(resolved_local_vol_surfaces) == 1
        ):
            resolved_default_local_vol_surface = next(iter(resolved_local_vol_surfaces))
    elif local_vol_surface is not None:
        resolved_default_local_vol_surface = (
            default_local_vol_surface or resolved_default_local_vol_surface or "default"
        )
        resolved_local_vol_surfaces[resolved_default_local_vol_surface] = local_vol_surface

    if credit_curve is not None:
        resolved_default_credit_curve = resolved_default_credit_curve or "default"
        resolved_credit_curves[resolved_default_credit_curve] = credit_curve
    elif resolved_default_credit_curve is None and len(resolved_credit_curves) == 1:
        resolved_default_credit_curve = next(iter(resolved_credit_curves))

    if resolved_default_underlier_spot is None and len(resolved_underlier_spots) == 1:
        resolved_default_underlier_spot = next(iter(resolved_underlier_spots))
    if default_underlier_spot is not None:
        resolved_default_underlier_spot = default_underlier_spot

    if jump_parameter_sets is not None:
        resolved_jump_parameter_sets.update({
            key: dict(value) for key, value in jump_parameter_sets.items()
        })
        resolved_default_jump_parameters = (
            default_jump_parameters or resolved_default_jump_parameters
        )
        if resolved_default_jump_parameters is None and len(resolved_jump_parameter_sets) == 1:
            resolved_default_jump_parameters = next(iter(resolved_jump_parameter_sets))
    elif jump_parameters is not None:
        resolved_default_jump_parameters = (
            default_jump_parameters or resolved_default_jump_parameters or "default"
        )
        resolved_jump_parameter_sets[resolved_default_jump_parameters] = dict(jump_parameters)

    if model_parameter_sets is not None:
        legacy_sets = {
            key: dict(value) for key, value in model_parameter_sets.items()
        }
        resolved_model_parameter_sets.update(legacy_sets)
        for name, params in legacy_sets.items():
            market_parameter_sources[name] = {
                "source_kind": "direct_quote",
                "source_ref": "resolver.model_parameter_sets",
                "parameters": dict(params),
            }
        resolved_default_model_parameters = (
            default_model_parameters or resolved_default_model_parameters
        )
        if resolved_default_model_parameters is None and len(resolved_model_parameter_sets) == 1:
            resolved_default_model_parameters = next(iter(resolved_model_parameter_sets))
    elif model_parameters is not None:
        resolved_default_model_parameters = (
            default_model_parameters or resolved_default_model_parameters or "default"
        )
        resolved_model_parameter_sets[resolved_default_model_parameters] = dict(model_parameters)
        market_parameter_sources[resolved_default_model_parameters] = {
            "source_kind": "direct_quote",
            "source_ref": "resolver.model_parameters",
            "parameters": dict(model_parameters),
        }
    elif model_parameter_sources is not None:
        normalized_sources: dict[str, Mapping[str, object]] = {}
        for raw_name, raw_spec in dict(model_parameter_sources).items():
            source_name = _normalize_token(raw_name)
            if not source_name:
                raise ValueError("Model-parameter source names must be non-empty.")
            if not isinstance(raw_spec, Mapping):
                raise ValueError(
                    f"Model-parameter source {source_name!r} must be a mapping payload."
                )
            normalized_sources[source_name] = raw_spec
        overlap = set(normalized_sources) & set(resolved_model_parameter_sets)
        if overlap:
            raise ValueError(
                f"Duplicate model parameter sources: {sorted(overlap)}"
            )
        for source_name, source_spec in normalized_sources.items():
            parameters, source_provenance, bootstrap_source_inputs = _resolve_model_parameter_source(
                source_name=source_name,
                source_spec=source_spec,
                discount_curves=resolved_discount_curves,
                forecast_curves=resolved_forecast_curves,
            )
            resolved_model_parameter_sets[source_name] = dict(parameters)
            market_parameter_sources[source_name] = dict(source_provenance)
            if bootstrap_source_inputs is not None:
                bootstrap_inputs.setdefault("model_parameters", {})
                bootstrap_inputs["model_parameters"][source_name] = dict(bootstrap_source_inputs)
        resolved_default_model_parameters = (
            default_model_parameters or resolved_default_model_parameters
        )
        if resolved_default_model_parameters is None and len(resolved_model_parameter_sets) == 1:
            resolved_default_model_parameters = next(iter(resolved_model_parameter_sets))

    if bootstrap_inputs:
        resolved_provenance["bootstrap_inputs"] = bootstrap_inputs
    if bootstrap_runs:
        resolved_provenance["bootstrap_runs"] = bootstrap_runs
    if market_parameter_sources:
        resolved_provenance["market_parameter_sources"] = market_parameter_sources
    if bootstrap_inputs or explicit_inputs_present:
        current_kind = str(resolved_provenance.get("source_kind") or "")
        if current_kind != "mixed":
            resolved_provenance["source_kind"] = "mixed"
        resolved_provenance["source_ref"] = "resolver.merged_snapshot"

    return MarketSnapshot(
        as_of=base_snapshot.as_of,
        source=base_snapshot.source,
        discount_curves=resolved_discount_curves,
        forecast_curves=resolved_forecast_curves,
        vol_surfaces=resolved_vol_surfaces,
        credit_curves=resolved_credit_curves,
        fixing_histories=resolved_fixing_histories,
        fx_rates=resolved_fx_rates,
        state_spaces=resolved_state_spaces,
        underlier_spots=resolved_underlier_spots,
        local_vol_surfaces=resolved_local_vol_surfaces,
        jump_parameter_sets=resolved_jump_parameter_sets,
        model_parameter_sets=resolved_model_parameter_sets,
        metadata=resolved_metadata,
        default_discount_curve=base_snapshot.default_discount_curve
        or (next(iter(resolved_discount_curves)) if len(resolved_discount_curves) == 1 else None),
        default_vol_surface=resolved_default_vol_surface,
        default_credit_curve=resolved_default_credit_curve,
        default_fixing_history=resolved_default_fixing_history,
        default_state_space=resolved_default_state_space,
        default_underlier_spot=resolved_default_underlier_spot,
        default_local_vol_surface=resolved_default_local_vol_surface,
        default_jump_parameters=resolved_default_jump_parameters,
        default_model_parameters=resolved_default_model_parameters,
        provenance=resolved_provenance,
    )


def resolve_curve(
    as_of: date | str | None = None,
    source: str = "treasury_gov",
) -> YieldCurve:
    """Fetch market data and build the default discount curve."""
    return resolve_market_snapshot(as_of=as_of, source=source).discount_curve()
