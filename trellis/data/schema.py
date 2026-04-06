"""Canonical market snapshot schema."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from trellis.core.market_state import MarketState

if TYPE_CHECKING:
    from trellis.curves.credit_curve import CreditCurve
    from trellis.core.types import DiscountCurve
    from trellis.instruments.fx import FXRate
    from trellis.models.vol_surface import VolSurface


def _freeze_mapping(mapping: Mapping | None) -> Mapping:
    """Return an immutable copy of a mapping."""
    return MappingProxyType(dict(mapping or {}))


def _freeze_string_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, str]:
    """Return an immutable string-to-string mapping."""
    return MappingProxyType(
        {
            str(key).strip(): str(value).strip()
            for key, value in dict(mapping or {}).items()
            if str(key).strip() and str(value).strip()
        }
    )


def _freeze_name_lists(mapping: Mapping[str, object] | None) -> Mapping[str, tuple[str, ...]]:
    """Return an immutable mapping of component families to name tuples."""
    frozen: dict[str, tuple[str, ...]] = {}
    for key, value in dict(mapping or {}).items():
        family = str(key).strip()
        if not family:
            continue
        if isinstance(value, str):
            names = (value.strip(),) if value.strip() else ()
        else:
            names = tuple(
                str(item).strip()
                for item in value
                if str(item).strip()
            )
        frozen[family] = names
    return MappingProxyType(frozen)


def _freeze_template_mapping(mapping: Mapping[str, object] | None) -> Mapping[str, Mapping[str, object]]:
    """Return an immutable shallow copy of named scenario-template specs."""
    return MappingProxyType(
        {
            str(key).strip(): MappingProxyType(dict(value))
            for key, value in dict(mapping or {}).items()
            if str(key).strip() and isinstance(value, Mapping)
        }
    )


def _string_tuple(values) -> tuple[str, ...]:
    """Return a stable ordered tuple of unique strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalize_date_value(value, *, field: str) -> date:
    """Normalize a ``date`` or ISO-8601 string into ``date``."""
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required.")
    return date.fromisoformat(text)


def _freeze_parameter_mapping(mapping: Mapping[str, Mapping[str, object]] | None) -> Mapping:
    """Return an immutable shallow copy of nested parameter mappings."""
    frozen = {
        key: MappingProxyType(dict(value))
        for key, value in dict(mapping or {}).items()
    }
    return MappingProxyType(frozen)


def _normalize_fixing_history_mapping(value) -> Mapping[date, float]:
    """Return one immutable fixing-history mapping keyed by ``date``."""
    normalized: dict[date, float] = {}
    if value is None:
        return MappingProxyType(normalized)
    if isinstance(value, Mapping):
        items = value.items()
    else:
        items = (
            (item.get("date"), item.get("value"))
            for item in value
        )
    for raw_date, raw_value in items:
        fixing_date = raw_date if isinstance(raw_date, date) else date.fromisoformat(str(raw_date).strip())
        normalized[fixing_date] = float(raw_value)
    return MappingProxyType(normalized)


def _freeze_fixing_history_mapping(mapping: Mapping[str, object] | None) -> Mapping:
    """Return an immutable shallow copy of named fixing histories."""
    frozen = {
        key: _normalize_fixing_history_mapping(value)
        for key, value in dict(mapping or {}).items()
    }
    return MappingProxyType(frozen)


SNAPSHOT_SELECTION_FIELDS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "discount_curve": ("discount_curves", "default_discount_curve", "discount curve"),
        "forecast_curve": ("forecast_curves", "", "forecast curve"),
        "vol_surface": ("vol_surfaces", "default_vol_surface", "vol surface"),
        "credit_curve": ("credit_curves", "default_credit_curve", "credit curve"),
        "fixing_history": ("fixing_histories", "default_fixing_history", "fixing history"),
        "fx_rate": ("fx_rates", "", "fx rate"),
        "state_space": ("state_spaces", "default_state_space", "state space"),
        "underlier_spot": ("underlier_spots", "default_underlier_spot", "underlier spot"),
        "local_vol_surface": ("local_vol_surfaces", "default_local_vol_surface", "local vol surface"),
        "jump_parameters": ("jump_parameter_sets", "default_jump_parameters", "jump parameter set"),
        "model_parameters": ("model_parameter_sets", "default_model_parameters", "model parameter set"),
    }
)


@dataclass(frozen=True)
class SnapshotSelectionResult:
    """Stable result for request-driven snapshot component selection."""

    selection_status: str
    selected_components: Mapping[str, str] = field(default_factory=dict)
    selected_curve_names: Mapping[str, str] = field(default_factory=dict)
    available_components: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    scenario_templates: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    missing_components: Mapping[str, str] = field(default_factory=dict)
    missing_scenario_templates: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    market_state_object: MarketState | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "selection_status", str(self.selection_status or "").strip())
        object.__setattr__(self, "selected_components", _freeze_string_mapping(self.selected_components))
        object.__setattr__(self, "selected_curve_names", _freeze_string_mapping(self.selected_curve_names))
        object.__setattr__(self, "available_components", _freeze_name_lists(self.available_components))
        object.__setattr__(self, "scenario_templates", _freeze_template_mapping(self.scenario_templates))
        object.__setattr__(self, "missing_components", _freeze_string_mapping(self.missing_components))
        object.__setattr__(
            self,
            "missing_scenario_templates",
            _string_tuple(self.missing_scenario_templates),
        )
        object.__setattr__(self, "warnings", _string_tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_status": self.selection_status,
            "selected_components": dict(self.selected_components),
            "selected_curve_names": dict(self.selected_curve_names),
            "available_components": {
                key: list(value)
                for key, value in self.available_components.items()
            },
            "scenario_templates": {
                key: dict(value)
                for key, value in self.scenario_templates.items()
            },
            "missing_components": dict(self.missing_components),
            "missing_scenario_templates": list(self.missing_scenario_templates),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MarketSnapshot:
    """Canonical market snapshot before runtime compilation to MarketState.

    This is the market-data analogue of ``ProductIR``:

    - it holds structured market objects and provenance
    - it is richer than the runtime ``MarketState``
    - it can compile down to a ``MarketState`` for pricing
    """

    as_of: date
    source: str
    discount_curves: Mapping[str, DiscountCurve] = field(default_factory=dict)
    forecast_curves: Mapping[str, DiscountCurve] = field(default_factory=dict)
    vol_surfaces: Mapping[str, VolSurface] = field(default_factory=dict)
    credit_curves: Mapping[str, CreditCurve] = field(default_factory=dict)
    fixing_histories: Mapping[str, Mapping[date, float]] = field(default_factory=dict)
    fx_rates: Mapping[str, FXRate] = field(default_factory=dict)
    state_spaces: Mapping[str, object] = field(default_factory=dict)
    underlier_spots: Mapping[str, float] = field(default_factory=dict)
    local_vol_surfaces: Mapping[str, object] = field(default_factory=dict)
    jump_parameter_sets: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    model_parameter_sets: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    default_discount_curve: str | None = None
    default_vol_surface: str | None = None
    default_credit_curve: str | None = None
    default_fixing_history: str | None = None
    default_state_space: str | None = None
    default_underlier_spot: str | None = None
    default_local_vol_surface: str | None = None
    default_jump_parameters: str | None = None
    default_model_parameters: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Freeze all mapping fields so snapshots remain immutable and shareable."""
        object.__setattr__(self, "discount_curves", _freeze_mapping(self.discount_curves))
        object.__setattr__(self, "forecast_curves", _freeze_mapping(self.forecast_curves))
        object.__setattr__(self, "vol_surfaces", _freeze_mapping(self.vol_surfaces))
        object.__setattr__(self, "credit_curves", _freeze_mapping(self.credit_curves))
        object.__setattr__(self, "fixing_histories", _freeze_fixing_history_mapping(self.fixing_histories))
        object.__setattr__(self, "fx_rates", _freeze_mapping(self.fx_rates))
        object.__setattr__(self, "state_spaces", _freeze_mapping(self.state_spaces))
        object.__setattr__(self, "underlier_spots", _freeze_mapping(self.underlier_spots))
        object.__setattr__(self, "local_vol_surfaces", _freeze_mapping(self.local_vol_surfaces))
        object.__setattr__(
            self,
            "jump_parameter_sets",
            _freeze_parameter_mapping(self.jump_parameter_sets),
        )
        object.__setattr__(
            self,
            "model_parameter_sets",
            _freeze_parameter_mapping(self.model_parameter_sets),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    def discount_curve(self, name: str | None = None):
        """Return a selected discount curve, if available."""
        return self._select_mapping_entry(
            self.discount_curves,
            explicit_name=name,
            default_name=self.default_discount_curve,
            kind="discount curve",
        )

    def vol_surface(self, name: str | None = None):
        """Return a selected volatility surface, if available."""
        return self._select_mapping_entry(
            self.vol_surfaces,
            explicit_name=name,
            default_name=self.default_vol_surface,
            kind="vol surface",
        )

    def credit_curve(self, name: str | None = None):
        """Return a selected credit curve, if available."""
        return self._select_mapping_entry(
            self.credit_curves,
            explicit_name=name,
            default_name=self.default_credit_curve,
            kind="credit curve",
        )

    def fixing_history(self, name: str | None = None):
        """Return a selected fixing history, if available."""
        return self._select_mapping_entry(
            self.fixing_histories,
            explicit_name=name,
            default_name=self.default_fixing_history,
            kind="fixing history",
        )

    @property
    def provider_id(self) -> str:
        """Return the stable provider id recorded in provenance, if present."""
        return str(self.provenance.get("provider_id", "")).strip()

    @property
    def snapshot_id(self) -> str:
        """Return the canonical snapshot id recorded in provenance, if present."""
        return str(self.provenance.get("snapshot_id", "")).strip()

    @property
    def market_snapshot_id(self) -> str:
        """Backward-compatible alias for canonical snapshot identity."""
        return self.snapshot_id

    def state_space(self, name: str | None = None):
        """Return a selected state-space object or factory, if available."""
        return self._select_mapping_entry(
            self.state_spaces,
            explicit_name=name,
            default_name=self.default_state_space,
            kind="state space",
        )

    def underlier_spot(self, name: str | None = None):
        """Return a selected underlier spot, if available."""
        return self._select_mapping_entry(
            self.underlier_spots,
            explicit_name=name,
            default_name=self.default_underlier_spot,
            kind="underlier spot",
        )

    def local_vol_surface(self, name: str | None = None):
        """Return a selected local-vol function, if available."""
        return self._select_mapping_entry(
            self.local_vol_surfaces,
            explicit_name=name,
            default_name=self.default_local_vol_surface,
            kind="local vol surface",
        )

    def jump_parameters(self, name: str | None = None):
        """Return a selected jump-parameter pack, if available."""
        return self._select_mapping_entry(
            self.jump_parameter_sets,
            explicit_name=name,
            default_name=self.default_jump_parameters,
            kind="jump parameter set",
        )

    def model_parameters(self, name: str | None = None):
        """Return a selected generic model-parameter pack, if available."""
        return self._select_mapping_entry(
            self.model_parameter_sets,
            explicit_name=name,
            default_name=self.default_model_parameters,
            kind="model parameter set",
        )

    def available_component_names(self) -> dict[str, list[str]]:
        """Return all named snapshot components grouped by request key."""
        available: dict[str, list[str]] = {}
        for request_key, (mapping_attr, _, _) in SNAPSHOT_SELECTION_FIELDS.items():
            mapping = getattr(self, mapping_attr)
            available[request_key] = list(mapping)
        return available

    def resolve_request(
        self,
        *,
        settlement: date | str,
        selected_components: Mapping[str, object] | None = None,
        scenario_templates=(),
        reference_date: date | str | None = None,
    ) -> SnapshotSelectionResult:
        """Resolve one named-component request onto a runtime market state."""
        normalized_settlement = _normalize_date_value(settlement, field="settlement")
        normalized_reference_date = (
            None
            if reference_date is None
            else _normalize_date_value(reference_date, field="reference_date")
        )
        requested_components = {
            str(key).strip(): str(value).strip()
            for key, value in dict(selected_components or {}).items()
            if str(key).strip() and str(value).strip()
        }
        available = self.available_component_names()
        missing_components: dict[str, str] = {}
        warnings: list[str] = []

        if normalized_reference_date is not None and self.as_of < normalized_reference_date:
            warnings.append(
                f"Market snapshot as_of {self.as_of.isoformat()} is stale for reference date {normalized_reference_date.isoformat()}."
            )

        for key, name in requested_components.items():
            if key not in SNAPSHOT_SELECTION_FIELDS:
                missing_components[key] = name
                warnings.append(f"Unsupported snapshot selection key: {key}.")
                continue
            mapping_attr, _, kind = SNAPSHOT_SELECTION_FIELDS[key]
            mapping = getattr(self, mapping_attr)
            if name not in mapping:
                missing_components[key] = name
                warnings.append(
                    f"Requested snapshot component {key}={name!r} ({kind}) is not present in this market snapshot."
                )

        scenario_catalog = dict(self.metadata.get("scenario_templates") or {})
        selected_scenario_templates: dict[str, Mapping[str, object]] = {}
        missing_scenario_templates: list[str] = []
        for template_name in _string_tuple(scenario_templates):
            if template_name not in scenario_catalog:
                missing_scenario_templates.append(template_name)
                warnings.append(
                    f"Requested scenario template {template_name!r} is not present in this market snapshot."
                )
                continue
            template_spec = scenario_catalog[template_name]
            if not isinstance(template_spec, Mapping):
                missing_scenario_templates.append(template_name)
                warnings.append(
                    f"Requested scenario template {template_name!r} is not a mapping payload."
                )
                continue
            selected_scenario_templates[template_name] = dict(template_spec)

        if missing_components or missing_scenario_templates:
            return SnapshotSelectionResult(
                selection_status="invalid",
                selected_components=requested_components,
                available_components=available,
                scenario_templates=selected_scenario_templates,
                missing_components=missing_components,
                missing_scenario_templates=tuple(missing_scenario_templates),
                warnings=warnings,
            )

        try:
            market_state = self.to_market_state(
                settlement=normalized_settlement,
                discount_curve=requested_components.get("discount_curve"),
                forecast_curve=requested_components.get("forecast_curve"),
                vol_surface=requested_components.get("vol_surface"),
                credit_curve=requested_components.get("credit_curve"),
                fixing_history=requested_components.get("fixing_history"),
                fx_rate=requested_components.get("fx_rate"),
                state_space=requested_components.get("state_space"),
                underlier_spot=requested_components.get("underlier_spot"),
                local_vol_surface=requested_components.get("local_vol_surface"),
                jump_parameters=requested_components.get("jump_parameters"),
                model_parameters=requested_components.get("model_parameters"),
            )
        except ValueError as exc:
            warnings.append(str(exc))
            return SnapshotSelectionResult(
                selection_status="invalid",
                selected_components=requested_components,
                available_components=available,
                scenario_templates=selected_scenario_templates,
                warnings=warnings,
            )

        return SnapshotSelectionResult(
            selection_status="parsed",
            selected_components=requested_components,
            selected_curve_names=dict(market_state.selected_curve_names or {}),
            available_components=available,
            scenario_templates=selected_scenario_templates,
            warnings=warnings,
            market_state_object=market_state,
        )

    def to_market_state(
        self,
        *,
        settlement: date,
        discount_curve: str | None = None,
        forecast_curve: str | None = None,
        vol_surface: str | None = None,
        credit_curve: str | None = None,
        fixing_history: str | None = None,
        fx_rate: str | None = None,
        state_space: str | None = None,
        underlier_spot: str | None = None,
        local_vol_surface: str | None = None,
        jump_parameters: str | None = None,
        model_parameters: str | None = None,
    ) -> MarketState:
        """Compile the snapshot into a runtime MarketState."""
        from trellis.curves.forward_curve import ForwardCurve

        selected_curve_names: dict[str, str] = {}

        selected_discount_curve_name, selected_discount_curve = self._select_named_mapping_entry(
            self.discount_curves,
            explicit_name=discount_curve,
            default_name=self.default_discount_curve,
            kind="discount curve",
        )
        if selected_discount_curve_name is not None:
            selected_curve_names["discount_curve"] = selected_discount_curve_name

        selected_forecast_curves = dict(self.forecast_curves) or None
        selected_forward_curve = None
        if forecast_curve is not None:
            selected_forecast_curve_name, forecast_discount = self._select_named_mapping_entry(
                self.forecast_curves,
                explicit_name=forecast_curve,
                default_name=None,
                kind="forecast curve",
            )
            if selected_forecast_curve_name is not None:
                selected_curve_names["forecast_curve"] = selected_forecast_curve_name
                selected_forecast_curves = {selected_forecast_curve_name: forecast_discount}
            selected_forward_curve = ForwardCurve(forecast_discount)
        elif len(self.forecast_curves) == 1:
            selected_curve_names["forecast_curve"] = next(iter(self.forecast_curves))

        selected_credit_curve_name, selected_credit_curve = self._select_named_mapping_entry(
            self.credit_curves,
            explicit_name=credit_curve,
            default_name=self.default_credit_curve,
            kind="credit curve",
        )
        if selected_credit_curve_name is not None:
            selected_curve_names["credit_curve"] = selected_credit_curve_name

        selected_fixing_histories = (
            {key: dict(value) for key, value in self.fixing_histories.items()}
            or None
        )
        selected_fixing_history_name = None
        if fixing_history is not None:
            selected_fixing_history_name, selected_fixing_history = self._select_named_mapping_entry(
                self.fixing_histories,
                explicit_name=fixing_history,
                default_name=None,
                kind="fixing history",
            )
            if selected_fixing_history_name is not None:
                selected_fixing_histories = {selected_fixing_history_name: dict(selected_fixing_history)}
        elif self.default_fixing_history is not None:
            if self.default_fixing_history not in self.fixing_histories:
                raise ValueError(f"Unknown default fixing history: {self.default_fixing_history}")
            selected_fixing_history_name = self.default_fixing_history
        elif len(self.fixing_histories) == 1:
            selected_fixing_history_name = next(iter(self.fixing_histories))
        if selected_fixing_history_name is not None:
            selected_curve_names["fixing_history"] = selected_fixing_history_name

        selected_fx_rates = dict(self.fx_rates) or None
        selected_spot = self.underlier_spot(underlier_spot)
        selected_underlier_spots = dict(self.underlier_spots) or None
        if fx_rate is not None:
            fx_quote = self._select_mapping_entry(
                self.fx_rates,
                explicit_name=fx_rate,
                default_name=None,
                kind="fx rate",
            )
            selected_fx_rates = {fx_rate: fx_quote}
            selected_underlier_spots = dict(selected_underlier_spots or {})
            selected_underlier_spots[fx_rate] = fx_quote.spot
            selected_spot = fx_quote.spot

        base_state = MarketState(
            as_of=self.as_of,
            settlement=settlement,
            discount=selected_discount_curve,
            forward_curve=selected_forward_curve,
            vol_surface=self.vol_surface(vol_surface),
            credit_curve=selected_credit_curve,
            fixing_histories=selected_fixing_histories,
            forecast_curves=selected_forecast_curves,
            selected_curve_names=selected_curve_names or None,
            fx_rates=selected_fx_rates,
            spot=selected_spot,
            underlier_spots=selected_underlier_spots,
            local_vol_surface=self.local_vol_surface(local_vol_surface),
            local_vol_surfaces=dict(self.local_vol_surfaces) or None,
            jump_parameters=(
                dict(self.jump_parameters(jump_parameters))
                if self.jump_parameters(jump_parameters) is not None else None
            ),
            jump_parameter_sets=(
                {key: dict(value) for key, value in self.jump_parameter_sets.items()}
                or None
            ),
            model_parameters=(
                dict(self.model_parameters(model_parameters))
                if self.model_parameters(model_parameters) is not None else None
            ),
            model_parameter_sets=(
                {key: dict(value) for key, value in self.model_parameter_sets.items()}
                or None
            ),
            market_provenance=dict(self.provenance) or None,
        )
        selected_state_space = self.state_space(state_space)
        if callable(selected_state_space):
            selected_state_space = selected_state_space(base_state, self, settlement)
        if selected_state_space is not None:
            return replace(base_state, state_space=selected_state_space)
        return base_state

    @staticmethod
    def _select_mapping_entry(
        mapping: Mapping[str, object],
        *,
        explicit_name: str | None,
        default_name: str | None,
        kind: str,
    ):
        """Resolve one named/default mapping entry, raising on ambiguous selections."""
        _, value = MarketSnapshot._select_named_mapping_entry(
            mapping,
            explicit_name=explicit_name,
            default_name=default_name,
            kind=kind,
        )
        return value

    @staticmethod
    def _select_named_mapping_entry(
        mapping: Mapping[str, object],
        *,
        explicit_name: str | None,
        default_name: str | None,
        kind: str,
    ) -> tuple[str | None, object | None]:
        """Resolve one named/default mapping entry and return its selected name."""
        if explicit_name is not None:
            if explicit_name not in mapping:
                raise ValueError(f"Unknown {kind}: {explicit_name}")
            return explicit_name, mapping[explicit_name]
        if not mapping:
            return None, None
        if default_name is not None:
            if default_name not in mapping:
                raise ValueError(f"Unknown default {kind}: {default_name}")
            return default_name, mapping[default_name]
        if len(mapping) == 1:
            name, value = next(iter(mapping.items()))
            return name, value
        raise ValueError(f"Multiple {kind}s available; set a default {kind} name")
