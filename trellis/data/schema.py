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


def _freeze_parameter_mapping(mapping: Mapping[str, Mapping[str, object]] | None) -> Mapping:
    """Return an immutable shallow copy of nested parameter mappings."""
    frozen = {
        key: MappingProxyType(dict(value))
        for key, value in dict(mapping or {}).items()
    }
    return MappingProxyType(frozen)


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
    default_state_space: str | None = None
    default_underlier_spot: str | None = None
    default_local_vol_surface: str | None = None
    default_jump_parameters: str | None = None
    default_model_parameters: str | None = None

    def __post_init__(self):
        """Freeze all mapping fields so snapshots remain immutable and shareable."""
        object.__setattr__(self, "discount_curves", _freeze_mapping(self.discount_curves))
        object.__setattr__(self, "forecast_curves", _freeze_mapping(self.forecast_curves))
        object.__setattr__(self, "vol_surfaces", _freeze_mapping(self.vol_surfaces))
        object.__setattr__(self, "credit_curves", _freeze_mapping(self.credit_curves))
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

    def to_market_state(
        self,
        *,
        settlement: date,
        discount_curve: str | None = None,
        forecast_curve: str | None = None,
        vol_surface: str | None = None,
        credit_curve: str | None = None,
        fx_rate: str | None = None,
        state_space: str | None = None,
        underlier_spot: str | None = None,
        local_vol_surface: str | None = None,
        jump_parameters: str | None = None,
        model_parameters: str | None = None,
    ) -> MarketState:
        """Compile the snapshot into a runtime MarketState."""
        from trellis.curves.forward_curve import ForwardCurve

        selected_forecast_curves = dict(self.forecast_curves) or None
        selected_forward_curve = None
        if forecast_curve is not None:
            forecast_discount = self._select_mapping_entry(
                self.forecast_curves,
                explicit_name=forecast_curve,
                default_name=None,
                kind="forecast curve",
            )
            selected_forecast_curves = {forecast_curve: forecast_discount}
            selected_forward_curve = ForwardCurve(forecast_discount)

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
            discount=self.discount_curve(discount_curve),
            forward_curve=selected_forward_curve,
            vol_surface=self.vol_surface(vol_surface),
            credit_curve=self.credit_curve(credit_curve),
            forecast_curves=selected_forecast_curves,
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
        if explicit_name is not None:
            if explicit_name not in mapping:
                raise ValueError(f"Unknown {kind}: {explicit_name}")
            return mapping[explicit_name]
        if not mapping:
            return None
        if default_name is not None:
            if default_name not in mapping:
                raise ValueError(f"Unknown default {kind}: {default_name}")
            return mapping[default_name]
        if len(mapping) == 1:
            return next(iter(mapping.values()))
        raise ValueError(f"Multiple {kind}s available; set a default {kind} name")
