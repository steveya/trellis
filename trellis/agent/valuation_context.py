"""Valuation-context normalization for semantic-contract compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from trellis.agent.sensitivity_support import normalize_requested_outputs


def _string_tuple(values) -> tuple[str, ...]:
    """Return a deduplicated tuple of normalized strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _normalize_label(value: str | None, *, fallback: str) -> str:
    """Normalize a policy/source label to a stable lowercase token."""
    text = str(value or "").strip().lower().replace(" ", "_")
    return text or fallback


def _normalize_token_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    """Return a deduplicated tuple of normalized lowercase token strings."""
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        token = _normalize_label(str(value), fallback="")
        if token and token not in result:
            result.append(token)
    return tuple(result)


def _market_source_label(snapshot) -> str:
    """Return a stable source label for one market snapshot binding."""
    if snapshot is None:
        return "unbound_market_snapshot"
    source = getattr(snapshot, "source", None) or getattr(snapshot, "data_source", None)
    if source:
        return _normalize_label(str(source), fallback="provided_market_snapshot")
    return f"provided_{type(snapshot).__name__.lower()}"


def _market_snapshot_handle(snapshot) -> str:
    """Return a lightweight handle string for traceable snapshot provenance."""
    if snapshot is None:
        return ""
    for attr in ("snapshot_id", "market_snapshot_id", "trace_id", "source"):
        value = getattr(snapshot, attr, None)
        if value:
            return str(value).strip()
    return type(snapshot).__name__


_SUPPORTED_ENGINE_MODELS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "rates": ("rates_bootstrap", "hull_white_1f"),
    "volatility": ("sabr", "heston", "local_vol"),
    "credit": ("reduced_form_credit",),
}

_SUPPORTED_SOURCE_KINDS = (
    "none",
    "coupon_stream",
    "running_cashflow",
    "recovery_leg",
    "fee_leg",
    "dividend_yield",
)

_SUPPORTED_BACKEND_HINTS = (
    "curve_bootstrap",
    "analytical",
    "lattice",
    "pde",
    "monte_carlo",
)

_SUPPORTED_CALIBRATION_REQUIREMENTS = (
    "bootstrap_curve",
    "fit_hw_strip",
    "fit_sabr_smile",
    "fit_heston_smile",
    "build_local_vol_surface",
    "fit_credit_curve",
)

_SUPPORTED_DISCOUNT_TERMS = (
    "",
    "risk_free_rate",
    "curve_discount",
    "collateral_adjusted_rate",
)

_SUPPORTED_DEFAULT_TERMS = (
    "",
    "hazard_rate",
)

_SUPPORTED_FUNDING_TERMS = (
    "",
    "funding_spread",
)

_SUPPORTED_COLLATERAL_TERMS = (
    "",
    "collateral_rate",
)

_LEGACY_ENGINE_MODEL_ALIASES = {
    "rates_bootstrap": "rates_bootstrap",
    "bootstrap_curve": "rates_bootstrap",
    "hull_white": "hull_white_1f",
    "hw": "hull_white_1f",
    "hull_white_1f": "hull_white_1f",
    "sabr": "sabr",
    "sabr_smile": "sabr",
    "heston": "heston",
    "heston_sv": "heston",
    "local_vol": "local_vol",
    "local_vol_dupire": "local_vol",
    "reduced_form_credit": "reduced_form_credit",
    "credit": "reduced_form_credit",
}


@dataclass(frozen=True)
class PotentialSpec:
    """Financial potential terms attached to one engine model."""

    discount_term: str = "risk_free_rate"
    default_term: str = ""
    funding_term: str = ""
    collateral_term: str = ""

    def __post_init__(self):
        """Normalize and validate potential-term labels."""
        discount_term = _normalize_label(self.discount_term, fallback="")
        default_term = _normalize_label(self.default_term, fallback="")
        funding_term = _normalize_label(self.funding_term, fallback="")
        collateral_term = _normalize_label(self.collateral_term, fallback="")
        object.__setattr__(self, "discount_term", discount_term)
        object.__setattr__(self, "default_term", default_term)
        object.__setattr__(self, "funding_term", funding_term)
        object.__setattr__(self, "collateral_term", collateral_term)

        errors: list[str] = []
        if discount_term not in _SUPPORTED_DISCOUNT_TERMS:
            errors.append(f"unsupported discount_term `{discount_term}`")
        if default_term not in _SUPPORTED_DEFAULT_TERMS:
            errors.append(f"unsupported default_term `{default_term}`")
        if funding_term not in _SUPPORTED_FUNDING_TERMS:
            errors.append(f"unsupported funding_term `{funding_term}`")
        if collateral_term not in _SUPPORTED_COLLATERAL_TERMS:
            errors.append(f"unsupported collateral_term `{collateral_term}`")
        if errors:
            raise ValueError("Invalid PotentialSpec: " + "; ".join(errors))


@dataclass(frozen=True)
class SourceSpec:
    """Running source-term semantics attached to one engine model."""

    source_kind: str = "none"
    description: str = ""

    def __post_init__(self):
        """Normalize and validate the source-kind label."""
        source_kind = _normalize_label(self.source_kind, fallback="none")
        object.__setattr__(self, "source_kind", source_kind)
        if source_kind not in _SUPPORTED_SOURCE_KINDS:
            raise ValueError(f"Invalid SourceSpec: unsupported source_kind `{source_kind}`")


@dataclass(frozen=True)
class RatesCurveRoleSpec:
    """Explicit multi-curve role labels for rates workflows."""

    discount_curve_role: str = ""
    forecast_curve_role: str = ""
    rate_index: str = ""

    def __post_init__(self):
        """Normalize role labels into stable lowercase tokens."""
        object.__setattr__(
            self,
            "discount_curve_role",
            _normalize_label(self.discount_curve_role, fallback=""),
        )
        object.__setattr__(
            self,
            "forecast_curve_role",
            _normalize_label(self.forecast_curve_role, fallback=""),
        )
        object.__setattr__(
            self,
            "rate_index",
            _normalize_label(self.rate_index, fallback=""),
        )


@dataclass(frozen=True)
class EngineModelSpec:
    """Bounded model-grammar authority for supported calibration workflows."""

    model_family: str
    model_name: str
    state_semantics: tuple[str, ...] = ()
    potential: PotentialSpec = field(default_factory=PotentialSpec)
    sources: tuple[SourceSpec, ...] = ()
    calibration_requirements: tuple[str, ...] = ()
    backend_hints: tuple[str, ...] = ()
    rates_curve_roles: RatesCurveRoleSpec | None = None
    description: str = ""

    def __post_init__(self):
        """Normalize labels and reject unsupported free-form combinations."""
        model_family = _normalize_label(self.model_family, fallback="")
        model_name = _normalize_label(self.model_name, fallback="")
        state_semantics = _normalize_token_tuple(self.state_semantics)
        calibration_requirements = _normalize_token_tuple(self.calibration_requirements)
        backend_hints = _normalize_token_tuple(self.backend_hints)
        normalized_sources = tuple(
            source if isinstance(source, SourceSpec) else SourceSpec(source_kind=str(source))
            for source in self.sources
        )
        object.__setattr__(self, "model_family", model_family)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "state_semantics", state_semantics)
        object.__setattr__(self, "sources", normalized_sources)
        object.__setattr__(self, "calibration_requirements", calibration_requirements)
        object.__setattr__(self, "backend_hints", backend_hints)

        errors = validate_engine_model_spec(self)
        if errors:
            raise ValueError("Invalid EngineModelSpec: " + "; ".join(errors))


def validate_engine_model_spec(spec: EngineModelSpec) -> tuple[str, ...]:
    """Return a tuple of validation errors for one engine model spec."""
    errors: list[str] = []
    family_models = _SUPPORTED_ENGINE_MODELS_BY_FAMILY.get(spec.model_family)
    if family_models is None:
        errors.append(
            f"unsupported model_family `{spec.model_family}`; expected one of {sorted(_SUPPORTED_ENGINE_MODELS_BY_FAMILY)}"
        )
        return tuple(errors)

    if spec.model_name not in family_models:
        errors.append(
            f"unsupported model_name `{spec.model_name}` for family `{spec.model_family}`; expected one of {sorted(family_models)}"
        )
    if not spec.state_semantics:
        errors.append("state_semantics must not be empty")

    if len(spec.sources) > 1 and any(source.source_kind == "none" for source in spec.sources):
        errors.append("source_kind `none` cannot be combined with other source terms")
    for requirement in spec.calibration_requirements:
        if requirement not in _SUPPORTED_CALIBRATION_REQUIREMENTS:
            errors.append(
                f"unsupported calibration requirement `{requirement}`; expected one of {sorted(_SUPPORTED_CALIBRATION_REQUIREMENTS)}"
            )
    for hint in spec.backend_hints:
        if hint not in _SUPPORTED_BACKEND_HINTS:
            errors.append(
                f"unsupported backend hint `{hint}`; expected one of {sorted(_SUPPORTED_BACKEND_HINTS)}"
            )

    has_rates_roles = spec.rates_curve_roles is not None and (
        bool(spec.rates_curve_roles.discount_curve_role)
        or bool(spec.rates_curve_roles.forecast_curve_role)
        or bool(spec.rates_curve_roles.rate_index)
    )
    if spec.model_family == "rates":
        if spec.rates_curve_roles is None:
            errors.append("rates models require explicit rates_curve_roles")
        else:
            if not spec.rates_curve_roles.discount_curve_role:
                errors.append("rates models require a non-empty discount_curve_role")
            if not spec.rates_curve_roles.forecast_curve_role:
                errors.append("rates models require a non-empty forecast_curve_role")
    elif has_rates_roles:
        errors.append("non-rates models must not declare rates_curve_roles")

    return tuple(errors)


def engine_model_spec_summary(spec: EngineModelSpec | None) -> dict[str, object] | None:
    """Return a compact YAML-safe summary for one engine model spec."""
    if spec is None:
        return None
    rates_curve_roles = None
    if spec.rates_curve_roles is not None:
        rates_curve_roles = {
            "discount_curve_role": spec.rates_curve_roles.discount_curve_role,
            "forecast_curve_role": spec.rates_curve_roles.forecast_curve_role,
            "rate_index": spec.rates_curve_roles.rate_index,
        }
    return {
        "model_family": spec.model_family,
        "model_name": spec.model_name,
        "state_semantics": list(spec.state_semantics),
        "potential": {
            "discount_term": spec.potential.discount_term,
            "default_term": spec.potential.default_term,
            "funding_term": spec.potential.funding_term,
            "collateral_term": spec.potential.collateral_term,
        },
        "sources": [
            {
                "source_kind": source.source_kind,
                "description": source.description,
            }
            for source in spec.sources
        ],
        "calibration_requirements": list(spec.calibration_requirements),
        "backend_hints": list(spec.backend_hints),
        "rates_curve_roles": rates_curve_roles,
        "description": spec.description,
    }


def _default_engine_model_spec(model_name: str) -> EngineModelSpec:
    """Return one canonical engine-model spec for supported legacy model names."""
    if model_name == "rates_bootstrap":
        return EngineModelSpec(
            model_family="rates",
            model_name="rates_bootstrap",
            state_semantics=("curve_nodes",),
            potential=PotentialSpec(discount_term="curve_discount"),
            sources=(SourceSpec(source_kind="none"),),
            calibration_requirements=("bootstrap_curve",),
            backend_hints=("curve_bootstrap",),
            rates_curve_roles=RatesCurveRoleSpec(
                discount_curve_role="discount_curve",
                forecast_curve_role="forward_curve",
            ),
        )
    if model_name == "hull_white_1f":
        return EngineModelSpec(
            model_family="rates",
            model_name="hull_white_1f",
            state_semantics=("short_rate",),
            potential=PotentialSpec(discount_term="risk_free_rate"),
            sources=(SourceSpec(source_kind="coupon_stream"),),
            calibration_requirements=("bootstrap_curve", "fit_hw_strip"),
            backend_hints=("lattice",),
            rates_curve_roles=RatesCurveRoleSpec(
                discount_curve_role="discount_curve",
                forecast_curve_role="forward_curve",
            ),
        )
    if model_name == "sabr":
        return EngineModelSpec(
            model_family="volatility",
            model_name="sabr",
            state_semantics=("forward", "implied_vol"),
            potential=PotentialSpec(discount_term="risk_free_rate"),
            sources=(SourceSpec(source_kind="none"),),
            calibration_requirements=("fit_sabr_smile",),
            backend_hints=("analytical",),
        )
    if model_name == "heston":
        return EngineModelSpec(
            model_family="volatility",
            model_name="heston",
            state_semantics=("log_spot", "variance"),
            potential=PotentialSpec(discount_term="risk_free_rate"),
            sources=(SourceSpec(source_kind="dividend_yield"),),
            calibration_requirements=("fit_heston_smile",),
            backend_hints=("monte_carlo",),
        )
    if model_name == "local_vol":
        return EngineModelSpec(
            model_family="volatility",
            model_name="local_vol",
            state_semantics=("spot", "time"),
            potential=PotentialSpec(discount_term="risk_free_rate"),
            sources=(SourceSpec(source_kind="none"),),
            calibration_requirements=("build_local_vol_surface",),
            backend_hints=("pde", "monte_carlo"),
        )
    if model_name == "reduced_form_credit":
        return EngineModelSpec(
            model_family="credit",
            model_name="reduced_form_credit",
            state_semantics=("hazard_rate",),
            potential=PotentialSpec(
                discount_term="risk_free_rate",
                default_term="hazard_rate",
            ),
            sources=(SourceSpec(source_kind="recovery_leg"),),
            calibration_requirements=("fit_credit_curve",),
            backend_hints=("analytical",),
        )
    raise ValueError(f"Unsupported canonical engine model `{model_name}`")


def _legacy_engine_model_name(model_spec: str | None) -> str:
    """Map one legacy model_spec label onto a supported canonical model name."""
    token = _normalize_label(model_spec, fallback="")
    return _LEGACY_ENGINE_MODEL_ALIASES.get(token, "")


def _engine_model_spec_from_legacy_model_spec(model_spec: str | None) -> EngineModelSpec | None:
    """Build a canonical engine-model spec from a supported legacy string model spec."""
    canonical_name = _legacy_engine_model_name(model_spec)
    if not canonical_name:
        return None
    return _default_engine_model_spec(canonical_name)


@dataclass(frozen=True)
class ReportingPolicy:
    """Tranche-1 reporting and FX conversion policy."""

    reporting_currency: str = ""
    fx_conversion_policy: str = "native_payout"
    include_native_currency: bool = True


@dataclass(frozen=True)
class ValuationContext:
    """Valuation policy and market-binding context compiled separately from semantics."""

    market_source: str = "unbound_market_snapshot"
    market_snapshot_handle: str = ""
    market_snapshot: object | None = None
    model_spec: str | None = None
    engine_model_spec: EngineModelSpec | None = None
    measure_spec: str = "risk_neutral"
    discounting_policy: str = "contract_convention_discounting"
    collateral_policy: str | None = None
    reporting_policy: ReportingPolicy = field(default_factory=ReportingPolicy)
    requested_outputs: tuple[str, ...] = ()

    def __post_init__(self):
        """Normalize policy labels and requested outputs into stable tuples."""
        model_spec = _normalize_label(self.model_spec, fallback="") if self.model_spec else ""
        engine_model_spec = self.engine_model_spec
        if engine_model_spec is None and model_spec:
            engine_model_spec = _engine_model_spec_from_legacy_model_spec(model_spec)
        if engine_model_spec is not None and not model_spec:
            model_spec = engine_model_spec.model_name
        if engine_model_spec is not None and model_spec:
            canonical_name = _legacy_engine_model_name(model_spec)
            if canonical_name and canonical_name != engine_model_spec.model_name:
                raise ValueError(
                    "Conflicting model specifications: "
                    f"legacy model_spec `{model_spec}` maps to `{canonical_name}` "
                    f"but engine_model_spec.model_name is `{engine_model_spec.model_name}`."
                )

        object.__setattr__(self, "model_spec", model_spec or None)
        object.__setattr__(self, "engine_model_spec", engine_model_spec)
        object.__setattr__(
            self,
            "market_source",
            _normalize_label(
                self.market_source,
                fallback=_market_source_label(self.market_snapshot),
            ),
        )
        object.__setattr__(
            self,
            "market_snapshot_handle",
            str(self.market_snapshot_handle or _market_snapshot_handle(self.market_snapshot)).strip(),
        )
        object.__setattr__(
            self,
            "measure_spec",
            _normalize_label(self.measure_spec, fallback="risk_neutral"),
        )
        object.__setattr__(
            self,
            "discounting_policy",
            _normalize_label(self.discounting_policy, fallback="contract_convention_discounting"),
        )
        object.__setattr__(
            self,
            "requested_outputs",
            normalize_requested_outputs(self.requested_outputs),
        )


def build_valuation_context(
    *,
    market_snapshot=None,
    model_spec: str | None = None,
    engine_model_spec: EngineModelSpec | None = None,
    measure_spec: str | None = None,
    discounting_policy: str | None = None,
    collateral_policy: str | None = None,
    reporting_currency: str | None = None,
    requested_outputs=None,
) -> ValuationContext:
    """Construct a normalized valuation context from front-door request data."""
    return ValuationContext(
        market_source=_market_source_label(market_snapshot),
        market_snapshot_handle=_market_snapshot_handle(market_snapshot),
        market_snapshot=market_snapshot,
        model_spec=model_spec,
        engine_model_spec=engine_model_spec,
        measure_spec=measure_spec or "risk_neutral",
        discounting_policy=discounting_policy or "contract_convention_discounting",
        collateral_policy=collateral_policy,
        reporting_policy=ReportingPolicy(
            reporting_currency=str(reporting_currency or "").strip(),
        ),
        requested_outputs=_string_tuple(requested_outputs),
    )


def normalize_valuation_context(
    valuation_context: ValuationContext | None = None,
    *,
    market_snapshot=None,
    model_spec: str | None = None,
    engine_model_spec: EngineModelSpec | None = None,
    measure_spec: str | None = None,
    discounting_policy: str | None = None,
    collateral_policy: str | None = None,
    reporting_currency: str | None = None,
    requested_outputs=None,
    requested_measures=None,
) -> ValuationContext:
    """Return one normalized valuation context, merging legacy request inputs."""
    normalized_outputs = normalize_requested_outputs(
        requested_outputs if requested_outputs is not None else requested_measures
    )
    if valuation_context is None:
        return build_valuation_context(
            market_snapshot=market_snapshot,
            model_spec=model_spec,
            engine_model_spec=engine_model_spec,
            measure_spec=measure_spec,
            discounting_policy=discounting_policy,
            collateral_policy=collateral_policy,
            reporting_currency=reporting_currency,
            requested_outputs=normalized_outputs,
        )

    merged_outputs = valuation_context.requested_outputs or ()
    for output in normalized_outputs:
        if output not in merged_outputs:
            merged_outputs += (output,)
    return ValuationContext(
        market_source=valuation_context.market_source,
        market_snapshot_handle=valuation_context.market_snapshot_handle,
        market_snapshot=valuation_context.market_snapshot or market_snapshot,
        model_spec=valuation_context.model_spec or model_spec,
        engine_model_spec=valuation_context.engine_model_spec or engine_model_spec,
        measure_spec=valuation_context.measure_spec or measure_spec or "risk_neutral",
        discounting_policy=valuation_context.discounting_policy or discounting_policy or "contract_convention_discounting",
        collateral_policy=valuation_context.collateral_policy or collateral_policy,
        reporting_policy=ReportingPolicy(
            reporting_currency=valuation_context.reporting_policy.reporting_currency or str(reporting_currency or "").strip(),
            fx_conversion_policy=valuation_context.reporting_policy.fx_conversion_policy,
            include_native_currency=valuation_context.reporting_policy.include_native_currency,
        ),
        requested_outputs=merged_outputs,
    )


def valuation_context_summary(context: ValuationContext) -> dict[str, object]:
    """Return a compact YAML-safe summary of one valuation context."""
    return {
        "market_source": context.market_source,
        "market_snapshot_handle": context.market_snapshot_handle,
        "model_spec": context.model_spec,
        "engine_model_spec": engine_model_spec_summary(context.engine_model_spec),
        "measure_spec": context.measure_spec,
        "discounting_policy": context.discounting_policy,
        "collateral_policy": context.collateral_policy,
        "reporting_policy": {
            "reporting_currency": context.reporting_policy.reporting_currency,
            "fx_conversion_policy": context.reporting_policy.fx_conversion_policy,
            "include_native_currency": context.reporting_policy.include_native_currency,
        },
        "requested_outputs": list(context.requested_outputs),
    }
