"""Typed single-name reduced-form credit calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as raw_np

from trellis.core.market_state import MarketState
from trellis.curves.credit_curve import CreditCurve
from trellis.models.calibration.materialization import materialize_credit_curve
from trellis.models.calibration.quote_maps import (
    CalibrationQuoteMap,
    QuoteAxisSpec,
    QuoteMapSpec,
    QuoteSemanticsSpec,
    QuoteSettlementSpec,
    QuoteUnitSpec,
    build_identity_quote_map,
)
from trellis.models.calibration.solve_request import (
    ObjectiveBundle,
    SolveBounds,
    SolveProvenance,
    SolveReplayArtifact,
    SolveRequest,
    SolveResult,
    WarmStart,
    build_solve_provenance,
    build_solve_replay_artifact,
    execute_solve_request,
)
from trellis.models.credit_default_swap import normalize_cds_running_spread


def _require_finite_positive(value: float, *, field_name: str) -> float:
    """Return one finite positive float value or raise ``ValueError``."""
    normalized = float(value)
    if not raw_np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return normalized


def _credit_quote_assumptions(
    market_state: MarketState,
    *,
    recovery: float,
) -> tuple[str, ...]:
    """Return normalized assumptions for the single-name credit quote maps."""
    selected = dict(market_state.selected_curve_names or {})
    return (
        (
            "Reduced-form single-name calibration binds potential terms as "
            "risky_discount(t)=discount(t)*survival_probability(t)."
        ),
        f"Recovery assumption: {float(recovery):.6f}.",
        f"Discount curve role: {selected.get('discount_curve') or '<unbound>'}.",
    )


def _credit_potential_binding(
    market_state: MarketState,
    *,
    curve_name: str,
    recovery: float,
) -> dict[str, object]:
    """Return explicit discount/default potential binding metadata."""
    selected = dict(market_state.selected_curve_names or {})
    return {
        "discount_curve_role": "discount_curve",
        "discount_curve_name": selected.get("discount_curve"),
        "default_curve_role": "credit_curve",
        "default_curve_name": curve_name,
        "recovery": float(recovery),
        "risky_discount_formula": "discount(t) * survival_probability(t)",
    }


def _spread_quote_map(
    *,
    recovery: float,
    source_ref: str,
    assumptions: tuple[str, ...],
    potential_binding: dict[str, object],
) -> CalibrationQuoteMap:
    """Return the explicit spread-to-hazard quote map."""
    scale = float(1.0 - recovery)
    return CalibrationQuoteMap(
        spec=QuoteMapSpec(
            quote_family="spread",
            semantics=QuoteSemanticsSpec(
                quote_family="spread",
                quote_subject="single_name_cds",
                axes=(QuoteAxisSpec("maturity", axis_kind="tenor", unit="years"),),
                unit=QuoteUnitSpec(
                    unit_name="decimal_running_spread",
                    value_domain="credit_spread",
                    scaling="absolute",
                ),
                settlement=QuoteSettlementSpec(
                    numeraire="discount_curve",
                    discount_curve_role="discount_curve",
                ),
            ),
        ),
        quote_to_price_fn=lambda quote: normalize_cds_running_spread(float(quote)) / scale,
        price_to_quote_fn=lambda hazard: float(hazard) * scale,
        source_ref=source_ref,
        assumptions=assumptions,
        metadata={
            "quote_kind": "spread",
            "quote_unit": "decimal_running_spread",
            "hazard_formula": "spread / (1 - recovery)",
            "potential_binding": dict(potential_binding),
        },
    )


def _hazard_quote_map(
    *,
    assumptions: tuple[str, ...],
    potential_binding: dict[str, object],
) -> CalibrationQuoteMap:
    """Return the explicit hazard quote map."""
    return build_identity_quote_map(
        QuoteMapSpec(
            quote_family="hazard",
            semantics=QuoteSemanticsSpec(
                quote_family="hazard",
                quote_subject="single_name_cds",
                axes=(QuoteAxisSpec("maturity", axis_kind="tenor", unit="years"),),
                unit=QuoteUnitSpec(
                    unit_name="hazard_rate",
                    value_domain="default_intensity",
                    scaling="absolute",
                ),
                settlement=QuoteSettlementSpec(
                    numeraire="discount_curve",
                    discount_curve_role="discount_curve",
                ),
            ),
        ),
        source_ref="_hazard_quote_map",
        assumptions=assumptions,
        metadata={
            "quote_kind": "hazard",
            "quote_unit": "hazard_rate",
            "potential_binding": dict(potential_binding),
        },
    )


@dataclass(frozen=True)
class CreditHazardCalibrationQuote:
    """One supported single-name credit calibration quote."""

    maturity_years: float
    quote: float
    quote_kind: Literal["spread", "hazard"] = "spread"
    label: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        maturity_years = _require_finite_positive(self.maturity_years, field_name="maturity_years")
        quote = _require_finite_positive(self.quote, field_name="quote")
        weight = _require_finite_positive(self.weight, field_name="weight")
        quote_kind = str(self.quote_kind).strip().lower()
        if quote_kind not in {"spread", "hazard"}:
            raise ValueError("quote_kind must be 'spread' or 'hazard'")
        object.__setattr__(self, "maturity_years", maturity_years)
        object.__setattr__(self, "quote", quote)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "quote_kind", quote_kind)
        object.__setattr__(self, "label", str(self.label))

    def resolved_label(self, index: int) -> str:
        """Return a stable per-quote label."""
        if self.label.strip():
            return self.label.strip()
        tenor_label = str(float(self.maturity_years)).replace(".", "_")
        return f"{self.quote_kind}_{tenor_label}y_{index}"

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly quote payload."""
        return {
            "maturity_years": float(self.maturity_years),
            "quote": float(self.quote),
            "quote_kind": self.quote_kind,
            "label": self.label,
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class CreditHazardCalibrationResult:
    """Structured result for the supported single-name credit calibration workflow."""

    quotes: tuple[CreditHazardCalibrationQuote, ...]
    credit_curve: CreditCurve
    solve_request: SolveRequest
    solve_result: SolveResult
    solver_provenance: SolveProvenance
    solver_replay_artifact: SolveReplayArtifact
    tenors: tuple[float, ...]
    target_hazards: tuple[float, ...]
    model_hazards: tuple[float, ...]
    target_quotes: tuple[float, ...]
    model_quotes: tuple[float, ...]
    hazard_residuals: tuple[float, ...]
    quote_residuals: tuple[float, ...]
    max_abs_hazard_residual: float
    max_abs_quote_residual: float
    curve_name: str = "single_name_credit"
    recovery: float = 0.4
    potential_binding: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quotes", tuple(self.quotes))
        object.__setattr__(self, "tenors", tuple(float(value) for value in self.tenors))
        object.__setattr__(self, "target_hazards", tuple(float(value) for value in self.target_hazards))
        object.__setattr__(self, "model_hazards", tuple(float(value) for value in self.model_hazards))
        object.__setattr__(self, "target_quotes", tuple(float(value) for value in self.target_quotes))
        object.__setattr__(self, "model_quotes", tuple(float(value) for value in self.model_quotes))
        object.__setattr__(self, "hazard_residuals", tuple(float(value) for value in self.hazard_residuals))
        object.__setattr__(self, "quote_residuals", tuple(float(value) for value in self.quote_residuals))
        object.__setattr__(self, "max_abs_hazard_residual", float(self.max_abs_hazard_residual))
        object.__setattr__(self, "max_abs_quote_residual", float(self.max_abs_quote_residual))
        object.__setattr__(self, "recovery", float(self.recovery))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "assumptions", tuple(str(assumption) for assumption in self.assumptions))

    def apply_to_market_state(self, market_state: MarketState) -> MarketState:
        """Return ``market_state`` enriched with the calibrated single-name credit curve."""
        selected_curve_roles = {
            "discount_curve": str(dict(market_state.selected_curve_names or {}).get("discount_curve") or ""),
            "credit_curve": str(self.curve_name),
        }
        return materialize_credit_curve(
            market_state,
            curve_name=self.curve_name,
            credit_curve=self.credit_curve,
            source_kind=str(self.provenance.get("source_kind", "calibrated_surface")),
            source_ref=str(
                self.provenance.get("source_ref", "calibrate_single_name_credit_curve_workflow")
            ),
            selected_curve_roles=selected_curve_roles,
            metadata={
                "instrument_family": "credit",
                "instrument_kind": "single_name_cds",
                "curve_name": self.curve_name,
                "potential_binding": dict(self.potential_binding),
            },
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-friendly payload."""
        return {
            "quotes": [quote.to_payload() for quote in self.quotes],
            "credit_curve": {
                "tenors": [float(value) for value in self.credit_curve.tenors],
                "hazard_rates": [float(value) for value in self.credit_curve.hazard_rates],
            },
            "solve_request": self.solve_request.to_payload(),
            "solve_result": self.solve_result.to_payload(),
            "solver_provenance": self.solver_provenance.to_payload(),
            "solver_replay_artifact": self.solver_replay_artifact.to_payload(),
            "tenors": list(self.tenors),
            "target_hazards": list(self.target_hazards),
            "model_hazards": list(self.model_hazards),
            "target_quotes": list(self.target_quotes),
            "model_quotes": list(self.model_quotes),
            "hazard_residuals": list(self.hazard_residuals),
            "quote_residuals": list(self.quote_residuals),
            "max_abs_hazard_residual": self.max_abs_hazard_residual,
            "max_abs_quote_residual": self.max_abs_quote_residual,
            "curve_name": self.curve_name,
            "recovery": self.recovery,
            "potential_binding": dict(self.potential_binding),
            "provenance": dict(self.provenance),
            "summary": dict(self.summary),
            "warnings": list(self.warnings),
            "assumptions": list(self.assumptions),
        }


def _normalize_quotes(
    quotes: Sequence[CreditHazardCalibrationQuote],
) -> tuple[CreditHazardCalibrationQuote, ...]:
    """Return sorted quotes and validate maturity uniqueness."""
    resolved = tuple(quotes)
    if not resolved:
        raise ValueError("at least one credit calibration quote is required")
    sorted_quotes = tuple(
        sorted(
            resolved,
            key=lambda quote: (float(quote.maturity_years), quote.quote_kind, float(quote.quote)),
        )
    )
    maturities = [float(quote.maturity_years) for quote in sorted_quotes]
    for left, right in zip(maturities, maturities[1:]):
        if abs(float(right) - float(left)) <= 1e-12:
            raise ValueError("credit calibration quote maturities must be strictly increasing")
    return sorted_quotes


def calibrate_single_name_credit_curve_workflow(
    quotes: Sequence[CreditHazardCalibrationQuote],
    market_state: MarketState,
    *,
    recovery: float = 0.4,
    curve_name: str = "single_name_credit",
    max_hazard: float = 5.0,
) -> CreditHazardCalibrationResult:
    """Calibrate one reduced-form single-name credit curve from spread/hazard quotes."""
    if market_state.discount is None:
        raise ValueError("single-name credit calibration requires market_state.discount")
    recovery = float(recovery)
    if not raw_np.isfinite(recovery) or recovery <= 0.0 or recovery >= 1.0:
        raise ValueError("recovery must be strictly between 0 and 1")
    if not raw_np.isfinite(max_hazard) or float(max_hazard) <= 0.0:
        raise ValueError("max_hazard must be finite and positive")
    normalized_quotes = _normalize_quotes(quotes)
    labels = tuple(quote.resolved_label(index) for index, quote in enumerate(normalized_quotes))
    tenors = tuple(float(quote.maturity_years) for quote in normalized_quotes)
    potential_binding = _credit_potential_binding(
        market_state,
        curve_name=curve_name,
        recovery=recovery,
    )
    assumptions = _credit_quote_assumptions(market_state, recovery=recovery)
    quote_maps: list[CalibrationQuoteMap] = []
    for quote in normalized_quotes:
        if quote.quote_kind == "spread":
            quote_maps.append(
                _spread_quote_map(
                    recovery=recovery,
                    source_ref="_spread_quote_map",
                    assumptions=assumptions,
                    potential_binding=potential_binding,
                )
            )
        else:
            quote_maps.append(
                _hazard_quote_map(
                    assumptions=assumptions,
                    potential_binding=potential_binding,
                )
            )

    target_hazards_values: list[float] = []
    target_quote_values: list[float] = []
    quote_transform_warnings: list[str] = []
    for label, quote, quote_map in zip(labels, normalized_quotes, quote_maps):
        target_transform = quote_map.target_price(float(quote.quote))
        if target_transform.failure is not None:
            raise ValueError(f"credit quote_to_hazard failed for `{label}`: {target_transform.failure}")
        for warning in target_transform.warnings:
            quote_transform_warnings.append(f"{label}: {warning}")
        target_hazard = float(target_transform.value)
        if target_hazard <= 0.0 or not raw_np.isfinite(target_hazard):
            raise ValueError(f"credit quote `{label}` mapped to non-positive hazard `{target_hazard}`")
        target_hazards_values.append(target_hazard)
        target_quote_transform = quote_map.model_quote(target_hazard)
        if target_quote_transform.failure is not None:
            raise ValueError(
                f"credit hazard_to_quote failed for `{label}` while normalizing targets: {target_quote_transform.failure}"
            )
        target_quote_values.append(float(target_quote_transform.value))
    target_hazards = tuple(target_hazards_values)
    target_quotes = tuple(target_quote_values)
    weights = tuple(float(quote.weight) for quote in normalized_quotes)
    upper_bound = max(float(max_hazard), max(target_hazards) * 2.0, 1.0)
    parameter_names = tuple(f"hazard_{index + 1}" for index in range(len(normalized_quotes)))
    solve_request = SolveRequest(
        request_id="single_name_credit_hazard_least_squares",
        problem_kind="least_squares",
        parameter_names=parameter_names,
        initial_guess=target_hazards,
        objective=ObjectiveBundle(
            objective_kind="least_squares",
            labels=labels,
            target_values=target_hazards,
            weights=weights,
            vector_objective_fn=lambda params: raw_np.asarray(params, dtype=float),
            metadata={
                "model_family": "reduced_form_credit",
                "curve_name": curve_name,
                "quote_maps": [quote_map.to_payload() for quote_map in quote_maps],
                "potential_binding": dict(potential_binding),
            },
        ),
        bounds=SolveBounds(
            lower=tuple(0.0 for _ in parameter_names),
            upper=tuple(upper_bound for _ in parameter_names),
        ),
        solver_hint="trf",
        warm_start=WarmStart(parameter_values=target_hazards, source="quote_map_seed"),
        metadata={
            "curve_name": curve_name,
            "model_family": "reduced_form_credit",
            "selected_curve_names": dict(market_state.selected_curve_names or {}),
            "potential_binding": dict(potential_binding),
        },
        options={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 80},
    )
    solve_result = execute_solve_request(solve_request)
    if not solve_result.success:
        raise ValueError(
            "single-name credit calibration failed: "
            f"{solve_result.metadata.get('message', 'unknown failure')}"
        )
    model_hazards = tuple(float(value) for value in solve_result.solution)
    credit_curve = CreditCurve(tenors, model_hazards)

    model_quotes_values: list[float] = []
    quote_inverse_failures: list[str] = []
    for label, model_hazard, quote_map in zip(labels, model_hazards, quote_maps):
        model_transform = quote_map.model_quote(float(model_hazard))
        if model_transform.failure is not None:
            quote_inverse_failures.append(f"{label}: {model_transform.failure}")
            model_quotes_values.append(float("nan"))
            continue
        for warning in model_transform.warnings:
            quote_transform_warnings.append(f"{label}: {warning}")
        model_quotes_values.append(float(model_transform.value))
    model_quotes = tuple(model_quotes_values)
    hazard_residuals = tuple(
        float(model_hazard - target_hazard)
        for model_hazard, target_hazard in zip(model_hazards, target_hazards)
    )
    quote_residuals = tuple(
        float(model_quote - target_quote)
        for model_quote, target_quote in zip(model_quotes, target_quotes)
    )
    max_abs_hazard_residual = max((abs(value) for value in hazard_residuals), default=0.0)
    finite_quote_residuals = tuple(abs(value) for value in quote_residuals if value == value)
    max_abs_quote_residual = max(finite_quote_residuals, default=0.0)

    solver_provenance = build_solve_provenance(solve_request, solve_result)
    solver_replay_artifact = build_solve_replay_artifact(solve_request, solve_result)
    warnings: list[str] = []
    warnings.extend(quote_transform_warnings)
    warnings.extend(quote_inverse_failures)
    market_provenance = dict(getattr(market_state, "market_provenance", None) or {})
    if "source_kind" not in market_provenance:
        warnings.append(
            "market_state.market_provenance did not include source_kind; "
            "calibration preserved selected curve names only."
        )

    provenance = {
        "source_kind": "calibrated_surface",
        "source_ref": "calibrate_single_name_credit_curve_workflow",
        "selected_curve_names": dict(market_state.selected_curve_names or {}),
        "market_provenance": market_provenance,
        "potential_binding": dict(potential_binding),
        "calibration_target": {
            "labels": list(labels),
            "tenors": list(tenors),
            "target_hazards": list(target_hazards),
            "target_quotes": list(target_quotes),
            "quote_kinds": [quote.quote_kind for quote in normalized_quotes],
            "quote_maps": [quote_map.to_payload() for quote_map in quote_maps],
            "quote_inverse_failures": list(quote_inverse_failures),
            "curve_name": curve_name,
            "recovery": recovery,
        },
        "solve_request": solve_request.to_payload(),
        "solve_result": solve_result.to_payload(),
        "solver_provenance": solver_provenance.to_payload(),
        "solver_replay_artifact": solver_replay_artifact.to_payload(),
        "warnings": list(warnings),
        "assumptions": list(assumptions),
    }
    summary = {
        "quote_count": len(normalized_quotes),
        "curve_name": curve_name,
        "recovery": recovery,
        "max_abs_hazard_residual": float(max_abs_hazard_residual),
        "max_abs_quote_residual": float(max_abs_quote_residual),
        "quote_families": [quote_map.spec.quote_family for quote_map in quote_maps],
        "quote_conventions": [quote_map.spec.convention for quote_map in quote_maps],
    }
    return CreditHazardCalibrationResult(
        quotes=normalized_quotes,
        credit_curve=credit_curve,
        solve_request=solve_request,
        solve_result=solve_result,
        solver_provenance=solver_provenance,
        solver_replay_artifact=solver_replay_artifact,
        tenors=tenors,
        target_hazards=target_hazards,
        model_hazards=model_hazards,
        target_quotes=target_quotes,
        model_quotes=model_quotes,
        hazard_residuals=hazard_residuals,
        quote_residuals=quote_residuals,
        max_abs_hazard_residual=max_abs_hazard_residual,
        max_abs_quote_residual=max_abs_quote_residual,
        curve_name=curve_name,
        recovery=recovery,
        potential_binding=potential_binding,
        provenance=provenance,
        summary=summary,
        warnings=tuple(warnings),
        assumptions=assumptions,
    )


__all__ = [
    "CreditHazardCalibrationQuote",
    "CreditHazardCalibrationResult",
    "calibrate_single_name_credit_curve_workflow",
]
