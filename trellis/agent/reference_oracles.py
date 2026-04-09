"""Deterministic single-method reference oracles for eligible runtime builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trellis.agent.instrument_identity import resolve_authoritative_instrument_type
from trellis.agent.knowledge.methods import normalize_method
from trellis.engine.payoff_pricer import price_payoff


@dataclass(frozen=True)
class ReferenceOracleSpec:
    """Static oracle definition for one eligible single-method route."""

    oracle_id: str
    instrument_type: str
    method: str
    source: str
    relation: str
    tolerance: float = 1e-4


@dataclass(frozen=True)
class ReferenceOracleExecution:
    """Structured result of running one reference oracle."""

    oracle_id: str
    instrument_type: str
    method: str
    source: str
    relation: str
    tolerance: float
    passed: bool
    sampled_prices: tuple[dict[str, float], ...] = ()
    max_abs_deviation: float | None = None
    max_rel_deviation: float | None = None
    failure_message: str | None = None


def select_reference_oracle(
    *,
    instrument_type: str | None,
    method: str | None,
    product_ir=None,
    semantic_blueprint=None,
) -> ReferenceOracleSpec | None:
    """Return the deterministic oracle definition for one supported route."""
    normalized_instrument = resolve_authoritative_instrument_type(
        instrument_type,
        getattr(product_ir, "instrument", None),
        getattr(semantic_blueprint, "semantic_id", None),
    ) or ""
    normalized_method = normalize_method(method or "")
    if normalized_method == "analytical" and normalized_instrument == "swaption":
        return ReferenceOracleSpec(
            oracle_id="swaption_black76_exact",
            instrument_type=normalized_instrument,
            method=normalized_method,
            source="trellis.models.rate_style_swaption.price_swaption_black76",
            relation="within_tolerance",
        )
    if normalized_method == "analytical" and normalized_instrument == "zcb_option":
        return ReferenceOracleSpec(
            oracle_id="zcb_option_jamshidian_exact",
            instrument_type=normalized_instrument,
            method=normalized_method,
            source="trellis.models.zcb_option.price_zcb_option_jamshidian",
            relation="within_tolerance",
        )
    if normalized_method == "rate_tree" and normalized_instrument == "callable_bond":
        return ReferenceOracleSpec(
            oracle_id="callable_bond_straight_bond_bound",
            instrument_type=normalized_instrument,
            method=normalized_method,
            source="reference_factory:straight_bond_bound",
            relation="<=",
        )
    if normalized_method == "rate_tree" and normalized_instrument == "puttable_bond":
        return ReferenceOracleSpec(
            oracle_id="puttable_bond_straight_bond_bound",
            instrument_type=normalized_instrument,
            method=normalized_method,
            source="reference_factory:straight_bond_bound",
            relation=">=",
        )
    return None


def execute_reference_oracle(
    oracle: ReferenceOracleSpec | None,
    *,
    payoff_factory: Callable[[], Any],
    market_state_factory: Callable[..., Any],
    reference_factory: Callable[[], Any] | None = None,
    semantic_blueprint=None,
) -> ReferenceOracleExecution | None:
    """Execute one selected oracle and return a structured outcome."""
    if oracle is None:
        return None

    sampled_prices: list[dict[str, float]] = []
    max_abs_deviation = 0.0
    max_rel_deviation = 0.0
    failure_message: str | None = None
    passed = True

    for scenario in _scenario_grid_for(oracle):
        try:
            market_state = market_state_factory(**scenario)
            payoff = payoff_factory()
            generated = float(price_payoff(payoff, market_state))
            reference = float(
                _reference_value_for(
                    oracle,
                    market_state,
                    payoff,
                    reference_factory,
                    semantic_blueprint=semantic_blueprint,
                )
            )
            abs_deviation = abs(generated - reference)
            rel_deviation = abs_deviation / max(abs(reference), 1.0)
            max_abs_deviation = max(max_abs_deviation, abs_deviation)
            max_rel_deviation = max(max_rel_deviation, rel_deviation)
            sampled_prices.append(
                {
                    "rate": float(scenario.get("rate", 0.0)),
                    "vol": float(scenario.get("vol", 0.0)),
                    "generated": generated,
                    "reference": reference,
                    "abs_deviation": abs_deviation,
                    "rel_deviation": rel_deviation,
                }
            )
            if _violates_relation(
                oracle.relation,
                generated=generated,
                reference=reference,
                tolerance=oracle.tolerance,
            ):
                passed = False
                failure_message = (
                    f"Reference oracle `{oracle.oracle_id}` failed: generated payoff {generated:.6f} "
                    f"violates relation `{oracle.relation}` against reference {reference:.6f} "
                    f"at rate={scenario.get('rate', 0.0):.2%}, vol={scenario.get('vol', 0.0):.2%}."
                )
                break
        except Exception as exc:
            passed = False
            failure_message = f"Reference oracle `{oracle.oracle_id}` failed: {exc}"
            break

    return ReferenceOracleExecution(
        oracle_id=oracle.oracle_id,
        instrument_type=oracle.instrument_type,
        method=oracle.method,
        source=oracle.source,
        relation=oracle.relation,
        tolerance=float(oracle.tolerance),
        passed=passed,
        sampled_prices=tuple(sampled_prices),
        max_abs_deviation=float(max_abs_deviation) if sampled_prices else None,
        max_rel_deviation=float(max_rel_deviation) if sampled_prices else None,
        failure_message=failure_message,
    )


def reference_oracle_summary(
    execution: ReferenceOracleExecution | None,
) -> dict[str, Any] | None:
    """Project one oracle execution into YAML-safe primitives."""
    if execution is None:
        return None
    return {
        "oracle_id": execution.oracle_id,
        "instrument_type": execution.instrument_type,
        "method": execution.method,
        "source": execution.source,
        "relation": execution.relation,
        "tolerance": execution.tolerance,
        "passed": execution.passed,
        "sampled_prices": [dict(sample) for sample in execution.sampled_prices],
        "max_abs_deviation": execution.max_abs_deviation,
        "max_rel_deviation": execution.max_rel_deviation,
        "failure_message": execution.failure_message,
    }


def _extract_spec(payoff) -> Any | None:
    """Return the bound spec object for an instantiated payoff."""
    spec = getattr(payoff, "spec", None)
    if spec is None:
        spec = getattr(payoff, "_spec", None)
    return spec


def _reference_value_for(
    oracle: ReferenceOracleSpec,
    market_state,
    payoff,
    reference_factory: Callable[[], Any] | None,
    *,
    semantic_blueprint=None,
) -> float:
    """Return the exact or bound reference value for one oracle scenario."""
    spec = _extract_spec(payoff)
    if oracle.oracle_id == "swaption_black76_exact":
        from trellis.models.rate_style_swaption import price_swaption_black76

        return float(
            price_swaption_black76(
                market_state,
                spec,
                **_swaption_comparison_kwargs_from_blueprint(semantic_blueprint),
            )
        )
    if oracle.oracle_id == "zcb_option_jamshidian_exact":
        from trellis.models.zcb_option import price_zcb_option_jamshidian

        return float(price_zcb_option_jamshidian(market_state, spec, mean_reversion=0.1))
    if oracle.oracle_id in {
        "callable_bond_straight_bond_bound",
        "puttable_bond_straight_bond_bound",
    }:
        if reference_factory is None:
            raise ValueError("Bound-style reference oracle requires reference_factory")
        return float(price_payoff(reference_factory(), market_state))
    raise ValueError(f"Unsupported reference oracle {oracle.oracle_id!r}")


def _swaption_comparison_kwargs_from_blueprint(semantic_blueprint) -> dict[str, float]:
    """Return explicit comparison kwargs for swaption oracles when present."""
    valuation_context = getattr(semantic_blueprint, "valuation_context", None)
    engine_model_spec = getattr(valuation_context, "engine_model_spec", None)
    if engine_model_spec is None or getattr(engine_model_spec, "model_name", "") != "hull_white_1f":
        return {}
    overrides = dict(getattr(engine_model_spec, "parameter_overrides", {}) or {})
    mean_reversion = overrides.get("mean_reversion")
    sigma = overrides.get("sigma")
    if mean_reversion is None or sigma is None:
        return {}
    return {
        "mean_reversion": float(mean_reversion),
        "sigma": float(sigma),
    }


def _scenario_grid_for(oracle: ReferenceOracleSpec) -> tuple[Mapping[str, float], ...]:
    """Return the deterministic market scenarios for one oracle."""
    if oracle.relation == "within_tolerance":
        return (
            {"rate": 0.03, "vol": 0.15},
            {"rate": 0.05, "vol": 0.20},
            {"rate": 0.07, "vol": 0.30},
        )
    return (
        {"rate": 0.02, "vol": 0.20},
        {"rate": 0.05, "vol": 0.20},
        {"rate": 0.08, "vol": 0.20},
    )


def _violates_relation(
    relation: str,
    *,
    generated: float,
    reference: float,
    tolerance: float,
) -> bool:
    """Return whether the generated price violates the oracle relation."""
    slack = float(tolerance) * max(abs(reference), 1.0)
    if relation == "<=":
        return generated > reference + slack
    if relation == ">=":
        return generated < reference - slack
    return abs(generated - reference) > slack
