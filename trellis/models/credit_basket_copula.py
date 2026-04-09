"""Semantic-facing basket-credit copula helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.instruments.nth_to_default import price_nth_to_default_basket
from trellis.models.copulas.factor import FactorCopula
from trellis.models.copulas.gaussian import GaussianCopula
from trellis.models.copulas.student_t import StudentTCopula


class DiscountCurveLike(Protocol):
    """Discount interface required by basket-credit helpers."""

    def discount(self, t: float) -> float:
        """Return a discount factor to time ``t``."""
        ...


class CreditCurveLike(Protocol):
    """Credit interface required by basket-credit helpers."""

    def survival_probability(self, t: float) -> float:
        """Return the survival probability to time ``t``."""
        ...


class CreditBasketMarketStateLike(Protocol):
    """Market-state interface required by semantic basket-credit helpers."""

    as_of: date | None
    settlement: date | None
    discount: DiscountCurveLike | None
    credit_curve: CreditCurveLike | None


class CreditBasketNthSpecLike(Protocol):
    """Minimal nth-to-default spec surface consumed by basket-credit helpers."""

    notional: float
    n_names: int
    n_th: int
    end_date: date


class CreditBasketTrancheSpecLike(Protocol):
    """Minimal tranche spec surface consumed by basket-credit helpers."""

    notional: float
    n_names: int
    attachment: float
    detachment: float
    end_date: date


@dataclass(frozen=True)
class ResolvedCreditBasketInputs:
    """Resolved market and contract inputs for bounded basket-credit helpers."""

    notional: float
    n_names: int
    horizon: float
    discount_factor: float
    survival_probability: float
    default_probability: float
    hazard_rate: float
    correlation: float
    recovery: float


@dataclass(frozen=True)
class CreditBasketTrancheResult:
    """Structured tranche-loss pricing result."""

    price: float
    expected_loss_fraction: float
    fair_spread_bp: float
    copula_family: str
    horizon: float


def resolve_credit_basket_inputs(
    market_state: CreditBasketMarketStateLike,
    spec: CreditBasketNthSpecLike | CreditBasketTrancheSpecLike,
) -> ResolvedCreditBasketInputs:
    """Resolve common market inputs for tranche and nth-to-default helpers."""
    settlement = market_state.settlement or market_state.as_of
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for basket-credit pricing")
    if market_state.discount is None:
        raise ValueError("basket-credit pricing requires market_state.discount")
    if market_state.credit_curve is None:
        raise ValueError("basket-credit pricing requires market_state.credit_curve")

    day_count = getattr(spec, "day_count", DayCountConvention.ACT_360)
    horizon = max(float(_credit_basket_horizon(settlement, spec.end_date, day_count)), 0.0)
    survival_probability = (
        1.0 if horizon <= 0.0 else float(market_state.credit_curve.survival_probability(horizon))
    )
    survival_probability = float(raw_np.clip(survival_probability, 1e-12, 1.0))
    default_probability = 1.0 - survival_probability
    hazard_rate = 0.0 if horizon <= 0.0 else float(-raw_np.log(survival_probability) / horizon)
    discount_factor = (
        1.0 if horizon <= 0.0 else float(market_state.discount.discount(horizon))
    )
    n_names = max(int(spec.n_names), 2)
    correlation = float(getattr(spec, "correlation", 0.3) or 0.3)
    recovery = float(getattr(spec, "recovery", 0.4) or 0.4)

    return ResolvedCreditBasketInputs(
        notional=float(spec.notional),
        n_names=n_names,
        horizon=horizon,
        discount_factor=discount_factor,
        survival_probability=survival_probability,
        default_probability=default_probability,
        hazard_rate=hazard_rate,
        correlation=correlation,
        recovery=recovery,
    )


def price_credit_basket_nth_to_default(
    market_state: CreditBasketMarketStateLike,
    spec: CreditBasketNthSpecLike,
    *,
    copula_family: str = "gaussian",
    degrees_of_freedom: float = 5.0,
    n_paths: int = 40_000,
    seed: int | None = 42,
) -> float:
    """Price an nth-to-default basket from semantic inputs."""
    resolved = resolve_credit_basket_inputs(market_state, spec)
    trigger_rank = max(int(spec.n_th), 1)
    family = _normalized_copula_family(copula_family)

    if family == "gaussian":
        return float(
            price_nth_to_default_basket(
                notional=resolved.notional,
                n_names=resolved.n_names,
                n_th=trigger_rank,
                horizon=resolved.horizon,
                correlation=resolved.correlation,
                recovery=resolved.recovery,
                credit_curve=market_state.credit_curve,
                discount_curve=market_state.discount,
            )
        )

    defaults_by_horizon = _simulate_default_counts(
        resolved,
        family=family,
        degrees_of_freedom=degrees_of_freedom,
        n_paths=n_paths,
        seed=seed,
    )
    trigger_probability = float(raw_np.mean(defaults_by_horizon >= trigger_rank))
    return float(
        resolved.notional
        * (1.0 - resolved.recovery)
        * resolved.discount_factor
        * trigger_probability
    )


def price_credit_basket_tranche_result(
    market_state: CreditBasketMarketStateLike,
    spec: CreditBasketTrancheSpecLike,
    *,
    copula_family: str = "gaussian",
    degrees_of_freedom: float = 5.0,
    n_paths: int = 40_000,
    seed: int | None = 42,
) -> CreditBasketTrancheResult:
    """Return a structured tranche-loss pricing result."""
    resolved = resolve_credit_basket_inputs(market_state, spec)
    family = _normalized_copula_family(copula_family)
    attachment = float(spec.attachment)
    detachment = float(spec.detachment)
    if not 0.0 <= attachment < detachment <= 1.0:
        raise ValueError("Tranche attachment/detachment must satisfy 0 <= A < D <= 1")

    if family == "gaussian":
        loss_counts, probabilities = FactorCopula(
            n_names=resolved.n_names,
            correlation=resolved.correlation,
        ).loss_distribution(resolved.default_probability)
        portfolio_loss = _portfolio_loss_fraction(loss_counts, resolved)
        tranche_loss = raw_np.clip(portfolio_loss - attachment, 0.0, detachment - attachment)
        expected_loss_fraction = float(raw_np.sum(tranche_loss * probabilities))
    else:
        defaults_by_horizon = _simulate_default_counts(
            resolved,
            family=family,
            degrees_of_freedom=degrees_of_freedom,
            n_paths=n_paths,
            seed=seed,
        )
        portfolio_loss = _portfolio_loss_fraction(defaults_by_horizon, resolved)
        tranche_loss = raw_np.clip(portfolio_loss - attachment, 0.0, detachment - attachment)
        expected_loss_fraction = float(raw_np.mean(tranche_loss))

    price = float(resolved.notional * resolved.discount_factor * expected_loss_fraction)
    tranche_width = detachment - attachment
    fair_spread_bp = 0.0
    annuity = _discounted_annuity(market_state.discount, resolved.horizon)
    if tranche_width > 0.0 and annuity > 0.0:
        fair_spread_bp = float(price / (resolved.notional * tranche_width * annuity) * 10_000.0)

    return CreditBasketTrancheResult(
        price=price,
        expected_loss_fraction=expected_loss_fraction,
        fair_spread_bp=fair_spread_bp,
        copula_family=family,
        horizon=resolved.horizon,
    )


def price_credit_basket_tranche(
    market_state: CreditBasketMarketStateLike,
    spec: CreditBasketTrancheSpecLike,
    *,
    copula_family: str = "gaussian",
    degrees_of_freedom: float = 5.0,
    n_paths: int = 40_000,
    seed: int | None = 42,
) -> float:
    """Return the scalar present value of a tranche-loss basket-credit price."""
    return float(
        price_credit_basket_tranche_result(
            market_state,
            spec,
            copula_family=copula_family,
            degrees_of_freedom=degrees_of_freedom,
            n_paths=n_paths,
            seed=seed,
        ).price
    )


def _credit_basket_horizon(start: date, end: date, day_count: DayCountConvention) -> float:
    """Normalize same-anniversary maturities to whole-year tenors for helper parity."""
    if (start.month, start.day) == (end.month, end.day):
        return float(end.year - start.year)
    return float(year_fraction(start, end, day_count))


def _normalized_copula_family(value: object) -> str:
    family = str(value or "gaussian").strip().lower().replace("-", "_")
    aliases = {
        "factor": "gaussian",
        "factor_gaussian": "gaussian",
        "gaussian_copula": "gaussian",
        "student_t_copula": "student_t",
        "t_copula": "student_t",
    }
    family = aliases.get(family, family)
    if family not in {"gaussian", "student_t"}:
        raise ValueError(f"Unsupported copula family {value!r}")
    return family


def _equicorrelation_matrix(n_names: int, correlation: float) -> raw_np.ndarray:
    corr = raw_np.full((n_names, n_names), float(correlation), dtype=float)
    raw_np.fill_diagonal(corr, 1.0)
    return corr


def _simulate_default_counts(
    resolved: ResolvedCreditBasketInputs,
    *,
    family: str,
    degrees_of_freedom: float,
    n_paths: int,
    seed: int | None,
) -> raw_np.ndarray:
    hazard_rates = raw_np.full(resolved.n_names, resolved.hazard_rate, dtype=float)
    rng = raw_np.random.default_rng(seed)
    corr = _equicorrelation_matrix(resolved.n_names, resolved.correlation)
    if family == "student_t":
        copula = StudentTCopula(correlation_matrix=corr, df=float(degrees_of_freedom))
    else:
        copula = GaussianCopula(correlation_matrix=corr)
    default_times = copula.sample_default_times(hazard_rates, int(n_paths), rng)
    return raw_np.sum(default_times <= resolved.horizon, axis=1)


def _portfolio_loss_fraction(
    default_counts,
    resolved: ResolvedCreditBasketInputs,
):
    defaults = raw_np.asarray(default_counts, dtype=float)
    return defaults * (1.0 - resolved.recovery) / float(resolved.n_names)


def _discounted_annuity(discount_curve: DiscountCurveLike, horizon: float) -> float:
    if horizon <= 0.0:
        return 0.0
    payment_count = max(int(raw_np.ceil(horizon * 4.0)), 1)
    payment_times = raw_np.linspace(horizon / payment_count, horizon, payment_count)
    discounted = raw_np.asarray([float(discount_curve.discount(float(t))) for t in payment_times])
    return float(raw_np.sum(discounted) / 4.0)


__all__ = [
    "CreditBasketTrancheResult",
    "ResolvedCreditBasketInputs",
    "price_credit_basket_nth_to_default",
    "price_credit_basket_tranche",
    "price_credit_basket_tranche_result",
    "resolve_credit_basket_inputs",
]
