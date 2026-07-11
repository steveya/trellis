"""Bounded single-underlier autocallable Monte Carlo helpers.

The helper surface intentionally covers the compact proof-task contract:
fixed observation schedule, first-trigger redemption, linear coupon accrual,
terminal protection, deterministic discounting, and pseudo/Sobol shock
selection over the same event payoff semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import math
from typing import Any

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.monte_carlo.variance_reduction import sobol_normals
from trellis.models.processes.gbm import GBM


@dataclass(frozen=True)
class AutocallableRuntimeSpec:
    """Resolved single-underlier autocallable contract used at runtime."""

    notional: float = 1.0
    initial_spot: float = 100.0
    spot: float = 100.0
    maturity: float = 1.0
    rate: float = 0.0
    sigma: float = 0.2
    observation_times: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    autocall_barrier: float = 100.0
    protection_barrier: float = 70.0
    coupon_rate: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.notional < 0.0:
            raise ValueError("notional must be non-negative")
        if self.initial_spot <= 0.0:
            raise ValueError("initial_spot must be positive")
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.maturity < 0.0:
            raise ValueError("maturity must be non-negative")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        observation_times = tuple(
            sorted({float(time) for time in self.observation_times if float(time) >= 0.0})
        )
        if not observation_times:
            observation_times = (float(self.maturity),)
        maturity = max(float(self.maturity), max(observation_times))
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(
            self,
            "observation_times",
            tuple(min(max(float(time), 0.0), maturity) for time in observation_times),
        )

    @classmethod
    def from_spec(cls, spec: Any, **overrides: Any) -> "AutocallableRuntimeSpec":
        """Build a runtime spec from common task-spec and generated-adapter aliases."""
        initial_spot = float(
            _coalesce_attr(
                spec,
                ("initial_spot", "spot", "underlier_spot", "s0", "strike_reference"),
                100.0,
            )
        )
        maturity = float(
            _coalesce_attr(
                spec,
                ("maturity", "expiry_years", "time_to_maturity", "tenor_years"),
                1.0,
            )
        )
        observation_times = _coerce_observation_times(
            _coalesce_attr(spec, ("observation_times", "fixing_times"), None),
            maturity=maturity,
            n_observations=_coalesce_attr(
                spec,
                ("n_observations", "num_observations", "observation_count"),
                None,
            ),
        )
        values = {
            "notional": _coalesce_attr(spec, ("notional", "face", "principal"), 1.0),
            "initial_spot": initial_spot,
            "spot": _coalesce_attr(spec, ("spot", "underlier_spot", "s0"), initial_spot),
            "maturity": maturity,
            "rate": _coalesce_attr(spec, ("rate", "risk_free_rate", "r"), 0.0),
            "sigma": _coalesce_attr(spec, ("sigma", "vol", "volatility"), 0.2),
            "observation_times": observation_times,
            "autocall_barrier": _coerce_level(
                _coalesce_attr(
                    spec,
                    ("autocall_barrier", "call_barrier", "trigger_level", "barrier", "barrier_level"),
                    1.0,
                ),
                initial_spot,
            ),
            "protection_barrier": _coerce_level(
                _coalesce_attr(
                    spec,
                    (
                        "protection_barrier",
                        "capital_protection_barrier",
                        "downside_barrier",
                        "protection_level",
                    ),
                    0.7,
                ),
                initial_spot,
            ),
            "coupon_rate": _coerce_rate(
                _coalesce_attr(spec, ("coupon_rate", "coupon", "annual_coupon"), 0.0)
            ),
            "dividend_yield": _coerce_rate(
                _coalesce_attr(spec, ("dividend_yield", "q", "carry_rate"), 0.0)
            ),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class AutocallableMonteCarloConfig:
    """Simulation controls for the autocallable MC/QMC helper."""

    n_paths: int = 50_000
    n_steps: int = 252
    seed: int | None = 12345
    sampling: str = "pseudo"


@dataclass(frozen=True)
class AutocallableMonteCarloResult:
    """Structured result and evidence for autocallable MC/QMC pricing."""

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    sampling: str
    resolved_spec: AutocallableRuntimeSpec
    observation_steps: tuple[int, ...]
    path_contract: tuple[str, ...]
    validation_bundle: str


def resolve_autocallable_inputs(market_state, spec) -> AutocallableRuntimeSpec:
    """Resolve dates, market data, and aliases into an autocallable runtime spec."""
    base = (
        spec
        if isinstance(spec, AutocallableRuntimeSpec)
        else AutocallableRuntimeSpec.from_spec(spec)
    )
    maturity = _resolve_maturity(market_state, spec, default=base.maturity)
    initial_spot = _resolve_initial_spot(market_state, spec, default=base.initial_spot)
    spot = _resolve_spot(market_state, spec, default=initial_spot)
    observation_times = _resolve_observation_times(market_state, spec, maturity=maturity)
    rate = _resolve_rate(market_state, maturity, default=base.rate)
    sigma = _resolve_sigma(market_state, maturity, initial_spot, default=base.sigma)
    return replace(
        base,
        initial_spot=initial_spot,
        spot=spot,
        maturity=maturity,
        rate=rate,
        sigma=sigma,
        observation_times=observation_times,
        autocall_barrier=_coerce_level(
            _coalesce_attr(
                spec,
                ("autocall_barrier", "call_barrier", "trigger_level", "barrier", "barrier_level"),
                base.autocall_barrier,
            ),
            initial_spot,
        ),
        protection_barrier=_coerce_level(
            _coalesce_attr(
                spec,
                (
                    "protection_barrier",
                    "capital_protection_barrier",
                    "downside_barrier",
                    "protection_level",
                ),
                base.protection_barrier,
            ),
            initial_spot,
        ),
        coupon_rate=_coerce_rate(
            _coalesce_attr(spec, ("coupon_rate", "coupon", "annual_coupon"), base.coupon_rate)
        ),
        dividend_yield=_coerce_rate(
            _coalesce_attr(spec, ("dividend_yield", "q", "carry_rate"), base.dividend_yield)
        ),
    )


def autocallable_observation_steps(
    spec: AutocallableRuntimeSpec,
    n_steps: int,
) -> tuple[int, ...]:
    """Map continuous observation times to simulation step indices."""
    steps = max(int(n_steps), 1)
    maturity = max(float(spec.maturity), 1e-12)
    mapped = tuple(
        min(max(int(round(float(time) / maturity * steps)), 1), steps)
        for time in spec.observation_times
    )
    return tuple(sorted(dict.fromkeys(mapped)))


def autocallable_path_payoffs(
    paths,
    spec: AutocallableRuntimeSpec,
) -> tuple[raw_np.ndarray, raw_np.ndarray, raw_np.ndarray]:
    """Return pathwise cashflows, payment times, and first-trigger indicators."""
    path_array = raw_np.asarray(paths, dtype=float)
    if path_array.ndim != 2 or path_array.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1)")
    n_steps = path_array.shape[1] - 1
    observation_steps = raw_np.asarray(autocallable_observation_steps(spec, n_steps), dtype=int)
    observed = path_array[:, observation_steps]
    triggers = observed >= float(spec.autocall_barrier)
    called = raw_np.any(triggers, axis=1)
    first_positions = raw_np.argmax(triggers, axis=1)
    trigger_steps = observation_steps[first_positions]
    trigger_times = trigger_steps.astype(float) / float(n_steps) * float(spec.maturity)

    terminal = path_array[:, -1]
    terminal_ratio = raw_np.maximum(terminal, 0.0) / float(spec.initial_spot)
    terminal_cashflows = raw_np.where(
        terminal >= float(spec.protection_barrier),
        float(spec.notional) * (1.0 + float(spec.coupon_rate) * float(spec.maturity)),
        float(spec.notional) * terminal_ratio,
    )
    trigger_cashflows = float(spec.notional) * (
        1.0 + float(spec.coupon_rate) * trigger_times
    )
    cashflows = raw_np.where(called, trigger_cashflows, terminal_cashflows)
    payment_times = raw_np.where(called, trigger_times, float(spec.maturity))
    return cashflows, payment_times, called


def price_autocallable_monte_carlo_result(
    market_state,
    spec,
    *,
    config: AutocallableMonteCarloConfig | None = None,
    sampling: str | None = None,
) -> AutocallableMonteCarloResult:
    """Return a pseudo-MC or Sobol-QMC price for the bounded autocallable contract."""
    resolved = resolve_autocallable_inputs(market_state, spec)
    cfg = config or AutocallableMonteCarloConfig()
    sample_kind = _normalize_sampling(sampling or cfg.sampling)
    n_paths = max(int(cfg.n_paths), 1)
    n_steps = max(int(cfg.n_steps), max(autocallable_observation_steps(resolved, cfg.n_steps)))
    paths = _simulate_gbm_paths(
        resolved,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=cfg.seed,
        sampling=sample_kind,
    )
    cashflows, payment_times, _called = autocallable_path_payoffs(paths, resolved)
    discounts = raw_np.exp(-float(resolved.rate) * payment_times)
    discounted = discounts * cashflows
    price = float(raw_np.mean(discounted))
    std_error = float(raw_np.std(discounted, ddof=1) / math.sqrt(n_paths)) if n_paths > 1 else 0.0
    observation_steps = autocallable_observation_steps(resolved, n_steps)
    return AutocallableMonteCarloResult(
        price=price,
        std_error=std_error,
        n_paths=n_paths,
        n_steps=n_steps,
        sampling=sample_kind,
        resolved_spec=resolved,
        observation_steps=observation_steps,
        path_contract=(
            "gbm_exact",
            "fixed_observation_schedule",
            "first_trigger_redemption",
            "linear_coupon_accrual",
            "terminal_protection_barrier",
        ),
        validation_bundle=(
            "autocallable:qmc_sobol_gbm"
            if sample_kind == "sobol"
            else "autocallable:monte_carlo_gbm"
        ),
    )


def price_autocallable_monte_carlo(
    market_state,
    spec,
    *,
    config: AutocallableMonteCarloConfig | None = None,
    sampling: str | None = None,
) -> float:
    """Return the scalar autocallable MC/QMC helper price."""
    return float(
        price_autocallable_monte_carlo_result(
            market_state,
            spec,
            config=config,
            sampling=sampling,
        ).price
    )


def _simulate_gbm_paths(
    spec: AutocallableRuntimeSpec,
    *,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    sampling: str,
) -> raw_np.ndarray:
    dt = float(spec.maturity) / max(int(n_steps), 1)
    shocks = _normal_shocks(
        n_paths=max(int(n_paths), 1),
        n_steps=max(int(n_steps), 1),
        seed=seed,
        sampling=sampling,
    )
    paths = raw_np.empty((max(int(n_paths), 1), max(int(n_steps), 1) + 1), dtype=float)
    paths[:, 0] = float(spec.spot)
    process = GBM(
        mu=float(spec.rate) - float(spec.dividend_yield),
        sigma=float(spec.sigma),
    )
    for step in range(max(int(n_steps), 1)):
        paths[:, step + 1] = process.exact_sample(paths[:, step], step * dt, dt, shocks[:, step])
    return paths


def _normal_shocks(
    *,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    sampling: str,
) -> raw_np.ndarray:
    if sampling == "sobol":
        return raw_np.asarray(sobol_normals(n_paths, n_steps, n_factors=1), dtype=float)
    rng = raw_np.random.default_rng(seed)
    return rng.standard_normal((n_paths, n_steps))


def _normalize_sampling(value: str) -> str:
    sampling = str(value or "pseudo").strip().lower().replace("-", "_")
    if sampling in {"sobol", "qmc", "quasi_monte_carlo", "quasi_random"}:
        return "sobol"
    if sampling in {"pseudo", "pseudo_random", "mc", "monte_carlo"}:
        return "pseudo"
    raise ValueError("sampling must be 'pseudo' or 'sobol'")


def _resolve_maturity(market_state, spec, *, default: float) -> float:
    for attr in ("maturity", "expiry_years", "time_to_maturity", "tenor_years"):
        value = getattr(spec, attr, None)
        if value is not None:
            return max(float(value), 0.0)
    expiry = getattr(spec, "expiry_date", None) or getattr(spec, "maturity_date", None)
    if expiry is not None:
        if isinstance(expiry, str):
            expiry = date.fromisoformat(expiry)
        settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
        if settlement is not None:
            day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
            return max(float(year_fraction(settlement, expiry, day_count)), 0.0)
    return max(float(default), 0.0)


def _resolve_observation_times(market_state, spec, *, maturity: float) -> tuple[float, ...]:
    explicit_times = _coalesce_attr(spec, ("observation_times", "fixing_times"), None)
    if explicit_times is not None:
        return _coerce_observation_times(
            explicit_times,
            maturity=maturity,
            n_observations=None,
        )
    observation_dates = _coalesce_attr(
        spec,
        ("observation_dates", "fixing_dates", "autocall_dates"),
        None,
    )
    if observation_dates is not None:
        settlement = getattr(market_state, "settlement", None) or getattr(market_state, "as_of", None)
        day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
        return _coerce_observation_dates(
            observation_dates,
            settlement=settlement,
            maturity=maturity,
            day_count=day_count,
        )
    n_observations = _coalesce_attr(
        spec,
        ("n_observations", "num_observations", "observation_count"),
        None,
    )
    return _coerce_observation_times(
        None,
        maturity=maturity,
        n_observations=n_observations,
    )


def _resolve_initial_spot(market_state, spec, *, default: float) -> float:
    for attr in ("initial_spot", "spot", "underlier_spot", "s0", "strike_reference"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    value = getattr(market_state, "spot", None)
    if value is not None:
        return float(value)
    return float(default)


def _resolve_spot(market_state, spec, *, default: float) -> float:
    for attr in ("spot", "underlier_spot", "s0"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    value = getattr(market_state, "spot", None)
    if value is not None:
        return float(value)
    return float(default)


def _resolve_rate(market_state, maturity: float, *, default: float) -> float:
    discount = getattr(market_state, "discount", None)
    if discount is None or maturity <= 0.0:
        return float(default)
    return float(discount.zero_rate(max(maturity, 1e-8)))


def _resolve_sigma(market_state, maturity: float, strike: float, *, default: float) -> float:
    vol_surface = getattr(market_state, "vol_surface", None)
    if vol_surface is None or maturity <= 0.0:
        return float(default)
    return float(vol_surface.black_vol(max(maturity, 1e-8), strike))


def _coerce_observation_dates(
    values,
    *,
    settlement,
    maturity: float,
    day_count,
) -> tuple[float, ...]:
    if settlement is None:
        return _coerce_observation_times(None, maturity=maturity, n_observations=None)
    dates = values.split(",") if isinstance(values, str) else values
    times: list[float] = []
    for value in dates:
        event_date = date.fromisoformat(value.strip()) if isinstance(value, str) else value
        times.append(max(float(year_fraction(settlement, event_date, day_count)), 0.0))
    return tuple(time for time in sorted(dict.fromkeys(times)) if time <= max(maturity, 0.0))


def _coerce_observation_times(
    values,
    *,
    maturity: float,
    n_observations,
) -> tuple[float, ...]:
    maturity = max(float(maturity), 0.0)
    if values is not None:
        raw_values = values.split(",") if isinstance(values, str) else values
        return tuple(
            time
            for time in sorted(dict.fromkeys(float(value) for value in raw_values))
            if 0.0 < time <= max(maturity, 0.0)
        ) or (maturity,)
    if n_observations is not None:
        count = max(int(n_observations), 1)
    else:
        count = max(int(round(maturity * 4.0)), 1)
    if maturity <= 0.0:
        return (0.0,)
    return tuple(float(maturity) * index / count for index in range(1, count + 1))


def _coerce_level(value, initial_spot: float) -> float:
    level = float(value)
    if 0.0 < level <= 3.0:
        return float(initial_spot) * level
    return level


def _coerce_rate(value) -> float:
    rate = float(value)
    if abs(rate) > 1.0 and abs(rate) <= 100.0:
        return rate / 100.0
    return rate


def _coalesce_attr(spec, names: tuple[str, ...], default):
    for name in names:
        value = getattr(spec, name, None)
        if value is not None:
            return value
    return default


__all__ = [
    "AutocallableMonteCarloConfig",
    "AutocallableMonteCarloResult",
    "AutocallableRuntimeSpec",
    "autocallable_observation_steps",
    "autocallable_path_payoffs",
    "price_autocallable_monte_carlo",
    "price_autocallable_monte_carlo_result",
    "resolve_autocallable_inputs",
]
