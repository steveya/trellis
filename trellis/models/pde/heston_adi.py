"""Checked Heston two-factor PDE helper with ADI-style operator splitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as raw_np

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import normalized_option_type, terminal_intrinsic
from trellis.models.pde.thomas import thomas_solve
from trellis.models.processes.heston import HestonRuntimeBinding, resolve_heston_runtime_binding
from trellis.models.transforms.heston import price_heston_option_transform


@dataclass(frozen=True)
class HestonAdiPDEConfig:
    """Numerical controls for the bounded Heston PDE helper."""

    spot_steps: int = 80
    variance_steps: int = 36
    time_steps: int = 120
    spot_max_multiplier: float = 4.0
    variance_max_multiplier: float = 6.0
    theta: float = 0.5
    reference_method: str | None = None


@dataclass(frozen=True)
class HestonAdiPDEResult:
    """Structured Heston ADI PDE result and route diagnostics."""

    price: float
    raw_adi_price: float
    reference_price: float | None
    reference_method: str | None
    reference_relative_error: float | None
    grid_shape: tuple[int, int]
    time_steps: int
    maturity: float
    model_parameters: dict[str, object]
    runtime_binding: dict[str, object]
    validation_bundle: str = "heston:adi_pde"


@dataclass(frozen=True)
class ResolvedHestonAdiPDEInputs:
    """Resolved inputs for the Heston ADI PDE route."""

    notional: float
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    runtime_binding: HestonRuntimeBinding
    validation_bundle: str = "heston:adi_pde"


def resolve_heston_adi_pde_inputs(
    market_state,
    spec,
    *,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> ResolvedHestonAdiPDEInputs:
    """Resolve market, contract, and model inputs for Heston ADI PDE pricing."""
    settlement = getattr(market_state, "settlement", None) or getattr(
        market_state,
        "as_of",
        None,
    )
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of for Heston ADI PDE pricing")
    day_count = getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = _resolve_maturity(settlement, spec, day_count)
    if getattr(market_state, "discount", None) is None and maturity > 0.0:
        raise ValueError("Heston ADI PDE pricing requires market_state.discount")

    rate = (
        0.0
        if maturity <= 0.0
        else float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    )
    dividend_yield = _resolve_dividend_yield(spec)
    runtime_mu = float(mu) if mu is not None else rate - dividend_yield
    spec_parameters = _resolve_spec_heston_parameters(spec)
    runtime_binding = resolve_heston_runtime_binding(
        market_state,
        mu=runtime_mu,
        kappa=spec_parameters.get("kappa"),
        theta=spec_parameters.get("theta"),
        xi=spec_parameters.get("xi"),
        rho=spec_parameters.get("rho"),
        v0=spec_parameters.get("v0"),
        parameter_set_name=parameter_set_name,
    )
    spot = _resolve_spot(market_state, spec)
    strike = _resolve_strike(spec, spot=spot)
    notional_value = getattr(spec, "notional", 1.0)
    notional = 1.0 if notional_value is None else float(notional_value)

    return ResolvedHestonAdiPDEInputs(
        notional=notional,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=normalized_option_type(getattr(spec, "option_type", "call")),
        runtime_binding=runtime_binding,
    )


def price_heston_option_adi_pde_result(
    market_state,
    spec,
    *,
    config: HestonAdiPDEConfig | None = None,
    mu: float | None = None,
    parameter_set_name: str = "heston",
) -> HestonAdiPDEResult:
    """Return a structured European Heston option price from the PDE route."""
    resolved = resolve_heston_adi_pde_inputs(
        market_state,
        spec,
        mu=mu,
        parameter_set_name=parameter_set_name,
    )
    cfg = config or HestonAdiPDEConfig()
    if resolved.maturity <= 0.0:
        price = float(
            resolved.notional
            * terminal_intrinsic(
                resolved.option_type,
                spot=resolved.spot,
                strike=resolved.strike,
            )
        )
    else:
        price = float(resolved.notional) * _solve_heston_adi_price(resolved, cfg)

    raw_adi_price = float(price)
    reference_price: float | None = None
    reference_method: str | None = None
    reference_relative_error: float | None = None
    if cfg.reference_method and resolved.maturity > 0.0:
        reference_method = str(cfg.reference_method or "fft").strip().lower()
        reference_price = float(
            price_heston_option_transform(
                market_state,
                spec,
                method=reference_method,
                mu=mu,
                parameter_set_name=parameter_set_name,
            )
        )
        denominator = max(abs(reference_price), 1.0)
        reference_relative_error = abs(raw_adi_price - reference_price) / denominator

    runtime_payload = resolved.runtime_binding.to_payload()
    return HestonAdiPDEResult(
        price=float(price),
        raw_adi_price=raw_adi_price,
        reference_price=reference_price,
        reference_method=reference_method,
        reference_relative_error=reference_relative_error,
        grid_shape=(max(int(cfg.spot_steps), 5), max(int(cfg.variance_steps), 5)),
        time_steps=max(int(cfg.time_steps), 1),
        maturity=float(resolved.maturity),
        model_parameters=dict(resolved.runtime_binding.model_parameters),
        runtime_binding=runtime_payload,
    )


def _solve_heston_adi_price(resolved, cfg: HestonAdiPDEConfig) -> float:
    process = resolved.runtime_binding.process
    n_s = max(int(cfg.spot_steps), 5)
    n_v = max(int(cfg.variance_steps), 5)
    n_t = max(int(cfg.time_steps), 1)
    dt = float(resolved.maturity) / n_t
    theta = min(max(float(cfg.theta), 0.0), 1.0)

    s_max = max(
        float(cfg.spot_max_multiplier) * max(resolved.spot, 1e-8),
        2.0 * max(resolved.strike, 1e-8),
    )
    v_max = _variance_grid_upper_bound(process, resolved.maturity, cfg)
    s_grid = raw_np.linspace(0.0, s_max, n_s)
    v_grid = raw_np.linspace(0.0, v_max, n_v)
    values = terminal_intrinsic(
        resolved.option_type,
        spot=s_grid[:, None],
        strike=resolved.strike,
    ) * raw_np.ones((1, n_v))
    _apply_spot_boundaries(values, s_grid, tau=0.0, resolved=resolved)

    for step in range(n_t):
        tau = (step + 1) * dt
        current = values.copy()
        explicit = (
            _apply_spot_operator(current, s_grid, v_grid, resolved)
            + _apply_variance_operator(current, v_grid, process)
            + _apply_mixed_operator(current, s_grid, v_grid, process)
        )
        y0 = current + dt * explicit
        y1_rhs = y0 - theta * dt * _apply_spot_operator(current, s_grid, v_grid, resolved)
        y1 = _solve_spot_direction(
            y1_rhs,
            s_grid,
            v_grid,
            resolved,
            theta_dt=theta * dt,
            tau=tau,
        )
        y2_rhs = y1 - theta * dt * _apply_variance_operator(current, v_grid, process)
        values = _solve_variance_direction(
            y2_rhs,
            v_grid,
            process,
            theta_dt=theta * dt,
        )
        _apply_spot_boundaries(values, s_grid, tau=tau, resolved=resolved)
        values[:, 0] = values[:, 1]
        values[:, -1] = values[:, -2]

    interpolated_variance = raw_np.asarray(
        [
            raw_np.interp(process.v0, v_grid, values[i, :])
            for i in range(n_s)
        ],
        dtype=float,
    )
    return float(raw_np.interp(resolved.spot, s_grid, interpolated_variance))


def _variance_grid_upper_bound(process, maturity: float, cfg: HestonAdiPDEConfig) -> float:
    """Return a scale-aware upper variance bound for the Heston ADI grid."""
    theta = max(float(process.theta), 1e-10)
    v0 = max(float(process.v0), 1e-10)
    xi = abs(float(process.xi))
    kappa = max(float(process.kappa), 1e-10)
    t = max(float(maturity), 0.0)

    long_run_bound = float(cfg.variance_max_multiplier) * max(theta, v0, 1e-6)
    if t <= 0.0 or xi <= 1e-14:
        dispersion_bound = max(theta, v0)
    else:
        variance_of_variance = (
            theta
            * xi
            * xi
            / (2.0 * kappa)
            * (1.0 - raw_np.exp(-2.0 * kappa * t))
        )
        std_variance = float(raw_np.sqrt(max(variance_of_variance, 0.0)))
        dispersion_bound = max(theta, v0) + 4.0 * std_variance
    return max(long_run_bound, dispersion_bound, v0 + 1e-4, 0.02)


def _apply_spot_operator(values, s_grid, v_grid, resolved):
    out = raw_np.zeros_like(values)
    ds = s_grid[1] - s_grid[0]
    r = float(resolved.rate)
    for i in range(1, len(s_grid) - 1):
        s = s_grid[i]
        second = (values[i - 1, :] - 2.0 * values[i, :] + values[i + 1, :]) / (ds * ds)
        first = (values[i + 1, :] - values[i - 1, :]) / (2.0 * ds)
        out[i, :] = 0.5 * v_grid * s * s * second + r * s * first - r * values[i, :]
    return out


def _apply_variance_operator(values, v_grid, process):
    out = raw_np.zeros_like(values)
    dv = v_grid[1] - v_grid[0]
    for j in range(1, len(v_grid) - 1):
        v = v_grid[j]
        second = (values[:, j - 1] - 2.0 * values[:, j] + values[:, j + 1]) / (dv * dv)
        first = (values[:, j + 1] - values[:, j - 1]) / (2.0 * dv)
        out[:, j] = 0.5 * process.xi * process.xi * v * second + process.kappa * (process.theta - v) * first
    return out


def _apply_mixed_operator(values, s_grid, v_grid, process):
    out = raw_np.zeros_like(values)
    ds = s_grid[1] - s_grid[0]
    dv = v_grid[1] - v_grid[0]
    for i in range(1, len(s_grid) - 1):
        s = s_grid[i]
        cross = (
            values[i + 1, 2:]
            - values[i + 1, :-2]
            - values[i - 1, 2:]
            + values[i - 1, :-2]
        ) / (4.0 * ds * dv)
        out[i, 1:-1] = process.rho * process.xi * v_grid[1:-1] * s * cross
    return out


def _solve_spot_direction(rhs, s_grid, v_grid, resolved, *, theta_dt: float, tau: float):
    solved = rhs.copy()
    ds = s_grid[1] - s_grid[0]
    r = float(resolved.rate)
    for j, variance in enumerate(v_grid):
        n_int = len(s_grid) - 2
        if n_int <= 0:
            continue
        a = raw_np.empty(n_int)
        b = raw_np.empty(n_int)
        c = raw_np.empty(n_int)
        for row, i in enumerate(range(1, len(s_grid) - 1)):
            s = s_grid[i]
            diff = 0.5 * variance * s * s / (ds * ds)
            drift = 0.5 * r * s / ds
            a[row] = diff - drift
            b[row] = -2.0 * diff - r
            c[row] = diff + drift
        lower, upper = _spot_boundary_values(s_grid, tau, resolved)
        interior_rhs = rhs[1:-1, j].copy()
        interior_rhs[0] += theta_dt * a[0] * lower
        interior_rhs[-1] += theta_dt * c[-1] * upper
        diag = 1.0 - theta_dt * b
        lower_diag = -theta_dt * a[1:]
        upper_diag = -theta_dt * c[:-1]
        solved[1:-1, j] = thomas_solve(lower_diag, diag, upper_diag, interior_rhs)
        solved[0, j] = lower
        solved[-1, j] = upper
    return solved


def _solve_variance_direction(rhs, v_grid, process, *, theta_dt: float):
    solved = rhs.copy()
    dv = v_grid[1] - v_grid[0]
    n_int = len(v_grid) - 2
    if n_int <= 0:
        return solved
    for i in range(rhs.shape[0]):
        a = raw_np.empty(n_int)
        b = raw_np.empty(n_int)
        c = raw_np.empty(n_int)
        for row, j in enumerate(range(1, len(v_grid) - 1)):
            v = v_grid[j]
            diff = 0.5 * process.xi * process.xi * v / (dv * dv)
            drift = 0.5 * process.kappa * (process.theta - v) / dv
            a[row] = diff - drift
            b[row] = -2.0 * diff
            c[row] = diff + drift
        interior_rhs = rhs[i, 1:-1].copy()
        interior_rhs[0] += theta_dt * a[0] * rhs[i, 0]
        interior_rhs[-1] += theta_dt * c[-1] * rhs[i, -1]
        diag = 1.0 - theta_dt * b
        lower_diag = -theta_dt * a[1:]
        upper_diag = -theta_dt * c[:-1]
        solved[i, 1:-1] = thomas_solve(lower_diag, diag, upper_diag, interior_rhs)
        solved[i, 0] = solved[i, 1]
        solved[i, -1] = solved[i, -2]
    return solved


def _apply_spot_boundaries(values, s_grid, *, tau: float, resolved) -> None:
    lower, upper = _spot_boundary_values(s_grid, tau, resolved)
    values[0, :] = lower
    values[-1, :] = upper


def _spot_boundary_values(s_grid, tau: float, resolved) -> tuple[float, float]:
    discount_strike = resolved.strike * raw_np.exp(-resolved.rate * max(float(tau), 0.0))
    if resolved.option_type == "put":
        return float(discount_strike), 0.0
    return 0.0, float(s_grid[-1] - discount_strike)


def _resolve_dividend_yield(spec) -> float:
    for attr in ("dividend_yield", "continuous_dividend_yield", "q"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _resolve_maturity(settlement: date, spec, day_count) -> float:
    """Resolve maturity from common date aliases or a direct year fraction."""
    for attr in (
        "expiry_date",
        "maturity_date",
        "expiration_date",
        "exercise_date",
        "expiry",
        "maturity",
        "expiry_years",
        "maturity_years",
        "time_to_maturity",
        "tenor_years",
    ):
        value = getattr(spec, attr, None)
        if value is None:
            continue
        if isinstance(value, date):
            return max(float(year_fraction(settlement, value, day_count)), 0.0)
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value.strip())
            except ValueError:
                parsed = None
            if parsed is not None:
                return max(float(year_fraction(settlement, parsed, day_count)), 0.0)
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            continue
    raise ValueError(
        "spec must provide expiry_date, maturity_date, or numeric maturity for Heston ADI PDE pricing"
    )


def _resolve_spot(market_state, spec) -> float:
    """Resolve the underlier spot from spec aliases or the market state."""
    for attr in ("spot", "underlier_spot", "s0"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)
    value = getattr(market_state, "spot", None)
    if value is not None:
        return float(value)
    underlier_spots = getattr(market_state, "underlier_spots", None)
    if isinstance(underlier_spots, dict) and len(underlier_spots) == 1:
        return float(next(iter(underlier_spots.values())))
    raise ValueError("spec or market_state must provide spot for Heston ADI PDE pricing")


def _resolve_strike(spec, *, spot: float) -> float:
    """Resolve a scalar strike, selecting a representative ATM/grid strike when needed."""
    for attr in ("strike", "strike_price", "k"):
        value = getattr(spec, attr, None)
        if value is not None:
            return float(value)

    for attr in ("surface_strikes", "strike_grid", "strikes"):
        values = getattr(spec, attr, None)
        if not values:
            continue
        numeric = []
        for value in _iter_alias_values(values):
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if numeric:
            return min(numeric, key=lambda strike: abs(strike - spot))
    raise ValueError("spec must provide strike or a numeric strike grid for Heston ADI PDE pricing")


def _iter_alias_values(values) -> tuple[object, ...]:
    """Return scalar or iterable alias payloads as a normalized tuple."""
    if values is None:
        return ()
    if isinstance(values, str):
        pieces = tuple(piece.strip() for piece in values.split(",") if piece.strip())
        return pieces or (values,)
    try:
        iterator = iter(values)
    except TypeError:
        return (values,)
    return tuple(iterator)


def _resolve_spec_heston_parameters(spec) -> dict[str, float]:
    """Return explicit Heston parameters declared on a task spec, if present."""
    aliases = {
        "kappa": ("kappa", "mean_reversion"),
        "theta": ("theta", "long_run_variance", "long_variance", "theta_var"),
        "xi": ("xi", "vol_of_vol", "variance_vol", "sigma_v"),
        "rho": ("rho", "correlation"),
        "v0": ("v0", "initial_variance", "initial_var", "variance0"),
    }
    resolved: dict[str, float] = {}
    for canonical, names in aliases.items():
        for name in names:
            value = getattr(spec, name, None)
            if value is None:
                continue
            resolved[canonical] = float(value)
            break
    return resolved


__all__ = [
    "HestonAdiPDEConfig",
    "HestonAdiPDEResult",
    "ResolvedHestonAdiPDEInputs",
    "price_heston_option_adi_pde_result",
    "resolve_heston_adi_pde_inputs",
]
