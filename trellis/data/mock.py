"""Mock data provider with embedded historical yield snapshots.

No network, no API keys, no disk I/O.  Ships with the core package
so ``pip install trellis`` is immediately productive.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from trellis.curves.credit_curve import CreditCurve
from trellis.core.state_space import StateSpace
from trellis.curves.yield_curve import YieldCurve
from trellis.core.differentiable import get_numpy
from trellis.data.base import BaseDataProvider
from trellis.data.schema import MarketSnapshot
from trellis.instruments.fx import FXRate
from trellis.models.vol_surface import FlatVol, GridVolSurface

np = get_numpy()

# ---------------------------------------------------------------------------
# Embedded snapshots: {date: {tenor_years: yield_decimal}}
#
# Tenors match the 11-point grid used by FRED and Treasury.gov providers:
#   1mo, 3mo, 6mo, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y
#
# Yields are in decimal (0.045 = 4.5%), semi-annual BEY convention
# (same as what the real providers return).
# ---------------------------------------------------------------------------

_TENOR_GRID = (1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)

SNAPSHOTS: dict[date, dict[float, float]] = {
    # Pre-COVID normal curve — modest steepening, ~1.6-2.1%
    date(2019, 9, 15): dict(zip(_TENOR_GRID, [
        0.0193, 0.0188, 0.0190, 0.0175, 0.0163, 0.0157,
        0.0155, 0.0163, 0.0172, 0.0195, 0.0210,
    ])),

    # COVID crisis — near-zero front end, mild steepening
    date(2020, 3, 15): dict(zip(_TENOR_GRID, [
        0.0008, 0.0022, 0.0033, 0.0026, 0.0025, 0.0032,
        0.0037, 0.0052, 0.0073, 0.0112, 0.0129,
    ])),

    # Peak rates, inverted curve — 5.3% front end, 4.6-4.8% long end
    date(2023, 10, 15): dict(zip(_TENOR_GRID, [
        0.0533, 0.0530, 0.0527, 0.0507, 0.0500, 0.0487,
        0.0469, 0.0470, 0.0473, 0.0509, 0.0495,
    ])),

    # Easing cycle begins — moderate curve, ~4.2-4.6%
    date(2024, 11, 15): dict(zip(_TENOR_GRID, [
        0.0455, 0.0447, 0.0435, 0.0420, 0.0415, 0.0418,
        0.0425, 0.0432, 0.0438, 0.0462, 0.0458,
    ])),
}

# Sorted for binary-ish lookup
_SORTED_DATES = sorted(SNAPSHOTS.keys())


def _latest_snapshot_date_not_after(
    dates: list[date],
    as_of: date | None,
) -> date | None:
    """Return the latest snapshot date not after *as_of*."""
    if not dates:
        return None
    if as_of is None:
        return dates[-1]

    best = None
    for d in dates:
        if d <= as_of:
            best = d
        else:
            break
    return best


def _regime_for_snapshot_date(snapshot_date: date) -> str:
    """Map a built-in snapshot date to the qualitative market regime label."""
    if snapshot_date == date(2020, 3, 15):
        return "covid_crisis"
    if snapshot_date == date(2023, 10, 15):
        return "peak_inversion"
    if snapshot_date == date(2024, 11, 15):
        return "easing_cycle"
    return "normal"


def _shifted_curve(curve: YieldCurve, bps: float) -> YieldCurve:
    """Return a parallel-shifted copy of ``curve`` in basis points."""
    return curve.shift(bps)


def _build_rate_vol_surfaces(
    curve: YieldCurve,
    regime: str,
) -> dict[str, object]:
    """Synthesize mock ATM and smile rate-vol surfaces for the chosen regime."""
    base_levels = {
        "covid_crisis": (0.62, 0.54, 0.46, 0.38, 0.32, 0.28),
        "peak_inversion": (0.44, 0.38, 0.33, 0.29, 0.25, 0.22),
        "easing_cycle": (0.30, 0.27, 0.24, 0.22, 0.20, 0.18),
        "normal": (0.26, 0.24, 0.22, 0.20, 0.18, 0.17),
    }
    expiries = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    strikes = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
    atm_term = base_levels[regime]
    center = max(float(curve.zero_rate(2.0)), 0.01)

    smile_rows = []
    for expiry_index, atm in enumerate(atm_term):
        row = []
        expiry_scale = 1.0 - 0.04 * min(expiry_index, 4)
        for strike in strikes:
            moneyness = strike - center
            smile = 1.0 + 1.8 * abs(moneyness) + 0.9 * max(center - strike, 0.0)
            row.append(round(atm * expiry_scale * smile, 6))
        smile_rows.append(tuple(row))

    return {
        "usd_rates_atm": FlatVol(atm_term[2]),
        "usd_rates_smile": GridVolSurface(
            expiries=expiries,
            strikes=strikes,
            vols=tuple(smile_rows),
        ),
    }


def _build_credit_curves(regime: str) -> dict[str, CreditCurve]:
    """Build simple regime-dependent IG and HY credit curves from spread grids."""
    spreads = {
        "covid_crisis": (
            {1.0: 0.0120, 3.0: 0.0160, 5.0: 0.0180, 10.0: 0.0200},
            {1.0: 0.0450, 3.0: 0.0550, 5.0: 0.0600, 10.0: 0.0650},
        ),
        "peak_inversion": (
            {1.0: 0.0080, 3.0: 0.0100, 5.0: 0.0110, 10.0: 0.0120},
            {1.0: 0.0350, 3.0: 0.0400, 5.0: 0.0430, 10.0: 0.0460},
        ),
        "easing_cycle": (
            {1.0: 0.0060, 3.0: 0.0070, 5.0: 0.0080, 10.0: 0.0090},
            {1.0: 0.0280, 3.0: 0.0320, 5.0: 0.0350, 10.0: 0.0380},
        ),
        "normal": (
            {1.0: 0.0050, 3.0: 0.0060, 5.0: 0.0070, 10.0: 0.0080},
            {1.0: 0.0250, 3.0: 0.0300, 5.0: 0.0330, 10.0: 0.0360},
        ),
    }
    ig_spreads, hy_spreads = spreads[regime]
    return {
        "usd_ig": CreditCurve.from_spreads(ig_spreads, recovery=0.4),
        "usd_hy": CreditCurve.from_spreads(hy_spreads, recovery=0.4),
    }


def _build_fx_rates(regime: str) -> dict[str, FXRate]:
    """Return regime-specific spot FX levels for a small built-in currency set."""
    spots = {
        "covid_crisis": {"EURUSD": 1.095, "GBPUSD": 1.245, "USDJPY": 107.5},
        "peak_inversion": {"EURUSD": 1.060, "GBPUSD": 1.220, "USDJPY": 149.0},
        "easing_cycle": {"EURUSD": 1.085, "GBPUSD": 1.270, "USDJPY": 151.5},
        "normal": {"EURUSD": 1.110, "GBPUSD": 1.280, "USDJPY": 108.0},
    }
    spot = spots[regime]
    return {
        "EURUSD": FXRate(spot=spot["EURUSD"], domestic="USD", foreign="EUR"),
        "GBPUSD": FXRate(spot=spot["GBPUSD"], domestic="USD", foreign="GBP"),
        "USDJPY": FXRate(spot=spot["USDJPY"], domestic="JPY", foreign="USD"),
    }


def _build_underlier_spots(regime: str) -> dict[str, float]:
    """Return regime-specific equity underlier spot levels."""
    spots = {
        "covid_crisis": {"SPX": 2584.0, "AAPL": 69.5, "MSFT": 141.0},
        "peak_inversion": {"SPX": 4237.0, "AAPL": 178.0, "MSFT": 332.0},
        "easing_cycle": {"SPX": 5890.0, "AAPL": 228.0, "MSFT": 419.0},
        "normal": {"SPX": 3008.0, "AAPL": 55.0, "MSFT": 137.0},
    }
    return dict(spots[regime])


def _make_local_vol_surface(
    *,
    ref_spot: float,
    base_sigma: float,
    skew: float,
    term_slope: float,
):
    """Return a smooth local-vol stand-in compatible with scalar/array inputs."""

    def local_vol(spot, time):
        """Return a smooth smile/term adjusted local volatility for spot/time inputs."""
        clipped_spot = np.maximum(spot, 1e-8)
        clipped_time = np.maximum(time, 0.0)
        moneyness = np.abs(np.log(clipped_spot / ref_spot))
        term = 1.0 + term_slope * np.minimum(clipped_time, 5.0) / 5.0
        smile = 1.0 + skew * moneyness
        return base_sigma * term * smile

    return local_vol


def _build_local_vol_surfaces(
    *,
    regime: str,
    underlier_spots: dict[str, float],
) -> dict[str, object]:
    """Build callable local-vol stand-ins keyed by underlier name."""
    params = {
        "covid_crisis": {"base_sigma": 0.34, "skew": 0.22, "term_slope": 0.08},
        "peak_inversion": {"base_sigma": 0.24, "skew": 0.15, "term_slope": 0.05},
        "easing_cycle": {"base_sigma": 0.20, "skew": 0.12, "term_slope": 0.04},
        "normal": {"base_sigma": 0.18, "skew": 0.10, "term_slope": 0.03},
    }[regime]
    return {
        "spx_local_vol": _make_local_vol_surface(
            ref_spot=underlier_spots["SPX"],
            **params,
        ),
        "aapl_local_vol": _make_local_vol_surface(
            ref_spot=underlier_spots["AAPL"],
            base_sigma=params["base_sigma"] * 1.15,
            skew=params["skew"] * 1.05,
            term_slope=params["term_slope"],
        ),
    }


def _build_jump_parameter_sets(regime: str) -> dict[str, dict[str, float]]:
    """Return representative Merton jump-diffusion parameter sets for the regime."""
    params = {
        "covid_crisis": {
            "mu": 0.0,
            "sigma": 0.32,
            "lam": 0.65,
            "jump_mean": -0.11,
            "jump_vol": 0.28,
        },
        "peak_inversion": {
            "mu": 0.0,
            "sigma": 0.24,
            "lam": 0.35,
            "jump_mean": -0.07,
            "jump_vol": 0.20,
        },
        "easing_cycle": {
            "mu": 0.0,
            "sigma": 0.20,
            "lam": 0.22,
            "jump_mean": -0.05,
            "jump_vol": 0.16,
        },
        "normal": {
            "mu": 0.0,
            "sigma": 0.18,
            "lam": 0.18,
            "jump_mean": -0.04,
            "jump_vol": 0.14,
        },
    }
    return {"merton_equity": dict(params[regime])}


def _build_model_parameter_sets(regime: str) -> dict[str, dict[str, float]]:
    """Return representative stochastic-volatility model parameters for the regime."""
    params = {
        "covid_crisis": {
            "mu": 0.0,
            "kappa": 1.25,
            "theta": 0.085,
            "xi": 0.95,
            "rho": -0.78,
            "v0": 0.10,
        },
        "peak_inversion": {
            "mu": 0.0,
            "kappa": 1.75,
            "theta": 0.055,
            "xi": 0.62,
            "rho": -0.72,
            "v0": 0.06,
        },
        "easing_cycle": {
            "mu": 0.0,
            "kappa": 2.00,
            "theta": 0.040,
            "xi": 0.48,
            "rho": -0.68,
            "v0": 0.04,
        },
        "normal": {
            "mu": 0.0,
            "kappa": 2.10,
            "theta": 0.032,
            "xi": 0.42,
            "rho": -0.60,
            "v0": 0.032,
        },
    }
    return {"heston_equity": dict(params[regime])}


def _scale_vol_surface(surface, scale: float):
    """Return a simple scaled vol-surface copy for scenario state spaces."""
    if isinstance(surface, FlatVol):
        return FlatVol(surface.vol * scale)
    if isinstance(surface, GridVolSurface):
        return GridVolSurface(
            expiries=surface.expiries,
            strikes=surface.strikes,
            vols=tuple(
                tuple(round(vol * scale, 6) for vol in row)
                for row in surface.vols
            ),
        )
    return surface


def _scale_fx_rates(fx_rates: dict[str, FXRate] | None, scale: float) -> dict[str, FXRate] | None:
    """Return scaled FX spot quotes for a scenario state."""
    if not fx_rates:
        return fx_rates
    return {
        pair: FXRate(
            spot=quote.spot * scale,
            domestic=quote.domestic,
            foreign=quote.foreign,
        )
        for pair, quote in fx_rates.items()
    }


def _build_state_spaces(regime: str) -> dict[str, object]:
    """Build bounded scenario state spaces off the compiled base market state."""
    scenario_scales = {
        "covid_crisis": {
            "bull_spot": 1.04,
            "bull_fx": 1.015,
            "bull_vol": 0.95,
            "bull_rates_bps": -20.0,
            "stress_spot": 0.90,
            "stress_fx": 0.96,
            "stress_vol": 1.12,
            "stress_rates_bps": 35.0,
            "stress_credit_bps": 80.0,
        },
        "peak_inversion": {
            "bull_spot": 1.05,
            "bull_fx": 1.020,
            "bull_vol": 0.94,
            "bull_rates_bps": -18.0,
            "stress_spot": 0.91,
            "stress_fx": 0.97,
            "stress_vol": 1.10,
            "stress_rates_bps": 28.0,
            "stress_credit_bps": 60.0,
        },
        "easing_cycle": {
            "bull_spot": 1.06,
            "bull_fx": 1.018,
            "bull_vol": 0.92,
            "bull_rates_bps": -15.0,
            "stress_spot": 0.92,
            "stress_fx": 0.975,
            "stress_vol": 1.08,
            "stress_rates_bps": 22.0,
            "stress_credit_bps": 45.0,
        },
        "normal": {
            "bull_spot": 1.05,
            "bull_fx": 1.015,
            "bull_vol": 0.93,
            "bull_rates_bps": -12.0,
            "stress_spot": 0.93,
            "stress_fx": 0.98,
            "stress_vol": 1.07,
            "stress_rates_bps": 18.0,
            "stress_credit_bps": 35.0,
        },
    }[regime]

    def macro_regime(base_state, snapshot, settlement):
        """Macro regime scenario weights over shifted conditional market states."""
        bull_forecasts = (
            {name: curve.shift(scenario_scales["bull_rates_bps"]) for name, curve in base_state.forecast_curves.items()}
            if base_state.forecast_curves else None
        )
        stress_forecasts = (
            {name: curve.shift(scenario_scales["stress_rates_bps"]) for name, curve in base_state.forecast_curves.items()}
            if base_state.forecast_curves else None
        )
        bull_underliers = (
            {name: spot * scenario_scales["bull_spot"] for name, spot in base_state.underlier_spots.items()}
            if base_state.underlier_spots else None
        )
        stress_underliers = (
            {name: spot * scenario_scales["stress_spot"] for name, spot in base_state.underlier_spots.items()}
            if base_state.underlier_spots else None
        )
        bull_state = replace(
            base_state,
            discount=base_state.discount.shift(scenario_scales["bull_rates_bps"]) if base_state.discount else None,
            vol_surface=_scale_vol_surface(base_state.vol_surface, scenario_scales["bull_vol"])
            if base_state.vol_surface is not None else None,
            credit_curve=base_state.credit_curve.shift(-15.0) if base_state.credit_curve else None,
            forecast_curves=bull_forecasts,
            fx_rates=_scale_fx_rates(base_state.fx_rates, scenario_scales["bull_fx"]),
            spot=(base_state.spot * scenario_scales["bull_spot"]) if base_state.spot is not None else None,
            underlier_spots=bull_underliers,
            state_space=None,
        )
        stress_state = replace(
            base_state,
            discount=base_state.discount.shift(scenario_scales["stress_rates_bps"]) if base_state.discount else None,
            vol_surface=_scale_vol_surface(base_state.vol_surface, scenario_scales["stress_vol"])
            if base_state.vol_surface is not None else None,
            credit_curve=base_state.credit_curve.shift(scenario_scales["stress_credit_bps"]) if base_state.credit_curve else None,
            forecast_curves=stress_forecasts,
            fx_rates=_scale_fx_rates(base_state.fx_rates, scenario_scales["stress_fx"]),
            spot=(base_state.spot * scenario_scales["stress_spot"]) if base_state.spot is not None else None,
            underlier_spots=stress_underliers,
            state_space=None,
        )
        return StateSpace(
            states={
                "base": (0.50, replace(base_state, state_space=None)),
                "bull_repricing": (0.25, bull_state),
                "stress_repricing": (0.25, stress_state),
            }
        )

    return {"macro_regime": macro_regime}


class MockDataProvider(BaseDataProvider):
    """In-memory data provider with embedded historical yield snapshots.

    Parameters
    ----------
    overrides : dict or None
        Additional ``{date: {tenor: yield}}`` entries that supplement
        (or shadow) the built-in snapshots.
    """

    def __init__(self, overrides: dict[date, dict[float, float]] | None = None):
        """Seed the provider with embedded snapshots plus optional caller overrides."""
        self._data = dict(SNAPSHOTS)
        if overrides:
            self._data.update(overrides)
        self._sorted_dates = sorted(self._data.keys())

    @classmethod
    def from_dict(cls, data: dict[date, dict[float, float]]) -> MockDataProvider:
        """Create a provider with *only* user-supplied data (no built-in snapshots)."""
        inst = cls.__new__(cls)
        inst._data = dict(data)
        inst._sorted_dates = sorted(inst._data.keys())
        return inst

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Return the snapshot closest to (but not after) *as_of*.

        If *as_of* is ``None``, returns the most recent snapshot.
        If *as_of* is before all snapshots, returns an empty dict.
        """
        best = _latest_snapshot_date_not_after(self._sorted_dates, as_of)
        if best is None:
            return {}
        return dict(self._data[best])

    def fetch_market_snapshot(self, as_of: date | None = None) -> MarketSnapshot:
        """Return a simulated full market snapshot keyed off the yield regime."""
        best = _latest_snapshot_date_not_after(self._sorted_dates, as_of)
        if best is None:
            requested_date = as_of or date.today()
            return MarketSnapshot(as_of=requested_date, source="mock")

        yields = self._data[best]
        regime = _regime_for_snapshot_date(best)
        usd_ois = YieldCurve.from_treasury_yields(yields)
        eur_ois = _shifted_curve(usd_ois, -175)
        gbp_ois = _shifted_curve(usd_ois, -75)

        forecast_curves = {
            "USD-SOFR-3M": _shifted_curve(usd_ois, 35),
            "USD-LIBOR-3M": _shifted_curve(usd_ois, 55),
            "USD-DISC": usd_ois,
            "EUR-DISC": eur_ois,
            "GBP-DISC": gbp_ois,
            "EUR-EURIBOR-3M": _shifted_curve(eur_ois, 30),
            "GBP-SONIA-3M": _shifted_curve(gbp_ois, 18),
        }
        underlier_spots = _build_underlier_spots(regime)
        metadata = {
            "simulated": True,
            "regime": regime,
            "requested_as_of": as_of.isoformat() if as_of is not None else None,
            "snapshot_date": best.isoformat(),
            "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
        }

        return MarketSnapshot(
            as_of=best,
            source="mock",
            discount_curves={
                "usd_ois": usd_ois,
                "eur_ois": eur_ois,
                "gbp_ois": gbp_ois,
            },
            forecast_curves=forecast_curves,
            vol_surfaces=_build_rate_vol_surfaces(usd_ois, regime),
            credit_curves=_build_credit_curves(regime),
            fx_rates=_build_fx_rates(regime),
            state_spaces=_build_state_spaces(regime),
            underlier_spots=underlier_spots,
            local_vol_surfaces=_build_local_vol_surfaces(
                regime=regime,
                underlier_spots=underlier_spots,
            ),
            jump_parameter_sets=_build_jump_parameter_sets(regime),
            model_parameter_sets=_build_model_parameter_sets(regime),
            metadata=metadata,
            default_discount_curve="usd_ois",
            default_vol_surface="usd_rates_smile",
            default_credit_curve="usd_ig",
            default_state_space="macro_regime",
            default_underlier_spot="SPX",
            default_local_vol_surface="spx_local_vol",
            default_jump_parameters="merton_equity",
            default_model_parameters="heston_equity",
        )

    @property
    def available_dates(self) -> list[date]:
        """List of snapshot dates available in this provider."""
        return list(self._sorted_dates)
