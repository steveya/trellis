"""Mock data provider with embedded historical yield snapshots.

No network, no API keys, no disk I/O.  Ships with the core package
so ``pip install trellis`` is immediately productive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
from types import MappingProxyType

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


def _freeze_payload(value):
    """Return a recursively frozen payload made of mappings and tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_payload(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_payload(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _payload_to_json(value):
    """Return a JSON-friendly thawed copy of a frozen payload."""
    if isinstance(value, Mapping):
        return {str(key): _payload_to_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_payload_to_json(item) for item in value]
    return value


def _stable_unit_interval(seed: int, label: str) -> float:
    """Return a deterministic value in ``[0, 1]`` for ``seed`` and ``label``."""
    payload = f"{int(seed)}|{label}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float((1 << 64) - 1)


def _seeded_scale(seed: int, label: str, *, amplitude: float) -> float:
    """Return a bounded deterministic multiplicative scale around ``1.0``."""
    centered = 2.0 * _stable_unit_interval(seed, label) - 1.0
    return 1.0 + centered * float(amplitude)


@dataclass(frozen=True)
class SyntheticRatesModelPack:
    """Seeded rates-side authority inputs for one synthetic snapshot."""

    family: str
    anchor_curve_set: str
    curve_roles: Mapping[str, str]
    discount_curve_shifts_bps: Mapping[str, float]
    forecast_basis_bps: Mapping[str, float]
    rate_vol_levels: tuple[float, ...]
    rate_vol_surface_family: str = "regime_rate_vol_surface"
    quote_families: tuple[str, ...] = ("price", "implied_vol")

    def __post_init__(self) -> None:
        object.__setattr__(self, "curve_roles", _freeze_payload(self.curve_roles))
        object.__setattr__(self, "discount_curve_shifts_bps", _freeze_payload(self.discount_curve_shifts_bps))
        object.__setattr__(self, "forecast_basis_bps", _freeze_payload(self.forecast_basis_bps))
        object.__setattr__(self, "rate_vol_levels", tuple(float(level) for level in self.rate_vol_levels))
        object.__setattr__(self, "quote_families", tuple(str(family) for family in self.quote_families))

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "anchor_curve_set": self.anchor_curve_set,
            "curve_roles": _payload_to_json(self.curve_roles),
            "discount_curve_shifts_bps": _payload_to_json(self.discount_curve_shifts_bps),
            "forecast_basis_bps": _payload_to_json(self.forecast_basis_bps),
            "rate_vol_levels": _payload_to_json(self.rate_vol_levels),
            "rate_vol_surface_family": self.rate_vol_surface_family,
            "quote_families": _payload_to_json(self.quote_families),
        }


@dataclass(frozen=True)
class SyntheticCreditModelPack:
    """Seeded credit-side authority inputs for one synthetic snapshot."""

    family: str
    recovery: float
    spread_inputs_decimal: Mapping[str, Mapping[str, float]]
    quote_families: tuple[str, ...] = ("spread", "hazard")

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery", float(self.recovery))
        object.__setattr__(self, "spread_inputs_decimal", _freeze_payload(self.spread_inputs_decimal))
        object.__setattr__(self, "quote_families", tuple(str(family) for family in self.quote_families))

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "recovery": self.recovery,
            "spread_inputs_decimal": _payload_to_json(self.spread_inputs_decimal),
            "quote_families": _payload_to_json(self.quote_families),
        }


@dataclass(frozen=True)
class SyntheticVolatilityModelPack:
    """Seeded volatility-side authority inputs for one synthetic snapshot."""

    family: str
    local_vol_surface_family: str
    jump_parameter_family: str
    model_parameter_family: str
    local_vol_parameters: Mapping[str, Mapping[str, float]]
    jump_parameter_sets: Mapping[str, Mapping[str, float]]
    model_parameter_sets: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_vol_parameters", _freeze_payload(self.local_vol_parameters))
        object.__setattr__(self, "jump_parameter_sets", _freeze_payload(self.jump_parameter_sets))
        object.__setattr__(self, "model_parameter_sets", _freeze_payload(self.model_parameter_sets))

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "local_vol_surface_family": self.local_vol_surface_family,
            "jump_parameter_family": self.jump_parameter_family,
            "model_parameter_family": self.model_parameter_family,
            "local_vol_parameters": _payload_to_json(self.local_vol_parameters),
            "jump_parameter_sets": _payload_to_json(self.jump_parameter_sets),
            "model_parameter_sets": _payload_to_json(self.model_parameter_sets),
        }


@dataclass(frozen=True)
class SyntheticQuoteBundles:
    """Serializable synthetic quote bundles derived from the model packs."""

    rates: Mapping[str, object]
    credit: Mapping[str, object]
    volatility: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rates", _freeze_payload(self.rates))
        object.__setattr__(self, "credit", _freeze_payload(self.credit))
        object.__setattr__(self, "volatility", _freeze_payload(self.volatility))

    def to_payload(self) -> dict[str, object]:
        return {
            "rates": _payload_to_json(self.rates),
            "credit": _payload_to_json(self.credit),
            "volatility": _payload_to_json(self.volatility),
        }


@dataclass(frozen=True)
class SyntheticRuntimeTargets:
    """Names of runtime-facing artifacts built from one synthetic contract."""

    discount_curves: tuple[str, ...]
    forecast_curves: tuple[str, ...]
    credit_curves: tuple[str, ...]
    vol_surfaces: tuple[str, ...]
    local_vol_surfaces: tuple[str, ...]
    jump_parameter_sets: tuple[str, ...]
    model_parameter_sets: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "discount_curves", tuple(str(name) for name in self.discount_curves))
        object.__setattr__(self, "forecast_curves", tuple(str(name) for name in self.forecast_curves))
        object.__setattr__(self, "credit_curves", tuple(str(name) for name in self.credit_curves))
        object.__setattr__(self, "vol_surfaces", tuple(str(name) for name in self.vol_surfaces))
        object.__setattr__(self, "local_vol_surfaces", tuple(str(name) for name in self.local_vol_surfaces))
        object.__setattr__(self, "jump_parameter_sets", tuple(str(name) for name in self.jump_parameter_sets))
        object.__setattr__(self, "model_parameter_sets", tuple(str(name) for name in self.model_parameter_sets))

    def to_payload(self) -> dict[str, object]:
        return {
            "discount_curves": _payload_to_json(self.discount_curves),
            "forecast_curves": _payload_to_json(self.forecast_curves),
            "credit_curves": _payload_to_json(self.credit_curves),
            "vol_surfaces": _payload_to_json(self.vol_surfaces),
            "local_vol_surfaces": _payload_to_json(self.local_vol_surfaces),
            "jump_parameter_sets": _payload_to_json(self.jump_parameter_sets),
            "model_parameter_sets": _payload_to_json(self.model_parameter_sets),
        }


@dataclass(frozen=True)
class SyntheticGenerationContract:
    """Seeded authority contract for the synthetic mock market path."""

    version: str
    seed: int
    source_kind: str
    regime: str
    snapshot_date: str
    requested_as_of: str | None
    rates: SyntheticRatesModelPack
    credit: SyntheticCreditModelPack
    volatility: SyntheticVolatilityModelPack
    quote_bundles: SyntheticQuoteBundles
    runtime_targets: SyntheticRuntimeTargets

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "source_kind": self.source_kind,
            "regime": self.regime,
            "snapshot_date": self.snapshot_date,
            "requested_as_of": self.requested_as_of,
            "model_packs": {
                "rates": self.rates.to_payload(),
                "credit": self.credit.to_payload(),
                "volatility": self.volatility.to_payload(),
            },
            "quote_bundles": self.quote_bundles.to_payload(),
            "runtime_targets": self.runtime_targets.to_payload(),
        }


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


def _build_synthetic_rates_model_pack(
    *,
    regime: str,
    seed: int,
) -> SyntheticRatesModelPack:
    """Return the seeded rates model pack for one synthetic snapshot."""
    base_levels = {
        "covid_crisis": (0.62, 0.54, 0.46, 0.38, 0.32, 0.28),
        "peak_inversion": (0.44, 0.38, 0.33, 0.29, 0.25, 0.22),
        "easing_cycle": (0.30, 0.27, 0.24, 0.22, 0.20, 0.18),
        "normal": (0.26, 0.24, 0.22, 0.20, 0.18, 0.17),
    }[regime]
    rate_vol_scale = _seeded_scale(seed, f"rates_vol_levels::{regime}", amplitude=0.035)
    return SyntheticRatesModelPack(
        family="shifted_curve_bundle",
        anchor_curve_set="embedded_treasury_yield_regime",
        curve_roles={
            "discount_curve": "usd_ois",
            "forecast_curve": "USD-SOFR-3M",
        },
        discount_curve_shifts_bps={
            "eur_ois": -175.0,
            "gbp_ois": -75.0,
        },
        forecast_basis_bps={
            "USD-SOFR-3M": 35.0,
            "USD-LIBOR-3M": 55.0,
            "EUR-EURIBOR-3M": 30.0,
            "GBP-SONIA-3M": 18.0,
        },
        rate_vol_levels=tuple(round(level * rate_vol_scale, 6) for level in base_levels),
    )


def _build_discount_curves(
    anchor_curve: YieldCurve,
    rates_pack: SyntheticRatesModelPack,
) -> dict[str, YieldCurve]:
    """Build named discount curves from the seeded rates model pack."""
    curves = {"usd_ois": anchor_curve}
    for curve_name, shift_bps in rates_pack.discount_curve_shifts_bps.items():
        curves[str(curve_name)] = _shifted_curve(anchor_curve, float(shift_bps))
    return curves


def _build_forecast_curves(
    discount_curves: Mapping[str, YieldCurve],
    rates_pack: SyntheticRatesModelPack,
) -> dict[str, YieldCurve]:
    """Build named forecast curves from the seeded rates model pack."""
    usd_ois = discount_curves["usd_ois"]
    eur_ois = discount_curves["eur_ois"]
    gbp_ois = discount_curves["gbp_ois"]
    return {
        "USD-SOFR-3M": _shifted_curve(usd_ois, float(rates_pack.forecast_basis_bps["USD-SOFR-3M"])),
        "USD-LIBOR-3M": _shifted_curve(usd_ois, float(rates_pack.forecast_basis_bps["USD-LIBOR-3M"])),
        "USD-DISC": usd_ois,
        "EUR-DISC": eur_ois,
        "GBP-DISC": gbp_ois,
        "EUR-EURIBOR-3M": _shifted_curve(eur_ois, float(rates_pack.forecast_basis_bps["EUR-EURIBOR-3M"])),
        "GBP-SONIA-3M": _shifted_curve(gbp_ois, float(rates_pack.forecast_basis_bps["GBP-SONIA-3M"])),
    }


def _build_rate_vol_surfaces(
    curve: YieldCurve,
    rates_pack: SyntheticRatesModelPack,
) -> dict[str, object]:
    """Synthesize mock ATM and smile rate-vol surfaces from the rates model pack."""
    expiries = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    strikes = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
    atm_term = rates_pack.rate_vol_levels
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


def _build_synthetic_credit_model_pack(
    *,
    regime: str,
    seed: int,
) -> SyntheticCreditModelPack:
    """Return the seeded credit model pack for one synthetic snapshot."""
    ig_spreads, hy_spreads, recovery = _credit_spread_inputs(regime)
    ig_scale = _seeded_scale(seed, f"credit_spreads::ig::{regime}", amplitude=0.04)
    hy_scale = _seeded_scale(seed, f"credit_spreads::hy::{regime}", amplitude=0.05)
    return SyntheticCreditModelPack(
        family="reduced_form_spread_grid",
        recovery=recovery,
        spread_inputs_decimal={
            "usd_ig": {
                str(tenor): round(float(spread) * ig_scale, 6)
                for tenor, spread in ig_spreads.items()
            },
            "usd_hy": {
                str(tenor): round(float(spread) * hy_scale, 6)
                for tenor, spread in hy_spreads.items()
            },
        },
    )


def _build_credit_curves(credit_pack: SyntheticCreditModelPack) -> dict[str, CreditCurve]:
    """Build named credit curves from the seeded credit model pack."""
    return {
        "usd_ig": CreditCurve.from_spreads(
            {float(tenor): float(spread) for tenor, spread in credit_pack.spread_inputs_decimal["usd_ig"].items()},
            recovery=credit_pack.recovery,
        ),
        "usd_hy": CreditCurve.from_spreads(
            {float(tenor): float(spread) for tenor, spread in credit_pack.spread_inputs_decimal["usd_hy"].items()},
            recovery=credit_pack.recovery,
        ),
    }


def _credit_spread_inputs(regime: str) -> tuple[dict[float, float], dict[float, float], float]:
    """Return deterministic spread grids and recovery used for mock credit curves."""
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
    return dict(ig_spreads), dict(hy_spreads), 0.4


def _build_model_consistency_contract(
    contract: SyntheticGenerationContract,
) -> dict[str, object]:
    """Return the legacy compatibility contract derived from the seeded authority."""
    credit_quotes = dict(contract.quote_bundles.credit["spread_inputs_decimal"])
    return {
        "version": "v1",
        "seed": int(contract.seed),
        "source_kind": contract.source_kind,
        "regime": contract.regime,
        "snapshot_date": contract.snapshot_date,
        "requested_as_of": contract.requested_as_of,
        "rates": {
            "workflow": "curve_shifted_from_discount",
            "curve_roles": _payload_to_json(contract.rates.curve_roles),
            "forecast_basis_bps": _payload_to_json(contract.quote_bundles.rates["forecast_basis_bps"]),
            "quote_families": _payload_to_json(contract.rates.quote_families),
            "materialization_targets": ("discount_curves", "forecast_curves"),
        },
        "credit": {
            "workflow": "calibrate_single_name_credit_curve_workflow",
            "quote_families": _payload_to_json(contract.credit.quote_families),
            "recovery": float(contract.credit.recovery),
            "spread_inputs_decimal": _payload_to_json(credit_quotes),
            "materialization_targets": ("credit_curve",),
        },
        "volatility": {
            "workflow": "calibration_surface_bundle",
            "quote_families": ("implied_vol",),
            "rate_vol_surfaces": _payload_to_json(contract.runtime_targets.vol_surfaces),
            "local_vol_surfaces": _payload_to_json(contract.runtime_targets.local_vol_surfaces),
            "model_parameter_sets": _payload_to_json(contract.volatility.model_parameter_sets),
            "materialization_targets": (
                "vol_surface",
                "local_vol_surface",
                "model_parameters",
            ),
        },
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


def _build_fixing_histories(snapshot_date: date, forecast_curves: dict[str, YieldCurve]) -> dict[str, dict[date, float]]:
    """Return deterministic recent fixing histories aligned to the mock rate regime."""
    base_sofr = float(forecast_curves["USD-SOFR-3M"].zero_rate(0.25))
    base_libor = float(forecast_curves["USD-LIBOR-3M"].zero_rate(0.25))
    base_euribor = float(forecast_curves["EUR-EURIBOR-3M"].zero_rate(0.25))
    base_sonia = float(forecast_curves["GBP-SONIA-3M"].zero_rate(0.25))

    def _history(base_rate: float, *, scale: float) -> dict[date, float]:
        return {
            snapshot_date - timedelta(days=offset): round(base_rate + scale * offset, 6)
            for offset in range(1, 6)
        }

    sofr_history = _history(base_sofr, scale=-0.00005)
    return {
        "USD-SOFR-3M": sofr_history,
        "SOFR": dict(sofr_history),
        "USD-LIBOR-3M": _history(base_libor, scale=-0.00004),
        "EUR-EURIBOR-3M": _history(base_euribor, scale=-0.00003),
        "GBP-SONIA-3M": _history(base_sonia, scale=-0.00002),
    }


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
    volatility_pack: SyntheticVolatilityModelPack,
    underlier_spots: dict[str, float],
) -> dict[str, object]:
    """Build callable local-vol stand-ins keyed by underlier name."""
    surfaces: dict[str, object] = {}
    for surface_name, params in volatility_pack.local_vol_parameters.items():
        underlier = "SPX" if surface_name.startswith("spx") else "AAPL"
        surfaces[str(surface_name)] = _make_local_vol_surface(
            ref_spot=underlier_spots[underlier],
            base_sigma=float(params["base_sigma"]),
            skew=float(params["skew"]),
            term_slope=float(params["term_slope"]),
        )
    return surfaces


def _build_synthetic_volatility_model_pack(
    *,
    regime: str,
    seed: int,
) -> SyntheticVolatilityModelPack:
    """Return the seeded volatility model pack for one synthetic snapshot."""
    local_base = {
        "covid_crisis": {"base_sigma": 0.34, "skew": 0.22, "term_slope": 0.08},
        "peak_inversion": {"base_sigma": 0.24, "skew": 0.15, "term_slope": 0.05},
        "easing_cycle": {"base_sigma": 0.20, "skew": 0.12, "term_slope": 0.04},
        "normal": {"base_sigma": 0.18, "skew": 0.10, "term_slope": 0.03},
    }[regime]
    local_scale = _seeded_scale(seed, f"local_vol::{regime}", amplitude=0.05)
    jump_params = {
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
    }[regime]
    jump_scale = _seeded_scale(seed, f"jump_pack::{regime}", amplitude=0.08)
    model_params = {
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
    }[regime]
    model_scale = _seeded_scale(seed, f"heston_pack::{regime}", amplitude=0.06)
    return SyntheticVolatilityModelPack(
        family="regime_surface_bundle",
        local_vol_surface_family="regime_local_vol_surface",
        jump_parameter_family="regime_jump_pack",
        model_parameter_family="regime_model_pack",
        local_vol_parameters={
            "spx_local_vol": {
                "base_sigma": round(local_base["base_sigma"] * local_scale, 6),
                "skew": round(local_base["skew"], 6),
                "term_slope": round(local_base["term_slope"], 6),
            },
            "aapl_local_vol": {
                "base_sigma": round(local_base["base_sigma"] * 1.15 * local_scale, 6),
                "skew": round(local_base["skew"] * 1.05, 6),
                "term_slope": round(local_base["term_slope"], 6),
            },
        },
        jump_parameter_sets={
            "merton_equity": {
                "mu": float(jump_params["mu"]),
                "sigma": round(float(jump_params["sigma"]) * jump_scale, 6),
                "lam": round(float(jump_params["lam"]) * jump_scale, 6),
                "jump_mean": float(jump_params["jump_mean"]),
                "jump_vol": round(float(jump_params["jump_vol"]) * jump_scale, 6),
            },
        },
        model_parameter_sets={
            "heston_equity": {
                "mu": float(model_params["mu"]),
                "kappa": round(float(model_params["kappa"]), 6),
                "theta": round(float(model_params["theta"]) * model_scale, 6),
                "xi": round(float(model_params["xi"]) * model_scale, 6),
                "rho": round(float(model_params["rho"]), 6),
                "v0": round(float(model_params["v0"]) * model_scale, 6),
            },
        },
    )


def _build_jump_parameter_sets(volatility_pack: SyntheticVolatilityModelPack) -> dict[str, dict[str, float]]:
    """Return representative jump-diffusion parameter sets from the volatility model pack."""
    return {
        str(name): {str(key): float(value) for key, value in params.items()}
        for name, params in volatility_pack.jump_parameter_sets.items()
    }


def _build_model_parameter_sets(volatility_pack: SyntheticVolatilityModelPack) -> dict[str, dict[str, float]]:
    """Return representative stochastic-volatility parameter sets from the volatility model pack."""
    return {
        str(name): {str(key): float(value) for key, value in params.items()}
        for name, params in volatility_pack.model_parameter_sets.items()
    }


def _build_synthetic_quote_bundles(
    *,
    rates_pack: SyntheticRatesModelPack,
    credit_pack: SyntheticCreditModelPack,
    volatility_pack: SyntheticVolatilityModelPack,
    rate_vol_surfaces: Mapping[str, object],
    local_vol_surfaces: Mapping[str, object],
) -> SyntheticQuoteBundles:
    """Return the synthetic quote bundles derived from the model packs."""
    return SyntheticQuoteBundles(
        rates={
            "quote_families": rates_pack.quote_families,
            "forecast_basis_bps": rates_pack.forecast_basis_bps,
            "rate_vol_surface_names": tuple(rate_vol_surfaces.keys()),
        },
        credit={
            "quote_families": credit_pack.quote_families,
            "spread_inputs_decimal": credit_pack.spread_inputs_decimal,
        },
        volatility={
            "surface_families": {
                "rate_vol_surface_family": rates_pack.rate_vol_surface_family,
                "local_vol_surface_family": volatility_pack.local_vol_surface_family,
                "jump_parameter_family": volatility_pack.jump_parameter_family,
                "model_parameter_family": volatility_pack.model_parameter_family,
            },
            "local_vol_surface_names": tuple(local_vol_surfaces.keys()),
            "jump_parameter_set_names": tuple(volatility_pack.jump_parameter_sets.keys()),
            "model_parameter_set_names": tuple(volatility_pack.model_parameter_sets.keys()),
        },
    )


def _build_synthetic_runtime_targets(
    *,
    discount_curves: Mapping[str, YieldCurve],
    forecast_curves: Mapping[str, YieldCurve],
    credit_curves: Mapping[str, CreditCurve],
    rate_vol_surfaces: Mapping[str, object],
    local_vol_surfaces: Mapping[str, object],
    jump_parameter_sets: Mapping[str, Mapping[str, float]],
    model_parameter_sets: Mapping[str, Mapping[str, float]],
) -> SyntheticRuntimeTargets:
    """Return the runtime-facing object names built from the synthetic contract."""
    return SyntheticRuntimeTargets(
        discount_curves=tuple(discount_curves.keys()),
        forecast_curves=tuple(forecast_curves.keys()),
        credit_curves=tuple(credit_curves.keys()),
        vol_surfaces=tuple(rate_vol_surfaces.keys()),
        local_vol_surfaces=tuple(local_vol_surfaces.keys()),
        jump_parameter_sets=tuple(jump_parameter_sets.keys()),
        model_parameter_sets=tuple(model_parameter_sets.keys()),
    )


def _build_synthetic_generation_contract(
    *,
    regime: str,
    seed: int,
    snapshot_date: date,
    requested_as_of: date | None,
    source_kind: str,
    rates_pack: SyntheticRatesModelPack,
    credit_pack: SyntheticCreditModelPack,
    volatility_pack: SyntheticVolatilityModelPack,
    quote_bundles: SyntheticQuoteBundles,
    runtime_targets: SyntheticRuntimeTargets,
) -> SyntheticGenerationContract:
    """Return the seeded synthetic-generation authority contract."""
    return SyntheticGenerationContract(
        version="v2",
        seed=seed,
        source_kind=source_kind,
        regime=regime,
        snapshot_date=snapshot_date.isoformat(),
        requested_as_of=requested_as_of.isoformat() if requested_as_of is not None else None,
        rates=rates_pack,
        credit=credit_pack,
        volatility=volatility_pack,
        quote_bundles=quote_bundles,
        runtime_targets=runtime_targets,
    )


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
        self._provenance_source_kind = "synthetic_snapshot"
        self._provenance_source_ref = "embedded_regime_snapshot"
        self._provenance_prior_family = "embedded_market_regime"

    @classmethod
    def from_dict(cls, data: dict[date, dict[float, float]]) -> MockDataProvider:
        """Create a provider with *only* user-supplied data (no built-in snapshots)."""
        inst = cls.__new__(cls)
        inst._data = dict(data)
        inst._sorted_dates = sorted(inst._data.keys())
        inst._provenance_source_kind = "user_supplied_snapshot"
        inst._provenance_source_ref = "MockDataProvider.from_dict"
        inst._provenance_prior_family = "user_supplied_snapshot_series"
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
        prior_seed_payload = "|".join(
            (
                self._provenance_prior_family,
                regime,
                best.isoformat(),
                as_of.isoformat() if as_of is not None else "",
            )
        ).encode("utf-8")
        prior_seed = int.from_bytes(
            hashlib.blake2s(prior_seed_payload, digest_size=8).digest(),
            "big",
        )
        underlier_spots = _build_underlier_spots(regime)
        if self._provenance_source_kind == "synthetic_snapshot":
            rates_pack = _build_synthetic_rates_model_pack(regime=regime, seed=prior_seed)
            credit_pack = _build_synthetic_credit_model_pack(regime=regime, seed=prior_seed)
            volatility_pack = _build_synthetic_volatility_model_pack(regime=regime, seed=prior_seed)
            discount_curves = _build_discount_curves(usd_ois, rates_pack)
            forecast_curves = _build_forecast_curves(discount_curves, rates_pack)
            rate_vol_surfaces = _build_rate_vol_surfaces(discount_curves["usd_ois"], rates_pack)
            credit_curves = _build_credit_curves(credit_pack)
            local_vol_surfaces = _build_local_vol_surfaces(
                volatility_pack=volatility_pack,
                underlier_spots=underlier_spots,
            )
            jump_parameter_sets = _build_jump_parameter_sets(volatility_pack)
            model_parameter_sets = _build_model_parameter_sets(volatility_pack)
            quote_bundles = _build_synthetic_quote_bundles(
                rates_pack=rates_pack,
                credit_pack=credit_pack,
                volatility_pack=volatility_pack,
                rate_vol_surfaces=rate_vol_surfaces,
                local_vol_surfaces=local_vol_surfaces,
            )
            runtime_targets = _build_synthetic_runtime_targets(
                discount_curves=discount_curves,
                forecast_curves=forecast_curves,
                credit_curves=credit_curves,
                rate_vol_surfaces=rate_vol_surfaces,
                local_vol_surfaces=local_vol_surfaces,
                jump_parameter_sets=jump_parameter_sets,
                model_parameter_sets=model_parameter_sets,
            )
            synthetic_generation_contract = _build_synthetic_generation_contract(
                regime=regime,
                seed=prior_seed,
                snapshot_date=best,
                requested_as_of=as_of,
                source_kind=self._provenance_source_kind,
                rates_pack=rates_pack,
                credit_pack=credit_pack,
                volatility_pack=volatility_pack,
                quote_bundles=quote_bundles,
                runtime_targets=runtime_targets,
            )
        else:
            discount_curves = {
                "usd_ois": usd_ois,
                "eur_ois": _shifted_curve(usd_ois, -175),
                "gbp_ois": _shifted_curve(usd_ois, -75),
            }
            forecast_curves = {
                "USD-SOFR-3M": _shifted_curve(discount_curves["usd_ois"], 35),
                "USD-LIBOR-3M": _shifted_curve(discount_curves["usd_ois"], 55),
                "USD-DISC": discount_curves["usd_ois"],
                "EUR-DISC": discount_curves["eur_ois"],
                "GBP-DISC": discount_curves["gbp_ois"],
                "EUR-EURIBOR-3M": _shifted_curve(discount_curves["eur_ois"], 30),
                "GBP-SONIA-3M": _shifted_curve(discount_curves["gbp_ois"], 18),
            }
            rate_vol_surfaces = _build_rate_vol_surfaces(
                discount_curves["usd_ois"],
                _build_synthetic_rates_model_pack(regime=regime, seed=prior_seed),
            )
            credit_curves = _build_credit_curves(
                _build_synthetic_credit_model_pack(regime=regime, seed=prior_seed),
            )
            volatility_pack = _build_synthetic_volatility_model_pack(regime=regime, seed=prior_seed)
            local_vol_surfaces = _build_local_vol_surfaces(
                volatility_pack=volatility_pack,
                underlier_spots=underlier_spots,
            )
            jump_parameter_sets = _build_jump_parameter_sets(volatility_pack)
            model_parameter_sets = _build_model_parameter_sets(volatility_pack)
            synthetic_generation_contract = None

        fixing_histories = _build_fixing_histories(best, forecast_curves)
        metadata = {
            "simulated": True,
            "regime": regime,
            "requested_as_of": as_of.isoformat() if as_of is not None else None,
            "snapshot_date": best.isoformat(),
            "description": "Deterministic stand-in market snapshot derived from embedded yield regimes.",
        }

        prior_parameters: dict[str, object] = {
            "regime": regime,
            "snapshot_date": best.isoformat(),
            "requested_as_of": as_of.isoformat() if as_of is not None else None,
        }
        if self._provenance_source_kind == "synthetic_snapshot":
            prior_parameters.update(
                {
                    "curve_set": rates_pack.anchor_curve_set,
                    "forecast_curve_shifts_bps": {
                        "USD-SOFR-3M": float(rates_pack.forecast_basis_bps["USD-SOFR-3M"]),
                        "USD-LIBOR-3M": float(rates_pack.forecast_basis_bps["USD-LIBOR-3M"]),
                        "EUR-DISC": float(rates_pack.discount_curve_shifts_bps["eur_ois"]),
                        "GBP-DISC": float(rates_pack.discount_curve_shifts_bps["gbp_ois"]),
                    },
                    "surface_sets": {
                        "vol_surfaces": rates_pack.rate_vol_surface_family,
                        "local_vol_surfaces": volatility_pack.local_vol_surface_family,
                        "jump_parameter_sets": volatility_pack.jump_parameter_family,
                        "model_parameter_sets": volatility_pack.model_parameter_family,
                    },
                }
            )
            prior_parameters["synthetic_generation_contract"] = synthetic_generation_contract.to_payload()
            prior_parameters["model_consistency_contract"] = _build_model_consistency_contract(
                synthetic_generation_contract,
            )
        else:
            prior_parameters.update(
                {
                    "curve_set": "user_supplied_yield_series",
                    "source": self._provenance_source_ref,
                }
            )

        return MarketSnapshot(
            as_of=best,
            source="mock",
            discount_curves=discount_curves,
            forecast_curves=forecast_curves,
            vol_surfaces=rate_vol_surfaces,
            credit_curves=credit_curves,
            fixing_histories=fixing_histories,
            fx_rates=_build_fx_rates(regime),
            state_spaces=_build_state_spaces(regime),
            underlier_spots=underlier_spots,
            local_vol_surfaces=local_vol_surfaces,
            jump_parameter_sets=jump_parameter_sets,
            model_parameter_sets=model_parameter_sets,
            metadata=metadata,
            provenance={
                "source": "mock",
                "source_kind": self._provenance_source_kind,
                "source_ref": self._provenance_source_ref,
                "prior_family": self._provenance_prior_family,
                "prior_seed": prior_seed,
                "prior_parameters": prior_parameters,
            },
            default_discount_curve="usd_ois",
            default_vol_surface="usd_rates_smile",
            default_credit_curve="usd_ig",
            default_fixing_history="USD-SOFR-3M",
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
