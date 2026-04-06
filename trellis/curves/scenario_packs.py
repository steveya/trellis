"""Reusable named rate-curve scenario packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trellis.curves.shocks import CurveShockWarning, build_curve_shock_surface

if TYPE_CHECKING:
    from trellis.curves.yield_curve import YieldCurve


DEFAULT_RATE_SCENARIO_BUCKET_TENORS = (2.0, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class RateCurveScenario:
    """One named rate-curve scenario template."""

    name: str
    pack: str
    description: str
    bucket_tenors: tuple[float, ...]
    _tenor_bumps: tuple[tuple[float, float], ...]
    warnings: tuple[CurveShockWarning, ...] = field(default_factory=tuple)

    @property
    def tenor_bumps(self) -> dict[float, float]:
        """Return the bucket shocks as a plain mapping in basis points."""
        return {float(tenor): float(bump) for tenor, bump in self._tenor_bumps}

    def to_pipeline_spec(self) -> dict[str, object]:
        """Project the scenario into the Pipeline scenario-spec shape."""
        return {
            "name": self.name,
            "scenario_pack": self.pack,
            "description": self.description,
            "bucket_tenors": self.bucket_tenors,
            "tenor_bumps": self.tenor_bumps,
            "scenario_warnings": [
                {
                    "code": warning.code,
                    "message": warning.message,
                    "tenor": warning.tenor,
                }
                for warning in self.warnings
            ],
        }


def build_rate_curve_scenario_pack(
    curve: YieldCurve,
    *,
    pack: str,
    bucket_tenors: tuple[float, ...] = DEFAULT_RATE_SCENARIO_BUCKET_TENORS,
    amplitude_bps: float = 25.0,
) -> tuple[RateCurveScenario, ...]:
    """Build a named desk-style rate scenario pack on the requested bucket grid."""
    surface = build_curve_shock_surface(curve, bucket_tenors)
    canonical_tenors = tuple(bucket.tenor for bucket in surface.buckets)
    warnings = surface.warnings
    amplitude_bps = float(amplitude_bps)
    amplitude_label = _format_bps_label(amplitude_bps)
    pack_key = pack.strip().lower().replace("-", "_")

    if pack_key == "twist":
        return (
            _make_scenario(
                name=f"twist_steepener_{amplitude_label}",
                pack="twist",
                description=(
                    "Short-end rates down and long-end rates up on the configured bucket grid."
                ),
                bucket_tenors=canonical_tenors,
                tenor_bumps=_linear_profile(canonical_tenors, -amplitude_bps, +amplitude_bps),
                warnings=warnings,
            ),
            _make_scenario(
                name=f"twist_flattener_{amplitude_label}",
                pack="twist",
                description=(
                    "Short-end rates up and long-end rates down on the configured bucket grid."
                ),
                bucket_tenors=canonical_tenors,
                tenor_bumps=_linear_profile(canonical_tenors, +amplitude_bps, -amplitude_bps),
                warnings=warnings,
            ),
        )

    if pack_key == "butterfly":
        if len(canonical_tenors) < 3:
            raise ValueError("Butterfly scenario packs require at least three bucket tenors.")
        return (
            _make_scenario(
                name=f"butterfly_belly_up_{amplitude_label}",
                pack="butterfly",
                description="Wings down and belly up on the configured bucket grid.",
                bucket_tenors=canonical_tenors,
                tenor_bumps=_butterfly_profile(canonical_tenors, amplitude_bps),
                warnings=warnings,
            ),
            _make_scenario(
                name=f"butterfly_belly_down_{amplitude_label}",
                pack="butterfly",
                description="Wings up and belly down on the configured bucket grid.",
                bucket_tenors=canonical_tenors,
                tenor_bumps=_butterfly_profile(canonical_tenors, -amplitude_bps),
                warnings=warnings,
            ),
        )

    raise ValueError(f"Unknown rate scenario pack: {pack!r}")


def _format_bps_label(amplitude_bps: float) -> str:
    normalized = f"{float(amplitude_bps):g}"
    return f"{normalized.replace('-', 'm').replace('.', 'p')}bp"


def _make_scenario(
    *,
    name: str,
    pack: str,
    description: str,
    bucket_tenors: tuple[float, ...],
    tenor_bumps: dict[float, float],
    warnings: tuple[CurveShockWarning, ...],
) -> RateCurveScenario:
    return RateCurveScenario(
        name=name,
        pack=pack,
        description=description,
        bucket_tenors=tuple(float(tenor) for tenor in bucket_tenors),
        _tenor_bumps=tuple((float(tenor), float(tenor_bumps[tenor])) for tenor in bucket_tenors),
        warnings=tuple(warnings),
    )


def _linear_profile(
    tenors: tuple[float, ...],
    start_bps: float,
    end_bps: float,
) -> dict[float, float]:
    if len(tenors) == 1:
        return {float(tenors[0]): float(end_bps)}
    span = tenors[-1] - tenors[0]
    if span == 0.0:
        return {float(tenor): float(end_bps) for tenor in tenors}
    return {
        float(tenor): float(start_bps + (end_bps - start_bps) * ((tenor - tenors[0]) / span))
        for tenor in tenors
    }


def _butterfly_profile(
    tenors: tuple[float, ...],
    amplitude_bps: float,
) -> dict[float, float]:
    return {
        float(tenor): float(amplitude_bps if 0 < index < len(tenors) - 1 else -amplitude_bps)
        for index, tenor in enumerate(tenors)
    }
