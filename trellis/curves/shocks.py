"""Interpolation-aware bucket shocks for yield curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trellis.core.differentiable import get_numpy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from trellis.curves.yield_curve import YieldCurve

np = get_numpy()


@dataclass(frozen=True)
class CurveShockWarning:
    """Explicit warning describing a bucket-support limitation."""

    code: str
    message: str
    tenor: float | None = None


@dataclass(frozen=True)
class CurveShockBucket:
    """Metadata describing how one requested bucket sits on the base curve."""

    tenor: float
    base_zero_rate: float
    is_exact_curve_tenor: bool
    left_support_tenor: float | None
    right_support_tenor: float | None
    support_width: float | None
    warnings: tuple[CurveShockWarning, ...] = ()


@dataclass(frozen=True)
class CurveShockSurface:
    """Reusable bucket surface for later KRD and scenario workflows."""

    base_tenors: tuple[float, ...]
    base_rates: tuple[float, ...]
    buckets: tuple[CurveShockBucket, ...]
    support_width_warning_years: float | None = None

    @property
    def warnings(self) -> tuple[CurveShockWarning, ...]:
        """Flatten all bucket warnings into one stable tuple."""
        return tuple(warning for bucket in self.buckets for warning in bucket.warnings)

    def bucketed_curve(self) -> YieldCurve:
        """Return the base curve re-expressed on the configured bucket grid."""
        from trellis.curves.yield_curve import YieldCurve

        if not self.buckets:
            return YieldCurve(self.base_tenors, self.base_rates)

        bucketed_curve = YieldCurve(
            [bucket.tenor for bucket in self.buckets],
            [bucket.base_zero_rate for bucket in self.buckets],
        )
        bucketed_curve.curve_shock_surface = self
        bucketed_curve.curve_shock_warnings = self.warnings
        return bucketed_curve

    def bucket_for_tenor(self, tenor: float) -> CurveShockBucket:
        """Return the configured bucket matching *tenor*."""
        tenor = float(tenor)
        for bucket in self.buckets:
            if np.isclose(bucket.tenor, tenor):
                return bucket
        raise KeyError(f"No bucket configured for tenor {tenor}.")

    def apply_bumps(self, tenor_bumps: Mapping[float, float]) -> YieldCurve:
        """Return a shocked yield curve with exact and off-grid buckets applied."""
        from trellis.curves.yield_curve import YieldCurve

        points = {
            float(tenor): float(rate)
            for tenor, rate in zip(self.base_tenors, self.base_rates)
        }
        applied_bumps: dict[float, float] = {}
        for tenor, bump_bps in tenor_bumps.items():
            bucket = self.bucket_for_tenor(float(tenor))
            bump_decimal = float(bump_bps) / 10_000.0
            if bucket.is_exact_curve_tenor:
                points[bucket.tenor] = points[bucket.tenor] + bump_decimal
            else:
                points[bucket.tenor] = bucket.base_zero_rate + bump_decimal
            applied_bumps[bucket.tenor] = float(bump_bps)

        shocked_tenors = sorted(points)
        shocked_rates = [points[tenor] for tenor in shocked_tenors]
        shocked_curve = YieldCurve(shocked_tenors, shocked_rates)
        shocked_curve.curve_shock_surface = self
        shocked_curve.curve_shock_warnings = self.warnings
        shocked_curve.curve_shock_bumps = applied_bumps
        return shocked_curve


def build_curve_shock_surface(
    curve: YieldCurve,
    bucket_tenors: Sequence[float],
    *,
    support_width_warning_years: float | None = 10.0,
) -> CurveShockSurface:
    """Describe bucket support for *bucket_tenors* on top of *curve*."""
    base_tenors = tuple(float(tenor) for tenor in np.asarray(curve.tenors, dtype=float))
    base_rates = tuple(float(rate) for rate in np.asarray(curve.rates, dtype=float))
    buckets = tuple(
        _build_curve_shock_bucket(
            curve,
            tenor,
            support_width_warning_years=support_width_warning_years,
        )
        for tenor in _normalize_bucket_tenors(bucket_tenors)
    )
    return CurveShockSurface(
        base_tenors=base_tenors,
        base_rates=base_rates,
        buckets=buckets,
        support_width_warning_years=support_width_warning_years,
    )


def _normalize_bucket_tenors(bucket_tenors: Sequence[float]) -> tuple[float, ...]:
    normalized: list[float] = []
    for tenor in sorted(float(tenor) for tenor in bucket_tenors):
        if normalized and np.isclose(normalized[-1], tenor):
            continue
        normalized.append(tenor)
    return tuple(normalized)


def _build_curve_shock_bucket(
    curve: YieldCurve,
    tenor: float,
    *,
    support_width_warning_years: float | None,
) -> CurveShockBucket:
    base_tenors = tuple(float(value) for value in np.asarray(curve.tenors, dtype=float))
    tenor = float(tenor)
    exact_index = _find_exact_index(base_tenors, tenor)
    warnings: list[CurveShockWarning] = []

    if exact_index is not None:
        effective_tenor = base_tenors[exact_index]
        left_support = base_tenors[exact_index - 1] if exact_index > 0 else None
        right_support = (
            base_tenors[exact_index + 1]
            if exact_index + 1 < len(base_tenors)
            else None
        )
        support_width = (
            right_support - left_support
            if left_support is not None and right_support is not None
            else None
        )
        is_exact_curve_tenor = True
    else:
        insertion_index = int(np.searchsorted(np.asarray(base_tenors, dtype=float), tenor))
        effective_tenor = tenor
        left_support = base_tenors[insertion_index - 1] if insertion_index > 0 else None
        right_support = (
            base_tenors[insertion_index]
            if insertion_index < len(base_tenors)
            else None
        )
        support_width = (
            right_support - left_support
            if left_support is not None and right_support is not None
            else None
        )
        is_exact_curve_tenor = False
        if left_support is None:
            warnings.append(
                CurveShockWarning(
                    code="below_curve_support",
                    tenor=effective_tenor,
                    message="Bucket tenor sits below the first curve knot and will extend the left edge.",
                )
            )
        if right_support is None:
            warnings.append(
                CurveShockWarning(
                    code="above_curve_support",
                    tenor=effective_tenor,
                    message="Bucket tenor sits above the last curve knot and will extend the right edge.",
                )
            )

    if (
        support_width_warning_years is not None
        and support_width is not None
        and support_width > support_width_warning_years
    ):
        warnings.append(
            CurveShockWarning(
                code="wide_support_interval",
                tenor=effective_tenor,
                message=(
                    "Bucket tenor spans a wide interpolation interval on the base curve; "
                    "later KRD or scenario consumers should treat the support as sparse."
                ),
            )
        )

    return CurveShockBucket(
        tenor=effective_tenor,
        base_zero_rate=float(curve.zero_rate(effective_tenor)),
        is_exact_curve_tenor=is_exact_curve_tenor,
        left_support_tenor=left_support,
        right_support_tenor=right_support,
        support_width=support_width,
        warnings=tuple(warnings),
    )


def _find_exact_index(tenors: tuple[float, ...], target: float) -> int | None:
    for index, tenor in enumerate(tenors):
        if np.isclose(tenor, target):
            return index
    return None
