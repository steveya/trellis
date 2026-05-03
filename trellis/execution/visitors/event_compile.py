"""Event-compilation visitors for bounded dynamic execution IR artifacts."""

from __future__ import annotations

from datetime import date
from typing import Mapping

from trellis.conventions.day_count import DayCountConvention
from trellis.core.types import Frequency
from trellis.execution.ir import ContractExecutionIR
from trellis.instruments.callable_bond import CallableBondSpec


def compile_callable_bond_spec_from_execution_ir(
    ir: ContractExecutionIR,
) -> CallableBondSpec:
    """Compile a bounded callable-bond execution IR back into helper inputs."""
    if not isinstance(ir, ContractExecutionIR):
        raise TypeError("ir must be a ContractExecutionIR")
    if ir.source_track.product_family != "callable_bond":
        raise ValueError(
            "compile_callable_bond_spec_from_execution_ir requires callable_bond execution IR"
        )

    metadata = _metadata_dict(ir.source_track.source_metadata)
    return CallableBondSpec(
        notional=float(metadata["notional"]),
        coupon=float(metadata["coupon"]),
        start_date=_coerce_date(metadata["start_date"], "start_date"),
        end_date=_coerce_date(metadata["end_date"], "end_date"),
        call_dates=tuple(
            _coerce_date(call_date, "call_dates item")
            for call_date in tuple(metadata.get("call_dates") or ())
        ),
        call_price=float(metadata.get("call_price_quoted", 100.0)),
        frequency=_frequency(metadata.get("frequency")),
        day_count=_day_count(metadata.get("day_count")),
    )


def _metadata_dict(values: object) -> dict[str, object]:
    if isinstance(values, Mapping):
        items = values.items()
    else:
        items = values or ()
    return {str(key): value for key, value in items}


def _coerce_date(value: object, label: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return date.fromisoformat(text)


def _day_count(value: object) -> DayCountConvention:
    if isinstance(value, DayCountConvention):
        return value
    text = str(value or "").strip()
    if not text:
        return DayCountConvention.ACT_365
    try:
        return DayCountConvention[text]
    except KeyError:
        return DayCountConvention(text)


def _frequency(value: object) -> Frequency:
    if isinstance(value, Frequency):
        return value
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"annual", "yearly"}:
        return Frequency.ANNUAL
    if text in {"semiannual", "semi_annual", "semiannually"}:
        return Frequency.SEMI_ANNUAL
    if text in {"quarterly", "quarter"}:
        return Frequency.QUARTERLY
    if text in {"monthly", "month"}:
        return Frequency.MONTHLY
    raise ValueError(f"Unsupported frequency {value!r}")


__all__ = ["compile_callable_bond_spec_from_execution_ir"]
