from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trellis.agent.benchmark_contracts import (
    benchmark_preferred_method,
    benchmark_request_description,
    benchmark_spec_overrides,
    canonical_benchmark_instrument_type,
)
from trellis.agent.task_manifests import load_task_manifest
from trellis.agent.task_runtime import prepare_existing_task
from trellis.core.types import DayCountConvention, Frequency


ROOT = Path(__file__).resolve().parents[2]


def _benchmark_tasks() -> dict[str, dict]:
    return {
        task["id"]: task
        for task in load_task_manifest("TASKS_BENCHMARK_FINANCEPY.yaml", root=ROOT)
    }


def test_canonical_benchmark_instrument_type_maps_broad_runtime_families():
    tasks = _benchmark_tasks()

    assert canonical_benchmark_instrument_type(tasks["F002"]) == "european_option"
    assert canonical_benchmark_instrument_type(tasks["F003"]) == "cap"
    assert canonical_benchmark_instrument_type(tasks["F008"]) == "basket_option"
    assert canonical_benchmark_instrument_type(tasks["F010"]) == "digital_option"
    assert canonical_benchmark_instrument_type(tasks["F011"]) == "lookback_option"
    assert canonical_benchmark_instrument_type(tasks["F012"]) == "chooser_option"
    assert canonical_benchmark_instrument_type(tasks["F013"]) == "compound_option"
    assert canonical_benchmark_instrument_type(tasks["F014"]) == "cliquet_option"
    assert canonical_benchmark_instrument_type(tasks["F015"]) == "variance_swap"


def test_benchmark_preferred_method_uses_single_declared_construct():
    tasks = _benchmark_tasks()

    assert benchmark_preferred_method(tasks["F002"]) == "analytical"
    assert benchmark_preferred_method(tasks["F014"]) == "analytical"


def test_benchmark_request_description_makes_cap_request_explicit():
    tasks = _benchmark_tasks()

    description = benchmark_request_description(tasks["F003"], root=ROOT)

    assert description is not None
    assert "Instrument class: cap." in description
    assert "Rate index: USD-SOFR-3M." in description
    assert "cap/floor" not in description.lower()


def test_benchmark_spec_overrides_cover_fx_rates_cap_and_swaption_contracts():
    tasks = _benchmark_tasks()

    fx = benchmark_spec_overrides(tasks["F002"], root=ROOT)
    assert fx["fx_pair"] == "EURUSD"
    assert fx["foreign_discount_key"] == "EUR-DISC"
    assert fx["expiry_date"] == date(2025, 11, 15)

    cap = benchmark_spec_overrides(tasks["F003"], root=ROOT)
    assert cap["start_date"] == date(2024, 11, 15)
    assert cap["end_date"] == date(2029, 11, 15)
    assert cap["frequency"] is Frequency.QUARTERLY
    assert cap["day_count"] is DayCountConvention.ACT_360
    assert cap["rate_index"] == "USD-SOFR-3M"
    assert cap["instrument_class"] == "cap"
    assert cap["model"] == "black"

    swaption = benchmark_spec_overrides(tasks["F006"], root=ROOT)
    assert swaption["expiry_date"] == date(2025, 11, 15)
    assert swaption["swap_start"] == date(2025, 11, 15)
    assert swaption["swap_end"] == date(2030, 11, 15)
    assert swaption["swap_frequency"] is Frequency.SEMI_ANNUAL
    assert swaption["day_count"] is DayCountConvention.THIRTY_E_360
    assert swaption["is_payer"] is True

    cds = benchmark_spec_overrides(tasks["F007"], root=ROOT)
    assert cds["start_date"] == date(2024, 9, 20)
    assert cds["end_date"] == date(2029, 12, 20)
    assert cds["frequency"] is Frequency.QUARTERLY
    assert cds["day_count"] is DayCountConvention.ACT_360

    digital = benchmark_spec_overrides(tasks["F010"], root=ROOT)
    assert digital["cash_payoff"] == pytest.approx(10.0)
    assert digital["payout_type"] == "cash_or_nothing"

    lookback = benchmark_spec_overrides(tasks["F011"], root=ROOT)
    assert lookback["running_extreme"] == pytest.approx(100.0)

    chooser = benchmark_spec_overrides(tasks["F012"], root=ROOT)
    assert chooser["call_strike"] == pytest.approx(100.0)
    assert chooser["put_strike"] == pytest.approx(100.0)

    compound = benchmark_spec_overrides(tasks["F013"], root=ROOT)
    assert compound["outer_option_type"] == "call"
    assert compound["inner_option_type"] == "call"

    cliquet = benchmark_spec_overrides(tasks["F014"], root=ROOT)
    assert cliquet["observation_dates"] == (
        date(2024, 11, 18),
        date(2025, 2, 17),
        date(2025, 5, 19),
        date(2025, 8, 18),
        date(2025, 11, 17),
    )
    assert cliquet["day_count"] is DayCountConvention.THIRTY_E_360
    assert cliquet["time_day_count"] is DayCountConvention.ACT_365

    variance_swap = benchmark_spec_overrides(tasks["F015"], root=ROOT)
    assert variance_swap["strike_variance"] == pytest.approx(0.04)
    assert variance_swap["replication_strikes"] == "60.0,80.0,100.0,120.0,140.0"
    assert variance_swap["replication_volatilities"] == "0.26,0.24,0.22,0.23,0.25"

    shifted_cap = benchmark_spec_overrides(tasks["F004"], root=ROOT)
    assert shifted_cap["model"] == "shifted_black"
    assert shifted_cap["shift"] == pytest.approx(0.01)

    sabr_cap = benchmark_spec_overrides(tasks["F005"], root=ROOT)
    assert sabr_cap["model"] == "sabr"
    assert sabr_cap["sabr"] == {
        "alpha": 0.025,
        "beta": 0.5,
        "rho": -0.2,
        "nu": 0.35,
    }

    barrier = benchmark_spec_overrides(tasks["F009"], root=ROOT)
    assert barrier["observations_per_year"] == 252


@pytest.mark.parametrize(
    ("task_id", "expected_spec_name", "expected_module_suffix"),
    [
        ("F002", "FXVanillaOptionSpec", "fxvanillaanalytical"),
        ("F003", "AgentCapSpec", "agentcap"),
        ("F009", "BarrierOptionSpec", "barrieroption"),
        ("F008", "BasketOptionSpec", "basketoption"),
        ("F010", "DigitalOptionSpec", "digitaloption"),
        ("F011", "LookbackOptionSpec", "lookbackoption"),
        ("F012", "ChooserOptionSpec", "chooseroption"),
        ("F013", "CompoundOptionSpec", "compoundoption"),
        ("F014", "CliquetOptionSpec", "cliquetoption"),
        ("F015", "VarianceSwapSpec", "varianceswap"),
    ],
)
def test_prepare_existing_task_uses_benchmark_normalization(
    task_id: str,
    expected_spec_name: str,
    expected_module_suffix: str,
):
    task = _benchmark_tasks()[task_id]

    prepared = prepare_existing_task(task)

    assert prepared.spec_schema is not None
    assert prepared.spec_schema.spec_name == expected_spec_name
    assert prepared.payoff_cls.__module__.endswith(expected_module_suffix)
