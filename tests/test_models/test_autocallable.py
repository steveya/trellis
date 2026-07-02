"""Tests for bounded single-underlier autocallable MC/QMC helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.models.vol_surface import FlatVol


@dataclass(frozen=True)
class AutocallableNoteSpec:
    notional: float = 1_000.0
    spot: float = 100.0
    initial_spot: float = 100.0
    autocall_barrier: float = 1.0
    protection_barrier: float = 0.7
    coupon_rate: float = 0.08
    expiry_date: date = date(2026, 1, 2)
    observation_times: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def _market_state() -> MarketState:
    return MarketState(
        as_of=date(2025, 1, 2),
        settlement=date(2025, 1, 2),
        discount=YieldCurve.flat(0.03),
        vol_surface=FlatVol(0.2),
        spot=100.0,
    )


def test_autocallable_path_payoffs_use_first_trigger_redemption():
    from trellis.models.autocallable import (
        AutocallableRuntimeSpec,
        autocallable_path_payoffs,
    )

    spec = AutocallableRuntimeSpec(
        notional=1_000.0,
        initial_spot=100.0,
        spot=100.0,
        maturity=1.0,
        rate=0.0,
        sigma=0.0,
        observation_times=(0.5, 1.0),
        autocall_barrier=100.0,
        protection_barrier=70.0,
        coupon_rate=0.08,
    )
    paths = np.asarray(
        [
            [100.0, 104.0, 80.0],
            [100.0, 99.0, 110.0],
            [100.0, 95.0, 65.0],
        ]
    )

    cashflows, payment_times, called = autocallable_path_payoffs(paths, spec)

    assert called.tolist() == [True, True, False]
    assert payment_times.tolist() == [0.5, 1.0, 1.0]
    assert cashflows.tolist() == pytest.approx([1040.0, 1080.0, 650.0])


def test_autocallable_resolver_binds_market_aliases_and_ratios():
    from trellis.models.autocallable import resolve_autocallable_inputs

    resolved = resolve_autocallable_inputs(_market_state(), AutocallableNoteSpec())

    assert resolved.initial_spot == 100.0
    assert resolved.spot == 100.0
    assert resolved.autocall_barrier == 100.0
    assert resolved.protection_barrier == 70.0
    assert resolved.rate == pytest.approx(0.03)
    assert resolved.sigma == pytest.approx(0.2)
    assert resolved.observation_times == (0.25, 0.5, 0.75, 1.0)


def test_autocallable_pseudo_sampling_does_not_call_sobol(monkeypatch):
    import trellis.models.autocallable as autocallable
    from trellis.models.autocallable import (
        AutocallableMonteCarloConfig,
        price_autocallable_monte_carlo_result,
    )

    def fail_sobol(*_args, **_kwargs):
        raise AssertionError("pseudo Monte Carlo must not use Sobol normals")

    monkeypatch.setattr(autocallable, "sobol_normals", fail_sobol)

    result = price_autocallable_monte_carlo_result(
        _market_state(),
        AutocallableNoteSpec(),
        config=AutocallableMonteCarloConfig(n_paths=512, n_steps=24, seed=11, sampling="pseudo"),
    )

    assert result.validation_bundle == "autocallable:monte_carlo_gbm"
    assert result.sampling == "pseudo"
    assert result.price > 0.0


def test_autocallable_qmc_sampling_uses_sobol(monkeypatch):
    import trellis.models.autocallable as autocallable
    from trellis.models.autocallable import (
        AutocallableMonteCarloConfig,
        price_autocallable_monte_carlo_result,
    )

    calls = []

    def fake_sobol(n_paths, n_steps, n_factors=1):
        calls.append((n_paths, n_steps, n_factors))
        return np.zeros((n_paths, n_steps))

    monkeypatch.setattr(autocallable, "sobol_normals", fake_sobol)

    result = price_autocallable_monte_carlo_result(
        _market_state(),
        AutocallableNoteSpec(),
        config=AutocallableMonteCarloConfig(n_paths=128, n_steps=12, seed=11, sampling="sobol"),
    )

    assert calls == [(128, 12, 1)]
    assert result.validation_bundle == "autocallable:qmc_sobol_gbm"
    assert result.sampling == "sobol"
    assert result.price > 0.0


def test_autocallable_pseudo_and_qmc_share_event_contract():
    from trellis.models.autocallable import (
        AutocallableMonteCarloConfig,
        price_autocallable_monte_carlo_result,
    )

    spec = AutocallableNoteSpec(autocall_barrier=1.08, protection_barrier=0.65)
    pseudo = price_autocallable_monte_carlo_result(
        _market_state(),
        spec,
        config=AutocallableMonteCarloConfig(n_paths=8192, n_steps=48, seed=29, sampling="pseudo"),
    )
    qmc = price_autocallable_monte_carlo_result(
        _market_state(),
        spec,
        config=AutocallableMonteCarloConfig(n_paths=8192, n_steps=48, seed=29, sampling="sobol"),
    )

    assert pseudo.path_contract == qmc.path_contract
    assert pseudo.observation_steps == qmc.observation_steps
    assert qmc.price == pytest.approx(pseudo.price, rel=0.06, abs=15.0)
