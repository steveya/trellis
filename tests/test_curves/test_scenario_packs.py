"""Tests for reusable rate-curve scenario packs."""

from __future__ import annotations

import pytest

from trellis.curves.scenario_packs import build_rate_curve_scenario_pack
from trellis.curves.yield_curve import YieldCurve


def test_twist_pack_builds_named_steepener_and_flattener():
    curve = YieldCurve([1.0, 5.0, 10.0, 30.0], [0.04, 0.042, 0.045, 0.047])

    scenarios = build_rate_curve_scenario_pack(
        curve,
        pack="twist",
        bucket_tenors=(2.0, 5.0, 10.0, 30.0),
        amplitude_bps=25.0,
    )

    assert [scenario.name for scenario in scenarios] == [
        "twist_steepener_25bp",
        "twist_flattener_25bp",
    ]
    steepener = scenarios[0].tenor_bumps
    assert steepener[2.0] == pytest.approx(-25.0)
    assert steepener[30.0] == pytest.approx(+25.0)


def test_butterfly_pack_builds_named_belly_up_and_down():
    curve = YieldCurve([1.0, 5.0, 10.0, 30.0], [0.04, 0.042, 0.045, 0.047])

    scenarios = build_rate_curve_scenario_pack(
        curve,
        pack="butterfly",
        bucket_tenors=(2.0, 5.0, 10.0, 30.0),
        amplitude_bps=20.0,
    )

    assert [scenario.name for scenario in scenarios] == [
        "butterfly_belly_up_20bp",
        "butterfly_belly_down_20bp",
    ]
    belly_up = scenarios[0].tenor_bumps
    assert belly_up[2.0] == pytest.approx(-20.0)
    assert belly_up[5.0] == pytest.approx(+20.0)
    assert belly_up[10.0] == pytest.approx(+20.0)
    assert belly_up[30.0] == pytest.approx(-20.0)


def test_scenario_pack_names_preserve_non_integer_amplitudes():
    curve = YieldCurve([1.0, 5.0, 10.0, 30.0], [0.04, 0.042, 0.045, 0.047])

    scenarios = build_rate_curve_scenario_pack(
        curve,
        pack="twist",
        bucket_tenors=(2.0, 5.0, 10.0, 30.0),
        amplitude_bps=25.5,
    )

    assert [scenario.name for scenario in scenarios] == [
        "twist_steepener_25p5bp",
        "twist_flattener_25p5bp",
    ]
