"""Tests for typed Hull-White model-parameter selection."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest


def _strict_parameter_state(**overrides):
    payload = {
        "model_family": "hull_white",
        "parameter_set_name": "usd_hw_2026_09_03",
        "mean_reversion": 0.08,
        "sigma": 0.011,
        "source_kind": "calibrated",
        "calibration_source": "usd_swaption_cube_close",
    }
    payload.update(overrides)
    return SimpleNamespace(
        model_parameters={
            "model_family": "hull_white",
            "mean_reversion": 9.9,
            "sigma": 9.9,
        },
        model_parameter_sets={"usd_hw_2026_09_03": payload},
    )


def test_hull_white_payload_builder_requires_explicit_calibration_source():
    from trellis.models.hull_white_parameters import build_hull_white_parameter_payload

    with pytest.raises(TypeError, match="calibration_source"):
        build_hull_white_parameter_payload(0.08, 0.011)

    payload = build_hull_white_parameter_payload(
        0.08,
        0.011,
        parameter_set_name="usd_hw_2026_09_03",
        source_kind="calibrated",
        calibration_source="calibrate_hull_white",
    )

    assert payload["calibration_source"] == "calibrate_hull_white"

    observed_payload = build_hull_white_parameter_payload(
        0.08,
        0.011,
        parameter_set_name="usd_hw_observed",
        source_kind="observed",
        calibration_source="validated_market_snapshot",
    )
    assert observed_payload["source_kind"] == "observed"

    with pytest.raises(ValueError, match="calibration_source"):
        build_hull_white_parameter_payload(
            0.08,
            0.011,
            parameter_set_name="usd_hw_2026_09_03",
            source_kind="calibrated",
            calibration_source="  ",
        )


@pytest.mark.parametrize("bad_name", ("", "  ", " usd_hw", "usd_hw ", None, 7))
def test_hull_white_payload_builder_rejects_inexact_parameter_set_names(bad_name):
    from trellis.models.hull_white_parameters import build_hull_white_parameter_payload

    with pytest.raises(ValueError, match="parameter_set_name.*exact nonblank string"):
        build_hull_white_parameter_payload(
            0.08,
            0.011,
            parameter_set_name=bad_name,  # type: ignore[arg-type]
            source_kind="calibrated",
            calibration_source="calibrate_hull_white",
        )


@pytest.mark.parametrize(
    "bad_source_kind",
    ("default", "inferred", "synthetic", " calibrated", "calibrated ", ""),
)
def test_hull_white_payload_builder_rejects_unsupported_provenance_kinds(
    bad_source_kind,
):
    from trellis.models.hull_white_parameters import build_hull_white_parameter_payload

    with pytest.raises(ValueError, match="source_kind.*observed.*calibrated"):
        build_hull_white_parameter_payload(
            0.08,
            0.011,
            parameter_set_name="usd_hw",
            source_kind=bad_source_kind,
            calibration_source="desk_close",
        )


@pytest.mark.parametrize("bad_name", (" usd_hw_2026_09_03", "usd_hw_2026_09_03 "))
def test_resolved_hull_white_identity_rejects_surrounding_whitespace(bad_name):
    from trellis.models.hull_white_parameters import ResolvedHullWhiteParameterSet

    with pytest.raises(ValueError, match="parameter_set_name.*exact nonblank string"):
        ResolvedHullWhiteParameterSet(
            parameter_set_name=bad_name,
            mean_reversion=0.08,
            sigma=0.011,
            source_kind="calibrated",
            calibration_source="test",
        )


def test_named_hull_white_parameter_selection_returns_frozen_provenance():
    from trellis.models.hull_white_parameters import (
        ResolvedHullWhiteParameterSet,
        resolve_named_hull_white_parameter_set,
    )

    resolved = resolve_named_hull_white_parameter_set(
        _strict_parameter_state(),
        parameter_set_name="usd_hw_2026_09_03",
    )

    assert resolved == ResolvedHullWhiteParameterSet(
        parameter_set_name="usd_hw_2026_09_03",
        mean_reversion=0.08,
        sigma=0.011,
        source_kind="calibrated",
        calibration_source="usd_swaption_cube_close",
    )
    with pytest.raises(FrozenInstanceError):
        resolved.sigma = 0.25


def test_named_hull_white_parameter_selection_accepts_zero_model_parameters():
    from trellis.models.hull_white_parameters import resolve_named_hull_white_parameter_set

    resolved = resolve_named_hull_white_parameter_set(
        _strict_parameter_state(mean_reversion=0.0, sigma=0.0),
        parameter_set_name="usd_hw_2026_09_03",
    )

    assert resolved.mean_reversion == 0.0
    assert resolved.sigma == 0.0


def test_named_hull_white_parameter_selection_never_falls_back_to_direct_parameters():
    from trellis.models.hull_white_parameters import resolve_named_hull_white_parameter_set

    market_state = SimpleNamespace(
        model_parameters={
            "model_family": "hull_white",
            "parameter_set_name": "requested",
            "mean_reversion": 0.08,
            "sigma": 0.011,
            "source_kind": "calibrated",
            "calibration_source": "direct_payload",
        },
        model_parameter_sets={},
    )

    with pytest.raises(ValueError, match="named Hull-White parameter set 'requested'"):
        resolve_named_hull_white_parameter_set(
            market_state,
            parameter_set_name="requested",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parameter_set_name": "different"}, "parameter_set_name"),
        ({"model_family": "black"}, "model_family"),
        ({"model_family": ""}, "model_family"),
        ({"mean_reversion": None}, "mean_reversion"),
        ({"mean_reversion": float("nan")}, "mean_reversion"),
        ({"mean_reversion": float("inf")}, "mean_reversion"),
        ({"mean_reversion": -0.01}, "mean_reversion"),
        ({"sigma": None}, "sigma"),
        ({"sigma": float("nan")}, "sigma"),
        ({"sigma": float("inf")}, "sigma"),
        ({"sigma": -0.01}, "sigma"),
        ({"source_kind": "  "}, "source_kind"),
        ({"source_kind": "default"}, "source_kind"),
        ({"source_kind": "inferred"}, "source_kind"),
        ({"source_kind": "synthetic"}, "source_kind"),
        ({"source_kind": " calibrated"}, "source_kind"),
        ({"calibration_source": "  "}, "calibration_source"),
    ],
)
def test_named_hull_white_parameter_selection_rejects_incomplete_or_invalid_sets(
    overrides,
    message,
):
    from trellis.models.hull_white_parameters import resolve_named_hull_white_parameter_set

    with pytest.raises(ValueError, match=message):
        resolve_named_hull_white_parameter_set(
            _strict_parameter_state(**overrides),
            parameter_set_name="usd_hw_2026_09_03",
        )


def test_named_hull_white_parameter_selection_does_not_translate_black_volatility():
    from trellis.models.hull_white_parameters import resolve_named_hull_white_parameter_set

    class BlackVolSurface:
        def black_vol(self, expiry, strike):
            return 0.25

    market_state = _strict_parameter_state(sigma=None)
    market_state.vol_surface = BlackVolSurface()

    with pytest.raises(ValueError, match="sigma"):
        resolve_named_hull_white_parameter_set(
            market_state,
            parameter_set_name="usd_hw_2026_09_03",
        )


def test_hull_white_parameter_selection_preserves_direct_legacy_payload():
    from trellis.models.hull_white_parameters import resolve_hull_white_parameters

    market_state = SimpleNamespace(
        model_parameters={"mean_reversion": 0.07, "sigma": 0.012},
        model_parameter_sets={},
    )

    assert resolve_hull_white_parameters(market_state) == (0.07, 0.012)


def test_hull_white_parameter_selection_accepts_explicitly_named_set():
    from trellis.models.hull_white_parameters import resolve_hull_white_parameters

    market_state = SimpleNamespace(
        model_parameters=None,
        model_parameter_sets={
            "desk_hull_white_fit": {"mean_reversion": 0.08, "sigma": 0.011}
        },
    )

    assert resolve_hull_white_parameters(market_state) == (0.08, 0.011)


def test_hull_white_parameter_selection_skips_unrelated_sigma_payloads():
    from trellis.models.hull_white_parameters import (
        extract_hull_white_parameter_payload,
        resolve_hull_white_parameters,
    )

    market_state = SimpleNamespace(
        model_parameters={
            "model_family": "heston",
            "sigma": 0.30,
        },
        model_parameter_sets={
            "variance_gamma_equity": {
                "family": "variance_gamma",
                "sigma": 0.21,
            },
            "t17_hull_white_comparison:hull_white": {
                "model_family": "hull_white",
                "mean_reversion": 0.1,
                "sigma": 0.01,
            },
        },
    )

    payload = extract_hull_white_parameter_payload(market_state)

    assert payload is not None
    assert payload["model_family"] == "hull_white"
    assert resolve_hull_white_parameters(market_state) == (0.1, 0.01)
