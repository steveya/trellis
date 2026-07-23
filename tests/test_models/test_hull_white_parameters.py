"""Tests for typed Hull-White model-parameter selection."""

from types import SimpleNamespace


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
