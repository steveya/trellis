from __future__ import annotations

from trellis.agent.semantic_track_classifier import classify_semantic_track


class TestSemanticTrackClassifier:
    def test_terminal_curve_spread_classifies_as_quoted_observable(self):
        classification = classify_semantic_track(
            "Terminal curve-spread payoff on USD_SWAP par rate 10Y minus 2Y, notional 1000000, expiry 2026-06-30",
            instrument_type="quoted_observable",
        )

        assert classification.track == "quoted_observable"
        assert classification.dynamic is False

    def test_vanilla_irs_classifies_as_static_leg(self):
        classification = classify_semantic_track(
            "Vanilla pay fixed USD IRS notional 1000000 fixed rate 4% effective 2025-06-30 maturity 2030-06-30 fixed semiannual float quarterly index SOFR",
            instrument_type="swap",
        )

        assert classification.track == "static_leg"
        assert classification.dynamic is False

    def test_period_rate_option_strip_classifies_as_static_leg(self):
        classification = classify_semantic_track(
            "Price a cap strip under the declared benchmark rates surface. Instrument class: cap. Start date: 2024-11-15. End date: 2029-11-15.",
            instrument_type="period_rate_option_strip",
        )

        assert classification.track == "static_leg"
        assert classification.dynamic is False

    def test_callable_bond_classifies_as_dynamic_wrapper_over_static_leg(self):
        classification = classify_semantic_track(
            "Issuer callable fixed coupon bond USD face 1000000 coupon 5% issue 2025-01-15 maturity 2030-01-15 call dates 2027-01-15, 2028-01-15",
            instrument_type="callable_bond",
        )

        assert classification.track == "dynamic_wrapper"
        assert classification.base_track == "static_leg"
        assert classification.dynamic is True

    def test_unknown_description_falls_back_to_payoff_expression(self):
        classification = classify_semantic_track("European call on AAPL strike 150 expiring 2025-11-15")

        assert classification.track == "payoff_expression"
        assert classification.dynamic is False
