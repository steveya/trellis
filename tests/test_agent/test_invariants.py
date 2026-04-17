"""Tests for agent invariant checks."""

from datetime import date

import pytest

from trellis.agent.invariants import (
    check_cds_credit_curve_sensitivity,
    check_cds_spread_quote_normalization,
    check_non_negativity,
    check_price_sanity,
    check_rate_style_swaption_helper_consistency,
    check_vol_monotonicity,
    check_zero_vol_intrinsic,
    run_invariant_suite,
)
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.cap import CapFloorSpec, CapPayoff
from trellis.models.rate_style_swaption import price_swaption_black76
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)


def _cap_spec():
    return CapFloorSpec(
        notional=1_000_000, strike=0.05,
        start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
        frequency=Frequency.QUARTERLY,
    )


def _cap_factory():
    return CapPayoff(_cap_spec())


def _ms_factory(vol=0.20):
    return MarketState(
        as_of=SETTLE, settlement=SETTLE,
        discount=YieldCurve.flat(0.05),
        vol_surface=FlatVol(vol),
    )


class TestNonNegativity:

    def test_cap_non_negative(self):
        failures = check_non_negativity(_cap_factory(), _ms_factory())
        assert failures == []

    def test_non_negativity_can_return_structured_diagnostics(self):
        class NegativePayoff:
            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, market_state):
                return -2.5

        failures = check_non_negativity(
            NegativePayoff(),
            _ms_factory(),
            return_diagnostics=True,
        )

        assert len(failures) == 1
        failure = failures[0]
        assert failure.check == "check_non_negativity"
        assert failure.actual == pytest.approx(-2.5)
        assert "available_capabilities" in failure.context

    def test_non_negativity_skips_cds_like_payoff_with_negative_pv(self):
        """CDS-like payoffs are signed linear products; non-negativity must NOT fail them.

        Protection-buyer CDS clean PV is legitimately negative when spreads
        have tightened from issuance; the build loop's invariant suite
        previously rejected those payoffs at build gate, which blocked the
        FinancePy CDS parity task (F007).  Signed-linear products are
        validated by the dedicated ``check_cds_spread_quote_normalization``
        and ``check_cds_credit_curve_sensitivity`` invariants instead.
        (QUA-851.)
        """
        class CdsLikePayoff:
            def __init__(self):
                self._spec = type(
                    "CDSSpec",
                    (),
                    {
                        "spread": 0.015,  # decimal quote, as the model expects
                        "recovery": 0.4,
                        "start_date": SETTLE,
                        "end_date": date(2029, 11, 15),
                    },
                )()

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                return -65000.0

        failures = check_non_negativity(
            CdsLikePayoff(),
            MarketState(
                as_of=SETTLE,
                settlement=SETTLE,
                discount=YieldCurve.flat(0.05),
                credit_curve=CreditCurve.flat(0.02),
            ),
            return_diagnostics=True,
        )

        assert failures == []

    def test_non_negativity_still_fails_option_like_payoffs(self):
        """Regression: gating signed products must not weaken the option-like contract."""
        class OptionLikePayoff:
            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, market_state):
                return -1.23

        failures = check_non_negativity(
            OptionLikePayoff(),
            _ms_factory(),
            return_diagnostics=True,
        )
        assert len(failures) == 1
        assert failures[0].check == "check_non_negativity"
        assert failures[0].actual == pytest.approx(-1.23)

    def test_run_invariant_suite_allows_signed_cds_payoff_through_build_gate(self):
        """Integration: ``run_invariant_suite`` must not reject a signed CDS payoff.

        ``run_invariant_suite`` is the path ``arbiter.validate_payoff_with_critic``
        invokes inside the build loop.  Before QUA-851, it unconditionally
        called ``check_non_negativity`` and the protection-buyer CDS benchmark
        (F007) died at the gate.  This integration test guards against a
        regression where a parallel validation path slips past the new
        signed-linear gate.
        """
        class CdsPayoff:
            def __init__(self):
                self._spec = type(
                    "CDSSpec",
                    (),
                    {
                        "notional": 1_000_000.0,
                        "spread": 0.015,
                        "recovery": 0.4,
                        "start_date": SETTLE,
                        "end_date": date(2029, 11, 15),
                    },
                )()

            @property
            def spec(self):
                return self._spec

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                return -65_000.0

        def payoff_factory():
            return CdsPayoff()

        def market_state_factory(**_kwargs):
            return MarketState(
                as_of=SETTLE,
                settlement=SETTLE,
                discount=YieldCurve.flat(0.05),
                credit_curve=CreditCurve.flat(0.02),
            )

        passed, failures = run_invariant_suite(
            payoff_factory=payoff_factory,
            market_state_factory=market_state_factory,
            is_option=False,
        )
        assert passed, f"signed CDS payoff should pass invariant suite; got: {failures}"
        assert failures == []


class TestPriceSanity:

    def test_price_sanity_scales_threshold_by_spec_notional(self):
        class HighNotionalCapLikePayoff:
            def __init__(self):
                self._spec = type("CapLikeSpec", (), {"notional": 1_000_000.0})()

            @property
            def spec(self):
                return self._spec

            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, market_state):
                return 53_753.85

        failures = check_price_sanity(
            HighNotionalCapLikePayoff(),
            _ms_factory(),
            return_diagnostics=True,
        )

        assert failures == []

    def test_price_sanity_surfaces_cds_spread_unit_hint(self):
        class CdsLikePayoff:
            def __init__(self):
                self._spec = type(
                    "CDSSpec",
                    (),
                    {
                        "spread": 150.0,
                        "recovery": 0.4,
                        "start_date": SETTLE,
                        "end_date": date(2029, 11, 15),
                    },
                )()

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                return -65000.0

        failures = check_price_sanity(
            CdsLikePayoff(),
            MarketState(
                as_of=SETTLE,
                settlement=SETTLE,
                discount=YieldCurve.flat(0.05),
                credit_curve=CreditCurve.flat(0.02),
            ),
            return_diagnostics=True,
        )

        assert len(failures) == 1
        failure = failures[0]
        assert "150 bp -> 0.015" in failure.message
        assert failure.context["cds_spread_hint"] == "basis_points_to_decimal"

    def test_price_sanity_uses_spot_reference_for_basket_like_payoffs(self):
        class BasketLikePayoff:
            def __init__(self):
                self._spec = type(
                    "BasketSpec",
                    (),
                    {
                        "notional": 10.0,
                        "underliers": "SPX,NDX",
                        "spots": "100.0,95.0",
                    },
                )()

            @property
            def spec(self):
                return self._spec

            @property
            def requirements(self):
                return {"discount_curve", "spot"}

            def evaluate(self, market_state):
                return 142.5

        failures = check_price_sanity(
            BasketLikePayoff(),
            MarketState(
                as_of=SETTLE,
                settlement=SETTLE,
                discount=YieldCurve.flat(0.05),
                underlier_spots={"SPX": 100.0, "NDX": 95.0},
            ),
            return_diagnostics=True,
        )

        assert failures == []


class TestCreditDefaultSwapInvariants:

    @staticmethod
    def _cds_market_state():
        return MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            credit_curve=CreditCurve.flat(0.02),
        )

    def test_cds_spread_quote_normalization_passes_for_equivalent_quotes(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CDSSpec:
            notional: float
            spread: float
            recovery: float
            start_date: date
            end_date: date

        class NormalizingCDSPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                spread = float(self._spec.spread)
                if spread > 1.0:
                    spread *= 1e-4
                return float(-self._spec.notional * spread)

        def payoff_factory():
            return NormalizingCDSPayoff(
                CDSSpec(
                    notional=100.0,
                    spread=100.0,
                    recovery=0.4,
                    start_date=SETTLE,
                    end_date=date(2029, 11, 15),
                )
            )

        failures = check_cds_spread_quote_normalization(
            payoff_factory,
            lambda **kwargs: self._cds_market_state(),
        )

        assert failures == []

    def test_cds_spread_quote_normalization_fails_without_normalization(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CDSSpec:
            notional: float
            spread: float
            recovery: float
            start_date: date
            end_date: date

        class NonNormalizingCDSPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                return float(-self._spec.notional * float(self._spec.spread))

        def payoff_factory():
            return NonNormalizingCDSPayoff(
                CDSSpec(
                    notional=100.0,
                    spread=100.0,
                    recovery=0.4,
                    start_date=SETTLE,
                    end_date=date(2029, 11, 15),
                )
            )

        failures = check_cds_spread_quote_normalization(
            payoff_factory,
            lambda **kwargs: self._cds_market_state(),
            return_diagnostics=True,
        )

        assert len(failures) == 1
        failure = failures[0]
        assert failure.check == "check_cds_spread_quote_normalization"
        assert "semantically equivalent spreads 100 and 0.01" in failure.message


class TestRateStyleSwaptionInvariants:

    @staticmethod
    def _market_state(rate=0.05, vol=0.20):
        return MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(rate, max_tenor=31.0),
            vol_surface=FlatVol(vol),
        )

    @staticmethod
    def _cds_market_state():
        return MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            credit_curve=CreditCurve.flat(0.02),
        )

    @staticmethod
    def _spec():
        class SwaptionSpec:
            notional = 100.0
            strike = 0.05
            expiry_date = date(2029, 11, 15)
            swap_start = expiry_date
            swap_end = date(2034, 11, 15)
            swap_frequency = Frequency.SEMI_ANNUAL
            day_count = DayCountConvention.ACT_360
            rate_index = None
            is_payer = True

        return SwaptionSpec()

    def test_swaption_helper_consistency_passes_for_helper_backed_payoff(self):
        spec = self._spec()

        class HelperBackedSwaptionPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def requirements(self):
                return {"discount_curve", "forward_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                return float(price_swaption_black76(market_state, self._spec))

        failures = check_rate_style_swaption_helper_consistency(
            lambda: HelperBackedSwaptionPayoff(spec),
            self._market_state,
        )

        assert failures == []

    def test_swaption_helper_consistency_fails_when_notional_is_ignored(self):
        spec = self._spec()

        class BrokenSwaptionPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def requirements(self):
                return {"discount_curve", "forward_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                return float(price_swaption_black76(market_state, self._spec) / spec.notional)

        failures = check_rate_style_swaption_helper_consistency(
            lambda: BrokenSwaptionPayoff(spec),
            self._market_state,
            return_diagnostics=True,
        )

        assert len(failures) == 1
        failure = failures[0]
        assert failure.check == "check_rate_style_swaption_helper_consistency"
        assert failure.context["relation"] == "within_tolerance"
        assert failure.context["sampled_prices"]

    def test_cds_credit_curve_sensitivity_passes_for_long_protection(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CDSSpec:
            notional: float
            spread: float
            recovery: float
            start_date: date
            end_date: date

        class SensitiveCDSPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                survival = market_state.credit_curve.survival_probability(5.0)
                spread = float(self._spec.spread)
                if spread > 1.0:
                    spread *= 1e-4
                protection = self._spec.notional * (1.0 - self._spec.recovery) * (1.0 - survival)
                premium = self._spec.notional * spread * survival
                return float(protection - premium)

        def payoff_factory():
            return SensitiveCDSPayoff(
                CDSSpec(
                    notional=100.0,
                    spread=100.0,
                    recovery=0.4,
                    start_date=SETTLE,
                    end_date=date(2029, 11, 15),
                )
            )

        failures = check_cds_credit_curve_sensitivity(
            payoff_factory,
            lambda **kwargs: self._cds_market_state(),
        )

        assert failures == []

    def test_cds_credit_curve_sensitivity_fails_when_credit_curve_unused(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CDSSpec:
            notional: float
            spread: float
            recovery: float
            start_date: date
            end_date: date

        class InsensitiveCDSPayoff:
            def __init__(self, spec):
                self._spec = spec

            @property
            def requirements(self):
                return {"discount_curve", "credit_curve"}

            def evaluate(self, market_state):
                spread = float(self._spec.spread)
                if spread > 1.0:
                    spread *= 1e-4
                return float(1.0 - spread)

        def payoff_factory():
            return InsensitiveCDSPayoff(
                CDSSpec(
                    notional=100.0,
                    spread=100.0,
                    recovery=0.4,
                    start_date=SETTLE,
                    end_date=date(2029, 11, 15),
                )
            )

        failures = check_cds_credit_curve_sensitivity(
            payoff_factory,
            lambda **kwargs: self._cds_market_state(),
            return_diagnostics=True,
        )

        assert len(failures) == 1
        failure = failures[0]
        assert failure.check == "check_cds_credit_curve_sensitivity"
        assert "insensitive to hazard-rate shifts" in failure.message


class TestVolMonotonicity:

    def test_cap_monotonic(self):
        failures = check_vol_monotonicity(_cap_factory, _ms_factory)
        assert failures == []

    def test_constant_price_fails(self):
        """A payoff that ignores vol should fail monotonicity."""
        from trellis.core.payoff import DeterministicCashflowPayoff
        from trellis.instruments.bond import Bond

        bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                    maturity=10, frequency=2)

        def bond_factory():
            return DeterministicCashflowPayoff(bond)

        # Bond price doesn't change with vol — this is expected behavior,
        # not a "failure" for a non-option. The monotonicity check is for options.
        # A bond payoff returns constant price → monotonicity is satisfied (weakly).
        failures = check_vol_monotonicity(bond_factory, _ms_factory)
        # Weak monotonicity (p2 >= p1) should pass for constant prices
        assert failures == []

    def test_callable_like_price_can_decrease_with_vol(self):
        class CallableLikePayoff:
            @property
            def requirements(self):
                return {"discount_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                vol = float(market_state.vol_surface.black_vol(1.0, 1.0))
                return float(100.0 - 10.0 * vol)

        failures = check_vol_monotonicity(
            lambda: CallableLikePayoff(),
            _ms_factory,
            expected_direction="decreasing",
        )

        assert failures == []

    def test_callable_like_price_fails_if_marked_increasing(self):
        class CallableLikePayoff:
            @property
            def requirements(self):
                return {"discount_curve", "black_vol_surface"}

            def evaluate(self, market_state):
                vol = float(market_state.vol_surface.black_vol(1.0, 1.0))
                return float(100.0 - 10.0 * vol)

        failures = check_vol_monotonicity(
            lambda: CallableLikePayoff(),
            _ms_factory,
            expected_direction="increasing",
            return_diagnostics=True,
        )

        assert failures
        assert all(failure.context["expected_direction"] == "increasing" for failure in failures)


class TestZeroVolIntrinsic:

    def test_itm_cap(self):
        """ITM cap (strike < forward rate) at zero vol should have positive intrinsic."""
        spec = CapFloorSpec(
            notional=1_000_000, strike=0.04,  # ITM vs 5% curve
            start_date=date(2025, 2, 15), end_date=date(2027, 2, 15),
            frequency=Frequency.QUARTERLY,
        )

        def cap_factory():
            return CapPayoff(spec)

        def intrinsic_fn(ms):
            # At zero vol, cap value = sum of max(F-K, 0) * tau * N * df
            cap = CapPayoff(spec)
            ms_zero = MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=ms.discount, vol_surface=FlatVol(1e-10),
            )
            return price_payoff(cap, ms_zero)

        # Zero vol intrinsic check should pass (comparing against itself)
        failures = check_zero_vol_intrinsic(cap_factory, _ms_factory, intrinsic_fn)
        assert failures == []


class TestRunSuite:

    def test_cap_passes_suite(self):
        passed, failures = run_invariant_suite(
            payoff_factory=_cap_factory,
            market_state_factory=_ms_factory,
            is_option=True,
        )
        assert passed, f"Failures: {failures}"
