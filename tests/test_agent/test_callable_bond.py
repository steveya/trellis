"""Integration test: callable bond with critic validation.

The callable bond was the instrument that exposed the need for the critic:
the agent built code that runs but has a fundamental pricing error
(comparing undiscounted cashflows to call price instead of PV).

This test verifies that the validation pipeline catches such errors
and the retry produces correct code.

Run with: pytest tests/test_agent/test_callable_bond.py -m integration -v
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pytest

from trellis.agent.builder import TRELLIS_ROOT
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol


SETTLE = date(2024, 11, 15)
MATURITY = date(2034, 11, 15)
CALLABLE_FILE = TRELLIS_ROOT / "instruments" / "_agent" / "callablebondpayoff.py"


def _has_api_key():
    import os
    from trellis.agent.config import load_env
    load_env()
    return bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )


def _straight_bond_pv(rate: float = 0.05) -> float:
    """Reference straight bond PV."""
    bond = Bond(face=100, coupon=0.05, maturity_date=MATURITY, maturity=10, frequency=2)
    ms = MarketState(as_of=SETTLE, settlement=SETTLE, discount=YieldCurve.flat(rate))
    return price_payoff(DeterministicCashflowPayoff(bond), ms, day_count=bond.day_count)


# ---------------------------------------------------------------------------
# Unit tests (no LLM)
# ---------------------------------------------------------------------------

class TestCallableBondInvariants:
    """Verify that the invariant checks would catch the known bug."""

    def test_callable_must_be_leq_straight(self):
        """At HIGH rates, callable bond must be ≤ straight bond.
        The original agent code violated this."""
        from trellis.agent.invariants import check_bounded_by_reference

        # Create a fake "bad" callable that returns straight bond + 1
        bond = Bond(face=100, coupon=0.05, maturity_date=MATURITY, maturity=10, frequency=2)

        class BadCallable:
            @property
            def requirements(self):
                return {"discount_curve"}

            def evaluate(self, ms):
                # Returns more than the straight bond — should fail bounding
                cfs = DeterministicCashflowPayoff(bond).evaluate(ms)
                return [(d, amt + 1.0) for d, amt in cfs]

        def bad_factory():
            return BadCallable()

        def ref_factory():
            return DeterministicCashflowPayoff(bond)

        def ms_factory(rate=0.05, vol=0.20):
            return MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=YieldCurve.flat(rate),
                vol_surface=FlatVol(vol),
            )

        failures = check_bounded_by_reference(
            bad_factory, ref_factory, ms_factory, relation="<=",
        )
        assert len(failures) > 0, "Bounding check should catch callable > straight"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_callable():
    if "trellis.instruments._agent.callablebondpayoff" in sys.modules:
        del sys.modules["trellis.instruments._agent.callablebondpayoff"]
    if CALLABLE_FILE.exists():
        CALLABLE_FILE.unlink()
    cache = CALLABLE_FILE.parent / "__pycache__"
    if cache.exists():
        for f in cache.glob("callablebond*"):
            f.unlink()
    yield


@pytest.mark.integration
class TestCallableBondBuild:

    def test_build_with_validation(self, clean_callable):
        """Build a callable bond with standard validation.

        The validation should catch any callable > straight bond violation.
        """
        if not _has_api_key():
            pytest.skip("No LLM API key set")

        from trellis.agent.executor import build_payoff

        PayoffCls = build_payoff(
            "Callable bond with a call schedule (Bermudan callable)",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=True,
            validation="standard",
        )

        assert PayoffCls is not None
        assert hasattr(PayoffCls, "evaluate")

    def test_callable_leq_straight_at_all_rates(self, clean_callable):
        """The key invariant: callable ≤ straight at all rate levels."""
        if not _has_api_key():
            pytest.skip("No LLM API key set")
        if not CALLABLE_FILE.exists():
            pytest.skip("No prior build")

        from trellis.agent.executor import build_payoff

        PayoffCls = build_payoff(
            "Callable bond with a call schedule (Bermudan callable)",
            {"discount_curve", "forward_curve", "black_vol_surface"},
            force_rebuild=False,
        )

        # Find and instantiate spec
        mod = sys.modules.get("trellis.instruments._agent.callablebondpayoff")
        if mod is None:
            import importlib
            mod = importlib.import_module("trellis.instruments._agent.callablebondpayoff")

        spec_cls = None
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and hasattr(obj, "__dataclass_fields__") and obj.__module__ == mod.__name__:
                spec_cls = obj
                break

        # Try to construct spec
        fields = list(spec_cls.__dataclass_fields__.keys())
        kwargs = {}
        name_map = {
            "notional": 100.0, "coupon": 0.05,
            "start_date": SETTLE, "expiry_date": MATURITY,
            "end_date": MATURITY, "maturity_date": MATURITY,
            "strike": 0.05, "is_payer": True,
        }
        for f in fields:
            if f in name_map:
                kwargs[f] = name_map[f]

        try:
            spec = spec_cls(**kwargs)
        except TypeError:
            pytest.skip(f"Cannot instantiate {spec_cls.__name__} with fields {fields}")

        # Test at multiple rate levels
        for rate in [0.03, 0.05, 0.07]:
            ms = MarketState(
                as_of=SETTLE, settlement=SETTLE,
                discount=YieldCurve.flat(rate),
                vol_surface=FlatVol(0.20),
            )

            callable_pv = price_payoff(PayoffCls(spec), ms)
            straight_pv = _straight_bond_pv(rate)

            assert callable_pv <= straight_pv + 0.5, (
                f"At rate={rate:.0%}: callable ({callable_pv:.2f}) > "
                f"straight ({straight_pv:.2f})"
            )
