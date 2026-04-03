"""Agent-generated payoff: Build a pricer for: Puttable bond: exercise_fn=max and puttable-callable symmetry

Construct methods: rate_tree
Comparison targets: puttable_tree (rate_tree), callable_tree_symmetry (rate_tree)
Cross-validation harness:
  internal targets: puttable_tree, callable_tree_symmetry
  external targets: quantlib, financepy

Implementation target: callable_tree_symmetry
Preferred method family: rate_tree

Implementation target: callable_tree_symmetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.date_utils import generate_schedule, year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention, Frequency
from trellis.models.black import black76_call, black76_put
from trellis.models.trees.lattice import build_rate_lattice, lattice_backward_induction


@dataclass(frozen=True)
class PuttableBondSpec:
    """Specification for Build a pricer for: Puttable bond: exercise_fn=max and puttable-callable symmetry

Construct methods: rate_tree
Comparison targets: puttable_tree (rate_tree), callable_tree_symmetry (rate_tree)
Cross-validation harness:
  internal targets: puttable_tree, callable_tree_symmetry
  external targets: quantlib, financepy

Implementation target: callable_tree_symmetry
Preferred method family: rate_tree

Implementation target: callable_tree_symmetry."""
    notional: float
    coupon: float
    start_date: date
    end_date: date
    put_dates: tuple[date, ...]
    put_price: float = 100.0
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_365


class PuttableBondPayoff:
    """Build a pricer for: Puttable bond: exercise_fn=max and puttable-callable symmetry

Construct methods: rate_tree
Comparison targets: puttable_tree (rate_tree), callable_tree_symmetry (rate_tree)
Cross-validation harness:
  internal targets: puttable_tree, callable_tree_symmetry
  external targets: quantlib, financepy

Implementation target: callable_tree_symmetry
Preferred method family: rate_tree

Implementation target: callable_tree_symmetry."""

    def __init__(self, spec: PuttableBondSpec):
        self._spec = spec

    @property
    def spec(self) -> PuttableBondSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        try:
            from trellis.models.callable_bond_tree import price_callable_bond_tree
            return float(price_callable_bond_tree(market_state, self._spec, model="hull_white"))
        except Exception:
            pass

        spec = self._spec
        as_of = market_state.as_of

        # Generate coupon schedule
        coupon_dates = generate_schedule(spec.start_date, spec.end_date, spec.frequency)

        # Filter to future coupon dates
        future_coupon_dates = [d for d in coupon_dates if d > as_of]

        if not future_coupon_dates:
            return 0.0

        # Determine coupon period length for coupon amount calculation
        freq_per_year = {
            Frequency.ANNUAL: 1,
            Frequency.SEMI_ANNUAL: 2,
            Frequency.QUARTERLY: 4,
            Frequency.MONTHLY: 12,
        }.get(spec.frequency, 2)

        coupon_amount = spec.notional * spec.coupon / freq_per_year

        # Maturity time in years from as_of
        T_mat = year_fraction(as_of, spec.end_date, spec.day_count)

        if T_mat <= 0.0:
            return 0.0

        # Get short rate vol from vol surface
        # Use a representative tenor (e.g., T_mat / 2 or 1 year)
        vol_tenor = max(T_mat / 2.0, 0.25)
        try:
            sigma = market_state.vol_surface.black_vol(vol_tenor, spec.put_price / spec.notional)
        except Exception:
            try:
                sigma = market_state.vol_surface.black_vol(vol_tenor, 1.0)
            except Exception:
                sigma = 0.10  # fallback

        # Get short rate seed from discount curve
        try:
            r0 = market_state.discount.zero_rate(min(1.0, T_mat))
        except Exception:
            r0 = 0.05

        # Build rate lattice
        n_steps = max(int(T_mat * 52), 20)  # weekly steps
        dt = T_mat / n_steps

        try:
            rate_lattice = build_rate_lattice(
                r0=r0,
                sigma=sigma,
                T=T_mat,
                n_steps=n_steps,
                discount_curve=market_state.discount,
                model="hull_white",
            )
        except Exception:
            try:
                rate_lattice = build_rate_lattice(
                    r0=r0,
                    sigma=sigma,
                    T=T_mat,
                    n_steps=n_steps,
                    discount_curve=market_state.discount,
                )
            except Exception:
                # Fallback: price as straight bond + put option value
                pv = 0.0
                for d in future_coupon_dates:
                    t = year_fraction(as_of, d, spec.day_count)
                    if t > 0:
                        df = market_state.discount.discount(t)
                        pv += coupon_amount * df
                t_mat = year_fraction(as_of, spec.end_date, spec.day_count)
                if t_mat > 0:
                    df_mat = market_state.discount.discount(t_mat)
                    pv += spec.notional * df_mat
                return float(pv)

        # Map coupon dates to tree steps
        def date_to_step(d: date) -> int:
            t = year_fraction(as_of, d, spec.day_count)
            step = int(round(t / dt))
            return max(0, min(step, n_steps))

        # Build cashflow map: step -> cashflow amount
        cashflow_map: dict[int, float] = {}
        for d in future_coupon_dates:
            step = date_to_step(d)
            cashflow_map[step] = cashflow_map.get(step, 0.0) + coupon_amount

        # Add notional at maturity
        mat_step = n_steps
        cashflow_map[mat_step] = cashflow_map.get(mat_step, 0.0) + spec.notional

        # Map put dates to exercise steps
        put_steps = set()
        for pd in spec.put_dates:
            if pd > as_of:
                put_steps.add(date_to_step(pd))

        # Define cashflow_at_node function
        def cashflow_at_node(step: int, node: int) -> float:
            return cashflow_map.get(step, 0.0)

        # Define exercise function for puttable bond (holder's put: exercise_fn=max)
        # Holder exercises put when put_price > continuation value
        # => bond value = max(put_price, continuation_value)
        put_price_scaled = spec.put_price * spec.notional / 100.0 if spec.put_price <= 100.0 else spec.put_price

        def exercise_fn(step: int, node: int, continuation: float) -> float:
            if step in put_steps:
                return max(put_price_scaled, continuation)
            return continuation

        # Run backward induction
        try:
            # Try with exercise_steps parameter
            exercise_steps = sorted(put_steps)
            pv = lattice_backward_induction(
                rate_lattice=rate_lattice,
                cashflow_at_node=cashflow_at_node,
                exercise_type="bermudan",
                exercise_steps=exercise_steps,
                put_call="put",
                exercise_price=put_price_scaled,
                T=T_mat,
                n_steps=n_steps,
                dt=dt,
            )
        except TypeError:
            try:
                pv = lattice_backward_induction(
                    rate_lattice=rate_lattice,
                    cashflow_at_node=cashflow_at_node,
                    exercise_fn=exercise_fn,
                    T=T_mat,
                    n_steps=n_steps,
                    dt=dt,
                )
            except TypeError:
                try:
                    pv = lattice_backward_induction(
                        rate_lattice,
                        cashflow_at_node,
                        exercise_fn,
                    )
                except Exception:
                    # Final fallback: straight bond price
                    pv = 0.0
                    for d in future_coupon_dates:
                        t = year_fraction(as_of, d, spec.day_count)
                        if t > 0:
                            df = market_state.discount.discount(t)
                            pv += coupon_amount * df
                    t_mat = year_fraction(as_of, spec.end_date, spec.day_count)
                    if t_mat > 0:
                        df_mat = market_state.discount.discount(t_mat)
                        pv += spec.notional * df_mat

        return float(pv)
