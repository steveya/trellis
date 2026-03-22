"""Cookbook patterns for evaluate() — one per pricing method.

Each cookbook is a complete, working evaluate() body template.
The builder agent adapts it for the specific instrument.
The quant agent's PricingPlan.method selects which cookbook to inject.
"""

from __future__ import annotations

COOKBOOKS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Pattern 1: Analytical (Black76 on forward rates)
# Used for: caps, floors, swaptions, European options on forwards
# Returns: Cashflows (price_payoff handles discounting)
# ---------------------------------------------------------------------------

COOKBOOKS["analytical"] = '''\
## Cookbook: Analytical (Black76)
Use this pattern for European options on forward rates.

```python
def evaluate(self, market_state):
    from trellis.core.payoff import Cashflows
    from trellis.core.date_utils import generate_schedule, year_fraction
    from trellis.models.black import black76_call, black76_put

    spec = self._spec
    fwd_curve = market_state.forecast_forward_curve(spec.rate_index)

    schedule = generate_schedule(spec.start_date, spec.end_date, spec.frequency)
    starts = [spec.start_date] + schedule[:-1]

    cashflows = []
    for p_start, p_end in zip(starts, schedule):
        if p_end <= market_state.settlement:
            continue
        tau = year_fraction(p_start, p_end, spec.day_count)
        t_start = year_fraction(market_state.settlement, p_start, spec.day_count)
        t_end = year_fraction(market_state.settlement, p_end, spec.day_count)
        t_start = max(t_start, 1e-6)

        F = fwd_curve.forward_rate(t_start, t_end)
        sigma = market_state.vol_surface.black_vol(t_start, spec.strike)

        # >>> INSTRUMENT-SPECIFIC: choose black76_call or black76_put <<<
        undiscounted = spec.notional * tau * black76_call(F, spec.strike, sigma, t_start)
        cashflows.append((p_end, float(undiscounted)))

    return Cashflows(cashflows)
```
'''

# ---------------------------------------------------------------------------
# Pattern 2: Rate Tree (backward induction)
# Used for: callable bonds, puttable bonds, Bermudan swaptions
# Returns: PresentValue (tree handles discounting internally)
# ---------------------------------------------------------------------------

COOKBOOKS["rate_tree"] = '''\
## Cookbook: Rate Tree (backward induction)
Use this pattern for instruments with early exercise decisions.
The tree handles discounting — return PresentValue.

```python
def evaluate(self, market_state):
    from trellis.core.payoff import PresentValue
    from trellis.core.date_utils import generate_schedule, year_fraction
    from trellis.models.trees.binomial import BinomialTree
    from trellis.models.trees.backward_induction import backward_induction

    spec = self._spec
    T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)
    r = float(market_state.discount.zero_rate(T / 2))
    sigma = float(market_state.vol_surface.black_vol(T / 2, r))

    n_steps = min(200, max(50, int(T * 50)))
    tree = BinomialTree.crr(r, T, n_steps, r, sigma)

    # >>> INSTRUMENT-SPECIFIC: define cashflows at each time step <<<
    # Example: coupon bond pays coupon_rate * notional * dt at each step
    dt = T / n_steps

    def payoff_at_node(step, node):
        \"\"\"Terminal payoff at maturity.\"\"\"
        # >>> Fill in: bond notional + final coupon, option payoff, etc. <<<
        return spec.notional  # example: return par at maturity

    def exercise_value(step, node, tree):
        \"\"\"Exercise value at a callable/puttable date.\"\"\"
        # >>> Fill in: call price, put price, etc. <<<
        return spec.notional  # example: callable at par

    # Determine which steps allow exercise
    # >>> INSTRUMENT-SPECIFIC: map call/put dates to step indices <<<
    exercise_steps = []  # list of step indices

    price = backward_induction(
        tree, payoff_at_node, r, "bermudan", exercise_steps, exercise_value
    )

    return PresentValue(price)
```
'''

# ---------------------------------------------------------------------------
# Pattern 3: Monte Carlo (path simulation)
# Used for: barrier options, Asian options, lookbacks, path-dependent exotics
# Returns: PresentValue (MC computes discounted expected payoff)
# ---------------------------------------------------------------------------

COOKBOOKS["monte_carlo"] = '''\
## Cookbook: Monte Carlo (path simulation)
Use this pattern for path-dependent instruments.
The MC engine computes the discounted expected payoff — return PresentValue.

```python
def evaluate(self, market_state):
    from trellis.core.payoff import PresentValue
    from trellis.core.date_utils import year_fraction
    from trellis.models.monte_carlo.engine import MonteCarloEngine
    from trellis.models.processes.gbm import GBM
    import numpy as np

    spec = self._spec
    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    r = float(market_state.discount.zero_rate(T))
    sigma = float(market_state.vol_surface.black_vol(T, spec.strike))

    process = GBM(mu=r, sigma=sigma)
    engine = MonteCarloEngine(process, n_paths=50000, n_steps=200, seed=42, method="exact")

    def payoff_fn(paths):
        \"\"\"Compute payoff from simulated paths.
        paths: ndarray of shape (n_paths, n_steps + 1)
        Return: ndarray of shape (n_paths,) — payoff per path
        \"\"\"
        S_T = paths[:, -1]
        # >>> INSTRUMENT-SPECIFIC: compute payoff from paths <<<
        # Example European call: max(S_T - K, 0)
        return np.maximum(S_T - spec.strike, 0) * spec.notional / spec.spot

    result = engine.price(spec.spot, T, payoff_fn, discount_rate=r)
    return PresentValue(result["price"])
```
'''

# ---------------------------------------------------------------------------
# Pattern 4: Copula (correlated defaults)
# Used for: CDO tranches, nth-to-default, basket credit derivatives
# Returns: PresentValue
# ---------------------------------------------------------------------------

COOKBOOKS["copula"] = '''\
## Cookbook: Copula (correlated defaults)
Use this pattern for portfolio credit instruments.

```python
def evaluate(self, market_state):
    from trellis.core.payoff import PresentValue
    from trellis.core.date_utils import year_fraction
    from trellis.models.copulas.factor import FactorCopula
    import numpy as np

    spec = self._spec
    T = year_fraction(market_state.settlement, spec.end_date, spec.day_count)

    # Get default probabilities from credit curve
    # >>> INSTRUMENT-SPECIFIC: may need per-name hazard rates <<<
    marginal_prob = 1 - float(market_state.credit_curve.survival_probability(T))

    copula = FactorCopula(n_names=spec.n_names, correlation=spec.correlation)
    losses, probs = copula.loss_distribution(marginal_prob)

    # >>> INSTRUMENT-SPECIFIC: compute tranche expected loss <<<
    # Example: mezzanine tranche [attachment, detachment]
    tranche_el = 0.0
    for n_defaults, prob in zip(losses, probs):
        portfolio_loss = n_defaults / spec.n_names  # as fraction
        tranche_loss = max(0, min(portfolio_loss - spec.attachment,
                                   spec.detachment - spec.attachment))
        tranche_el += prob * tranche_loss

    # PV = notional * (spread * annuity - expected_loss)
    df = float(market_state.discount.discount(T))
    tranche_pv = spec.notional * (tranche_el * df)

    return PresentValue(tranche_pv)
```
'''

# ---------------------------------------------------------------------------
# Pattern 5: Waterfall (structured products / MBS / ABS)
# Used for: MBS pass-throughs, CMOs, CLOs, ABS
# Returns: Cashflows or PresentValue depending on complexity
# ---------------------------------------------------------------------------

COOKBOOKS["waterfall"] = '''\
## Cookbook: Waterfall (structured products)
Use this pattern for MBS, ABS, CLO tranching.

```python
def evaluate(self, market_state):
    from trellis.core.payoff import Cashflows
    from trellis.core.date_utils import year_fraction
    from trellis.models.cashflow_engine.waterfall import Waterfall, Tranche
    from trellis.models.cashflow_engine.prepayment import PSA
    from trellis.models.cashflow_engine.amortization import level_pay

    spec = self._spec

    # Generate base amortization schedule
    monthly_rate = spec.coupon / 12
    n_months = spec.term_months
    base_schedule = level_pay(spec.notional, monthly_rate, n_months)

    # Apply prepayment model
    prepay = PSA(speed=spec.psa_speed)
    adjusted = []
    balance = spec.notional
    for month, (interest, principal) in enumerate(base_schedule, 1):
        smm = prepay.smm(month)
        scheduled_principal = principal
        prepaid = balance * smm
        total_principal = scheduled_principal + prepaid
        total_principal = min(total_principal, balance)
        adjusted.append((interest, total_principal))
        balance -= total_principal
        if balance <= 0:
            break

    # >>> INSTRUMENT-SPECIFIC: set up tranches <<<
    tranches = [
        Tranche("A", spec.notional * 0.8, spec.coupon_a, subordination=0),
        Tranche("B", spec.notional * 0.2, spec.coupon_b, subordination=1),
    ]
    wf = Waterfall(tranches)
    distributions = wf.run(adjusted, period=1/12)

    # Collect cashflows for the target tranche
    cashflows = []
    # >>> INSTRUMENT-SPECIFIC: pick the right tranche, compute dates <<<
    for i, dist in enumerate(distributions):
        if spec.tranche_name in dist:
            d = dist[spec.tranche_name]
            total = d["interest"] + d["principal"]
            if total > 0:
                # Approximate payment date
                from trellis.core.date_utils import add_months
                pay_date = add_months(market_state.settlement, i + 1)
                cashflows.append((pay_date, total))

    return Cashflows(cashflows)
```
'''


def get_cookbook(method: str) -> str:
    """Return the cookbook for a pricing method, or empty string if none."""
    return COOKBOOKS.get(method, "")


def get_all_cookbooks() -> str:
    """Return all cookbooks concatenated (for general prompts)."""
    return "\n\n".join(COOKBOOKS.values())
