"""Analytics cookbook — teaches the agent how to compute measures beyond price.

The agent reads this when the user asks for analytics (OAS, Greeks, scenarios).
It mirrors the pricing cookbooks but for the analytics layer.

The cookbook is injected into the agent's prompt when:
- The user asks for analytics, not just price
- The agent needs to produce a risk report or scenario analysis
- The user provides a market price (implies OAS/spread computation)
"""

from __future__ import annotations


ANALYTICS_COOKBOOK = '''\
## Analytics Framework

Trellis has a composable analytics engine. Instead of computing just price,
you can request any combination of measures via `session.analyze()`.

### Available Measures

| Measure | String key | Parameters | Returns |
|---------|-----------|------------|---------|
| Price | `"price"` | — | float (PV) |
| DV01 | `"dv01"` | `bump_bps=1.0` | float ($ per 1bp) |
| Duration | `"duration"` | `bump_bps=1.0` | float (years) |
| Convexity | `"convexity"` | `bump_bps=10.0` | float (years²) |
| Vega | `"vega"` | `bump_pct=1.0` | float ($ per 1% vol) |
| Key Rate Durations | `"krd"` | `tenors=(2,5,10,30)`, `bump_bps=25` | dict {tenor: krd} |
| OAS | `"oas"` | `market_price` (required) | float (bps) |
| Z-Spread | `"z_spread"` | `market_price` (required) | float (bps) |
| Scenario P&L | `"scenario_pnl"` | `shifts_bps=(-100,+100,+200)` | dict {shift: pnl} |

### Three Ways to Specify Measures

```python
# 1. Simple strings (defaults)
result = session.analyze(payoff, measures=["price", "dv01", "vega"])

# 2. Dicts with parameters
result = session.analyze(payoff, measures=[
    "price",
    {"oas": {"market_price": 95.0}},
    {"key_rate_durations": {"tenors": (2, 5, 10, 30), "bump_bps": 25}},
    {"scenario_pnl": {"shifts_bps": (-100, -50, +50, +100, +200)}},
])

# 3. Measure objects (full control)
from trellis.analytics.measures import KRD, OAS, ScenarioPnL
result = session.analyze(payoff, measures=[
    "price",
    KRD(tenors=(1, 2, 3, 5, 7, 10, 20, 30), bump_bps=10),
    OAS(market_price=93.0),
    ScenarioPnL(shifts_bps=(-200, -100, -50, +50, +100, +200)),
])
```

### Accessing Results

```python
result.price                 # attribute access
result["dv01"]               # dict access
result.to_dict()             # all measures as dict
result.key_rate_durations    # dict {tenor: krd}
result.scenario_pnl          # dict {shift_bps: pnl}
```

### Book-Level Analytics

```python
book_result = session.analyze(book, measures=["price", "dv01", "duration"])
book_result.total_mv         # sum(price * notional)
book_result.book_dv01        # notional-weighted DV01
book_result.book_duration    # MV-weighted average duration
book_result.to_dataframe()   # one row per position
book_result["10Y"].price     # single position access
```

### When to Use Which Measure

- User asks for "price" → `["price"]`
- User asks for "risk" or "Greeks" → `["price", "dv01", "duration", "convexity"]`
- User asks for "spread" or "OAS" → `["price", {"oas": {"market_price": X}}]`
- User asks for "scenario" or "stress test" → `["price", {"scenario_pnl": {"shifts_bps": [...]}}]`
- User asks for "KRD" or "key rate" → `["price", {"krd": {"tenors": [...]}}]`
- User asks for "vega" or "vol sensitivity" → `["price", "vega"]`
- User asks for "full analytics" → all of the above

### Important Notes

1. OAS and Z-Spread require `market_price` — if the user provides one, include OAS.
2. KRD tenors should match the curve tenors for accurate results.
3. Vega is negative for callable bonds (higher vol → more call option value → lower price).
4. Duration is shorter for callable bonds than for straight bonds at the same maturity.
5. Convexity is negative for callable bonds (price-yield curve bends the wrong way).
'''


MEASURE_DESCRIPTIONS = {
    "price": "Present value of the instrument",
    "dv01": "Dollar sensitivity to a 1bp parallel rate shift",
    "duration": "Modified duration in years (rate sensitivity)",
    "convexity": "Second-order rate sensitivity in years²",
    "vega": "Sensitivity to a 1% absolute vol bump",
    "key_rate_durations": "Per-tenor rate sensitivities",
    "krd": "Per-tenor rate sensitivities (alias for key_rate_durations)",
    "oas": "Option-adjusted spread over the treasury curve (requires market_price)",
    "z_spread": "Parallel spread ignoring optionality (requires market_price)",
    "scenario_pnl": "P&L under user-defined rate shifts",
}


def get_analytics_cookbook() -> str:
    """Return the full analytics cookbook for prompt injection."""
    return ANALYTICS_COOKBOOK


def get_measure_list() -> str:
    """Return a short list of available measures for the agent."""
    lines = ["Available analytics measures:"]
    for name, desc in MEASURE_DESCRIPTIONS.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
