"""Critic agent: reads agent-generated code and produces structured test cases.

The critic sees ONLY the generated code (not the builder's reasoning).
It outputs (concern, test_code) pairs that the arbiter executes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from trellis.agent.config import llm_generate_json


@dataclass(frozen=True)
class CriticConcern:
    """A potential issue identified by the critic."""
    description: str
    test_code: str
    severity: str = "error"  # "error" or "warning"


CRITIC_PROMPT_TEMPLATE = """\
You are a quantitative model validator reviewing agent-generated pricing code.
Your job is to find errors, not to praise. Be adversarial.

## Code to review
```python
{code}
```

## Instrument description
{description}

## What to look for
1. **Discounting errors**: Is the code double-discounting or not discounting correctly?
   - evaluate() should return UNDISCOUNTED cashflows; price_payoff() handles discounting
   - If the code discounts cashflows internally AND they get discounted again externally, that's a bug

2. **Call/exercise decision errors**: For callable/puttable instruments, does the exercise
   decision compare PRESENT VALUES (not undiscounted sums)? The issuer calls when the
   PV of remaining cashflows exceeds the call price.

3. **Missing vol dependence**: If requirements include "black_vol" but the evaluate() body
   never reads market_state.vol_surface, the option component is not being priced.

4. **Day count / schedule errors**: Are year fractions computed correctly?
   Is the schedule generation using the right frequency?

5. **Edge cases**: What happens at maturity? What if all call dates are past?
   What if the instrument is deeply in/out of the money?

## Output
Return a JSON array of concerns. Each concern:
{{
    "description": "what might be wrong",
    "test_code": "executable Python assertion (one-liner or short block)",
    "severity": "error" or "warning"
}}

The test_code will be executed with these variables available:
- `payoff`: an instance of the payoff class
- `ms`: a MarketState with flat 5% curve, 20% vol, settlement=2024-11-15
- `ms_low_rate`: MarketState with flat 3% curve
- `ms_high_rate`: MarketState with flat 7% curve
- `price_payoff`: the pricing function
- `straight_bond_pv`: PV of an equivalent straight bond at 5%

Focus on errors that would cause INCORRECT PRICES, not style issues.
Return at most 5 concerns, ordered by severity.
Return ONLY the JSON array."""


def critique(
    code: str,
    description: str,
    model: str | None = None,
) -> list[CriticConcern]:
    """Run the critic agent on generated code.

    Parameters
    ----------
    code : str
        The full Python module source.
    description : str
        What the instrument is (e.g. "Callable bond with call schedule").
    model : str or None
        LLM model to use.

    Returns
    -------
    list[CriticConcern]
    """
    prompt = CRITIC_PROMPT_TEMPLATE.format(code=code, description=description)

    try:
        data = llm_generate_json(prompt, model=model)
    except Exception:
        # If JSON parsing fails, try text mode and extract JSON
        from trellis.agent.config import llm_generate
        text = llm_generate(prompt, model=model)
        # Find JSON array in text
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            return []

    concerns = []
    if isinstance(data, list):
        for item in data:
            concerns.append(CriticConcern(
                description=item.get("description", ""),
                test_code=item.get("test_code", ""),
                severity=item.get("severity", "warning"),
            ))
    return concerns
