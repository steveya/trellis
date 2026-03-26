# T74 Substrate Repair

## Review

`T74` should be a discovery problem for the agent, not a hand-written payoff.
The relevant substrate already existed:

- analytical kernels in `trellis.models.black`
- critic + arbiter in standard validation
- model validator in thorough validation

The actual blockers were upstream:

- the task runner did not classify `European equity call` as `european_option`
- task reruns still used a thin flat `MarketState` instead of the simulated snapshot
- the analytical cookbook and requirements were still biased toward forward-rate products
- the `analytical_black76` primitive plan always suggested forward/annuity extraction, even for vanilla equity options

## Tests First

Added focused tests for:

- `task_to_instrument_type()` recognizing `European equity call`
- `build_market_state()` exposing the richer simulated snapshot context
- `select_pricing_method_for_product_ir()` carrying analytical modeling requirements
- `analytical_black76` primitive planning for vanilla equity options
- analytical cookbook coverage for spot-to-forward Black-style pricing

## Implementation

- `task_runtime.py`
  - recognize `European equity call` and `European equity put`
  - build task market state from `resolve_market_snapshot(source="mock")`
- `quant.py`
  - propagate canonical method requirements into `PricingPlan` for ProductIR-compiled requests
- `codegen_guardrails.py`
  - make `analytical_black76` route instrument-aware:
    - vanilla equity option -> `spot + discount + vol -> forward -> Black-style kernel`
    - forward-rate products keep the schedule/annuity route
- `method_requirements.yaml`
  - add analytical requirements covering European vanilla equity Black-Scholes style pricing
- `cookbooks.yaml`
  - add an explicit analytical equity vanilla example before the forward-rate template

## Validation

Focused red-to-green slice:

- `5 passed`

Broader analytical-path regression:

- `62 passed, 1 deselected`

The deselected test is an unrelated pre-existing generic cached-module collision in `_agent/buildapayoff.py`.

## Outcome

This tranche improves the agent substrate without hand-writing the T74 payoff:

- the task is now typed correctly
- the agent sees richer analytical guidance
- the analytical primitive plan is no longer rate-product-specific by default
- task reruns use the simulated market snapshot instead of a minimal flat state
