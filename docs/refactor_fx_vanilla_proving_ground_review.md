# FX Vanilla Proving Ground Review

Date: 2026-03-26

## Scope

This tranche implemented the reusable substrate for the first missing-primitive
proving ground:

- Garman-Kohlhagen analytical kernels for domestic-currency FX vanilla options
- FX-aware analytical planning so FX vanilla requests do not reuse the equity
  Black-style route blindly
- deterministic primitive planning and lite-review support for the FX
  analytical route

## Main changes

### Reusable pricing primitive

- Added `garman_kohlhagen_call(...)` and `garman_kohlhagen_put(...)` in
  `trellis.models.black`.
- Re-exported the kernels through `trellis.models` and package-level
  `trellis`.

The pricing identity is the standard Garman-Kohlhagen form:

- call = `S * df_foreign * N(d1) - K * df_domestic * N(d2)`
- put = `K * df_domestic * N(-d2) - S * df_foreign * N(-d1)`

Operationally the implementation is factored as:

- `forward = spot * df_foreign / df_domestic`
- `price = df_domestic * black76_(call|put)(forward, strike, sigma, T)`

That keeps the FX primitive aligned with the existing analytical kernel family.

### Route selection and planning

- `trellis.agent.quant` now applies a conservative FX-context override when the
  request text clearly indicates an FX vanilla option, for example:
  - `FX option`
  - `FX vanilla`
  - `Garman-Kohlhagen`
  - explicit currency pairs like `EURUSD`
- That override enriches the required market-data contract with:
  - `discount_curve`
  - `forward_curve`
  - `black_vol_surface`
  - `fx_rates`
  - `spot`

- `trellis.agent.codegen_guardrails` now plans a dedicated
  `analytical_garman_kohlhagen` route when the pricing plan carries FX-specific
  market-data requirements.

The route uses:

- primitives:
  - `trellis.models.black.garman_kohlhagen_call`
  - `trellis.models.black.garman_kohlhagen_put`
  - `trellis.core.date_utils.year_fraction`
- adapter obligation:
  - `map_fx_spot_and_curves_to_garman_kohlhagen_inputs`

### Deterministic review

- `trellis.agent.lite_review` now validates the FX analytical route separately
  from the equity Black-style route.
- The FX route now expects generated code to source:
  - domestic discounting from `market_state.discount`
  - foreign curve input from `market_state.forward_curve` or
    `market_state.forecast_curves`
  - volatility from `market_state.vol_surface`
  - spot from `market_state.fx_rates`, `market_state.spot`, or
    `market_state.underlier_spots`

## Validation

Focused validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_models/test_black.py \
  tests/test_agent/test_quant.py \
  tests/test_agent/test_primitive_planning.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_lite_review.py \
  tests/test_public_api_surface.py -q
```

Result:

- `80 passed`

Nearby regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_instruments/test_fx.py \
  tests/test_verification/test_analytical_pricing.py \
  tests/test_agent/test_task_runtime.py \
  tests/test_agent/test_user_defined_products.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- `53 passed, 1 deselected`

## What this tranche does not claim

- It does **not** claim canonical knowledge-corpus registration is complete.
  The route/cookbook/method-requirements YAML surfaces are still a separate
  follow-up because they live under the knowledge-agent domain.
- It does **not** claim that `E25`, `T94`, `T105`, and `T108` have already been
  rerun live and promoted. This tranche closes the deterministic substrate gap
  so those reruns are now meaningful.

## Next follow-up

The next FX-specific follow-up should be:

1. knowledge registration for the new FX route and kernels
2. live reruns of `E25`, `T94`, `T105`, and `T108`
3. before/after comparison of failure buckets and task outcomes
