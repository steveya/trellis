# FX Knowledge Registration Review

Date: 2026-03-26

## Scope

This tranche completed the knowledge-layer follow-up for the FX vanilla proving
ground:

- canonical analytical cookbook guidance now includes a Garman-Kohlhagen FX
  vanilla pattern
- analytical method requirements now distinguish equity Black-style guidance
  from FX domestic/foreign discounting guidance
- analytical data contracts now include explicit domestic/foreign discounting
  inputs for Garman-Kohlhagen
- the hot-start API map now exposes the FX analytical imports and clarifies the
  foreign-curve bridge

## Main changes

### Canonical cookbook registration

Updated:

- `trellis/agent/knowledge/canonical/cookbooks.yaml`

The `analytical` cookbook now includes:

- an explicit European vanilla FX option example using:
  - `garman_kohlhagen_call`
  - `garman_kohlhagen_put`
  - `market_state.fx_rates`
  - `market_state.forecast_curves[...]`
  - domestic and foreign discount factors
- a note explaining the runtime bridge:
  - `market_state.forward_curve` is the carry/routing view
  - `market_state.forecast_curves[...]` remains the explicit foreign discount
    source

### Method requirements and contracts

Updated:

- `trellis/agent/knowledge/canonical/method_requirements.yaml`
- `trellis/agent/knowledge/canonical/data_contracts.yaml`
- `trellis/agent/knowledge/canonical/api_map.yaml`

New analytical FX guidance now states:

- Garman-Kohlhagen is the correct analytical contract for European vanilla FX
  options
- the foreign carry enters through
  `F = S0 * df_foreign / df_domestic`
- analytical FX pricing requires:
  - `market_state.discount`
  - `market_state.forecast_curves[...]`
  - `market_state.forward_curve`
  - `market_state.vol_surface`
  - `market_state.fx_rates` or `market_state.spot`

New data contract:

- `FX_DOMESTIC_FOREIGN_DISCOUNTING`

API map update:

- `trellis.models.black.garman_kohlhagen_call`
- `trellis.models.black.garman_kohlhagen_put`

## Validation

Targeted knowledge validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_cookbooks.py \
  tests/test_agent/test_knowledge_store.py \
  tests/test_agent/test_import_registry.py -q
```

Result:

- `70 passed`

Focused FX regression slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_quant.py \
  tests/test_agent/test_primitive_planning.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_task_runtime.py \
  -k 'fx or garman or T94 or E25 or T105 or T108' -q
```

Result:

- `4 passed, 70 deselected`

## Live rerun findings

Live reruns were attempted immediately after the knowledge registration.

What the reruns proved:

- the new FX knowledge is present in live request traces
- `knowledge_summary` now includes:
  - `FX_DOMESTIC_FOREIGN_DISCOUNTING`
  - the analytical cookbook method
  - the expanded analytical requirement count
- task-run persistence now records the FX knowledge artifacts correctly

What still blocked the tranche:

- OpenAI reruns for the FX tranche stalled in `spec_design` before
  `build_started`
- Anthropic reruns exposed provider/runtime noise:
  - stale Anthropic stage-model default `claude-3-5-haiku-latest`
  - invalid JSON / empty-response failures even after forcing all stages to
    `claude-sonnet-4-6`

Concrete observed rerun:

- `E25` persisted at:
  - `task_runs/latest/E25.json`
  - `task_runs/history/E25/20260326T095008308473.json`
- the latest result failed for provider-response reasons, not missing FX market
  data and not missing Garman-Kohlhagen knowledge

## Conclusion

The knowledge-registration half of `M3.4` is complete.

The remaining blocker for the FX proving-ground rerun half is now clearly
provider/runtime noise:

- stale Anthropic stage defaults
- invalid JSON / empty provider responses
- generic European-option spec-design hangs on the OpenAI path

So the next work after this tranche is not more FX cookbook registration. It is
provider/config stabilization and then another FX rerun pass.
