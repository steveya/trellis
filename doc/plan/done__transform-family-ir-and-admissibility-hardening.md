# Transform Family IR And Admissibility Hardening

## Objective

Make transform pricing admit on its own lowered family contract instead of
falling back to raw vanilla-option semantics. The end state is a bounded
``TransformPricingIR`` with family-first admissibility, explicit backend
capability splitting, stable `T39`/`T40` canary recovery, and matching docs and
trace visibility.

## Why Now

`T40` narrowed the bug to a lower-layer contract problem: the semantic layer
was already correct, but transform admissibility still read upstream
option-family tags such as ``holder_max`` and ``recombining_safe``. The same
route family also needed a cleaner helper-vs-kernel split so diffusion
vanillas and stochastic-volatility transforms could share one family without
pretending they use the same backend surface.

## Scope

- ``trellis/agent/family_lowering_ir.py``
- ``trellis/agent/dsl_lowering.py``
- ``trellis/agent/route_registry.py``
- ``trellis/agent/lane_obligations.py``
- ``trellis/agent/platform_traces.py``
- ``trellis/agent/executor.py``
- ``trellis/agent/knowledge/canonical/routes.yaml``
- transform canaries ``T39`` and ``T40``
- official docs touched by the new transform family contract

## Non-goals

- new transform numerical methods beyond the existing FFT/COS/kernel surface
- widening the vanilla transform helper to unsupported model families
- removing all analytical route-local compatibility machinery

## Ticket Mirror

| Ticket | Title | Status |
| --- | --- | --- |
| QUA-771 | Transform pricing lowered family IR and admissibility hardening | Done |
| QUA-772 | Bounded transform family IR and lowering contract | Done |
| QUA-773 | Family-first transform admissibility | Done |
| QUA-774 | Backend binding split: helper vs raw kernel by capability | Done |
| QUA-775 | `T39` / `T40` canary recovery on the lowered family IR | Done |
| QUA-776 | Docs, observability, and compatibility cleanup | Done |

## Acceptance Criteria

- ``TransformPricingIR`` exists and lowers bounded transform-compatible claims.
- transform admissibility reads the lowered family contract before raw semantic
  option tags.
- transform traces surface transform-specific family summaries.
- helper-backed diffusion and raw-kernel stochastic-volatility transforms share
  the same family with explicit backend capability separation.
- `T39` and `T40` both pass on the migrated path.

## Validation

- ``/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_family_lowering_ir.py tests/test_agent/test_dsl_lowering.py tests/test_agent/test_route_registry.py tests/test_agent/test_lane_obligations.py tests/test_agent/test_platform_traces.py tests/test_agent/test_executor.py tests/test_models/test_transforms/test_single_state_diffusion.py tests/test_models/test_transforms/test_equity_option_transforms.py -q``
- ``/Users/steveyang/miniforge3/bin/python3 scripts/run_canary.py --task T39 --model gpt-5.4-mini --output task_results_t39_qua771.json``
- ``/Users/steveyang/miniforge3/bin/python3 scripts/run_canary.py --task T40 --model gpt-5.4-mini --output task_results_t40_qua771.json``

## Closeout Notes

- The transform family now carries its own typed lowering and admissibility
  surface.
- `T39` also required a deterministic analytical comparator wrapper that binds
  the checked Black76 kernels through runtime ``discount(T)`` and
  ``black_vol(T, K)`` protocols instead of regenerating an ad hoc analytical
  adapter.
