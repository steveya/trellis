# OpenAI Timeout / Retry Review

## Goal

Stop live OpenAI-backed task reruns from failing with opaque empty-response
parsing errors or hanging indefinitely in the first JSON generation call.

## Changes

Implemented in:

- [config.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py)
- [test_llm_guards.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_llm_guards.py)

This tranche adds:

- explicit OpenAI JSON/text timeout configuration
- bounded retry handling in the OpenAI path
- SDK retry disablement so Trellis controls retry behavior explicitly
- continued explicit errors for empty text / empty JSON / invalid JSON

## Validation

Deterministic:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_llm_guards.py
```

Broader:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_task_runtime.py -k 'not generic_cached_transform_task' \
  tests/test_agent/test_evals.py \
  tests/test_agent/test_stress_task_preflight.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_llm_guards.py \
  tests/test_data/test_resolver.py \
  tests/test_data/test_mock.py
```

## Live Check

Live rerun performed:

- `E28`: `European equity call: transform-family separation (FFT vs COS)`
- provider: OpenAI
- model: `gpt-5-mini`
- env:
  - `OPENAI_JSON_TIMEOUT_SECONDS=8`
  - `OPENAI_TEXT_TIMEOUT_SECONDS=12`
  - `OPENAI_MAX_RETRIES=0`

Observed result:

- the task returned a structured failure payload instead of the old empty-JSON
  parser crash
- `comparison_targets` and `market_context` were preserved in the result
- `cross_validation.status` was `insufficient_results`

Residual issue:

- end-to-end latency was still higher than the configured per-call timeout, so
  there is likely still an underlying SDK/network layer that needs stricter
  wall-clock enforcement if we want hard per-target deadlines
