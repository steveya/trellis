# M1.7 Extension: Route-Card and Retry-Focused Prompts

Date: 2026-03-25

## Goal

Reduce builder prompt size with prompt engineering, not by weakening validation,
but by making first attempts cheaper:

- compact route card on attempt 1
- fuller generation-plan context only on retries
- truncated reference excerpts on attempt 1

## What Changed

### 1. Compact route-card prompting

The builder prompt now uses a compact structured route card on the first
generation attempt instead of the full rendered generation plan.

The compact route card includes:

- method family
- instrument type
- primitive route and engine family
- required primitives and adapters
- primary modules to inspect
- post-build test targets

### 2. Retry-time prompt expansion

When a build retries, the prompt escalates from the compact route card to the
full structured generation plan.

This keeps the first pass cheap while preserving the ability to inject richer
context when the initial attempt fails.

### 3. Compact reference excerpts

The builder prompt now truncates large reference source blocks on compact
attempts and caps how many reference modules are included. Expanded retries keep
the fuller reference surface.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_llm_guards.py -q
```

Result:

- `28 passed`

## Residual Limits

- Retry feedback still appends the full validation text block; it is smaller
  than before only because the base prompt is smaller.
- Expanded retries still use the full generation plan rather than a failure-type
  specific expansion policy.
- Reference excerpts are character-capped, not semantic excerpts yet.

## Next Step

The next prompt-efficiency step should make retry prompts failure-type aware:
for example, semantic-failure retries should expand only the route/primitive
parts that matter, not all builder context.
