# OpenAI Stage Model Selection

Date: 2026-03-26

## Purpose

Trellis uses stage-aware model selection so different LLM tasks can trade off speed,
cost, and reasoning depth.

This note covers the OpenAI path specifically.

## Current Trellis Defaults

OpenAI stage defaults in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py`:

- `decomposition`: `gpt-5-mini`
- `spec_design`: `gpt-5-mini`
- `code_generation`: `gpt-5-mini`
- `critic`: `gpt-5-mini`
- `model_validator`: `gpt-5`
- `reflection`: `gpt-5-mini`

## Why Code Generation Uses `gpt-5-mini`

For Trellis rerun workflows, bounded latency matters more than maximum single-call
codegen depth. The FX proving-ground reruns showed that promoting OpenAI codegen to
the heavier `gpt-5` tier made some routes time out before any code was produced.

So the default OpenAI codegen path now stays on the faster mini tier.

## How To Override

Use either the generic stage override or the provider-specific override:

- `TRELLIS_MODEL_CODE_GENERATION`
- `TRELLIS_OPENAI_MODEL_CODE_GENERATION`

Examples:

```bash
export TRELLIS_OPENAI_MODEL_CODE_GENERATION=gpt-5
```

```bash
export TRELLIS_OPENAI_MODEL_CODE_GENERATION=gpt-5.4-mini
```

```bash
export TRELLIS_OPENAI_MODEL_CODE_GENERATION=gpt-5.4
```

The provider-specific override wins over the generic one.

## Recommended Selection Rule

Use:

- fast reruns / stress tasks / thin adapters:
  - `gpt-5-mini`
- harder code synthesis where latency is acceptable:
  - `gpt-5`
- newer OpenAI GPT-5.4 tiers, if enabled in the environment and verified on your account:
  - `gpt-5.4-mini` for faster/lower-cost codegen
  - `gpt-5.4` for stronger codegen

## OpenAI Documentation

Current OpenAI docs indicate:

- `gpt-5.4` is the recommended default for most coding uses
- `gpt-5.4-mini` is the lower-latency, lower-cost GPT-5.4 option

Official references:

- [Models](https://platform.openai.com/docs/models)
- [Code generation](https://platform.openai.com/docs/guides/code-generation)

## Practical Guidance

Do not change the shared Trellis default model tier just because a newer OpenAI model
exists. First verify:

1. the model id is enabled in the target environment
2. it works under Trellis timeout settings
3. it improves rerun completion rate or code quality on representative tasks

That is especially important for `code_generation`, because a slower model can turn a
recoverable build into a timeout-only failure.
