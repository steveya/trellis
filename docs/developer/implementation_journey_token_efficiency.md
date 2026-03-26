# Implementation Journey: Minimizing Token Waste

This note records how token pressure changed the architecture of Trellis.

The token-efficiency work started as a practical concern, but it ended up
forcing the platform to become more disciplined about where LLM reasoning was
actually necessary.

## Why Token Work Became Important

Once realistic task batches and stress runs were in place, token usage stopped
being an abstract cost. It became obvious that repeated tasks could burn large
amounts of tokens because each run could trigger several stages:

- decomposition
- spec design
- code generation
- critic
- model validator
- reflection

Comparison tasks amplified this because one task could fan out into several
method-specific builds.

The core problem was not just "the prompts are long." It was:

- too many calls
- too much repeated context
- too much escalation on low-information failures
- too little visibility into where the spend was going

## Phase 1: Make Token Usage Visible

The first useful step was token telemetry.

Trellis needed to answer:

> For this task, where did the tokens actually go?

So token usage started being persisted by stage into:

- platform traces
- task-run records
- batch summaries

This turned cost control into something measurable rather than anecdotal.

## Phase 2: Stage-Aware Model Hierarchies

The next step was to stop using one model shape for every stage.

The system now uses stage-aware defaults for both Anthropic and OpenAI flows.
That means:

- lighter models for decomposition, critic, and reflection
- stronger models reserved for code generation and model validation

This was one of the simplest changes with immediate leverage because it aligned
model spend with stage importance.

## Phase 3: Token Budgets

Once token usage was visible, budgets became possible.

The key idea was that a failed task should sometimes stop with:

> this task exceeded budget

instead of quietly consuming more tokens and leaving no explicit stop reason.

This made batch reruns more controlled and made cost an operational constraint
the platform could reason about.

## Phase 4: Prompt Compression

The next layer was shrinking the prompt surface without removing important
context.

That included:

- limiting lessons, principles, and requirement payloads
- truncating large cookbook/import blocks
- recording compact vs expanded prompt sizes
- using compact shared-knowledge views by default

The important lesson here was that "more memory" and "more prompt text" are not
the same thing.

## Phase 5: Compact-First Prompt Surfaces

After prompt compression, the next step was to change the default prompt shape.

First attempts now use:

- compact knowledge
- compact route cards
- truncated references

Retries expand only when the first pass actually failed. This was a better use
of context because many tasks did not need the full plan on attempt one.

## Phase 6: Deterministic-First Review

The biggest conceptual shift came when token efficiency stopped being only a
prompt problem.

The platform started skipping expensive reviewer stages when deterministic gates
already knew enough. For low-risk supported routes, Trellis can now avoid
spending tokens on critic/model-validator LLM review by default.

That was the point where token work clearly became architectural work.

## Phase 7: Failure-Type-Aware Retry Prompts

The next improvement was to stop treating all retries as the same.

Now the retry surface depends on why the previous attempt failed:

- import failure -> import-repair card, no reference excerpts
- semantic failure -> semantic-repair card, small reference surface
- later validation failure -> expanded builder context and full generation plan

This matters because many failed attempts do not need broader context. They
need narrower repair guidance.

## What Worked Best

The most valuable token improvements were not the flashiest ones.

The best gains came from:

- visibility
- staged model selection
- compact-first prompts
- deterministic skip paths
- failure-specific retries

In other words, Trellis got cheaper by becoming more explicit about what each
stage was actually for.

## What Still Needs To Happen

The next efficiency steps are even more structural:

- prompt surface minimization through smaller route and primitive cards
- deterministic-first routing and lite review
- toolization of primitive assembly and validation
- memory distillation and caching

Those are the changes that should move the platform from:

- "use fewer tokens per call"

to:

- "need fewer calls and less context overall"

## The Main Lesson

The deepest lesson from the token-efficiency journey is this:

Prompt engineering helps, but architecture matters more.

The real savings came when the platform got better at answering:

- what is deterministic?
- what is ambiguous?
- what context is actually necessary for this specific retry?

That is why token work ended up improving not just cost, but the design of the
whole system.
