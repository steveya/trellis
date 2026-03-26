# Implementation Journey: Learning, Feedback, and Reruns

This note records how the task loop evolved from a benchmark harness into the
main mechanism by which Trellis learns what is missing, what improved, and what
the "Library Developer" should build next.

## The Original Role Of Tasks

At first, the task corpus mainly answered a narrow question:

> Can the build loop solve this pricing problem?

That was useful, but it was too narrow for a real platform. It did not by
itself answer:

- why did the task fail?
- what market data was used?
- what methods were compared?
- what did the agents learn?
- is someone fixing the failure now?

## The Task Loop Became A Development Engine

The next major step was turning task execution into a full evidence loop.

That involved:

- canonical task-run persistence
- latest and historical run stores
- platform traces and knowledge traces
- comparison-task support
- market-context capture
- issue creation for tracked failures
- UI audit trails

This changed the meaning of a failed task. A failed task was no longer just a
red line in a JSON file. It became a concrete, inspectable development signal.

## Why Reruns Matter So Much

Repeated reruns are the only honest way to tell whether the platform is
actually learning.

They let us separate:

- stale historical failures
- provider or quota noise
- market-data plumbing gaps
- knowledge gaps
- real missing pricing primitives

Without reruns, the platform only accumulates logs. With reruns, it accumulates
evidence.

## The Audit Trail Shift

One of the most important changes was making the latest run of a task explain
itself.

A good task record now answers questions like:

- why did `T104` pass?
- what methods ran?
- what were the prices and deviations?
- why did `T105` fail?
- what issues were created?
- what traces and lessons came out of the run?

That required task-run persistence, canonical latest/history stores, and UI
surfaces that read those records instead of reconstructing state from merged
batch outputs.

## The Learning Loop Became More Explicit

The runtime now has a more visible cycle:

1. run a task
2. classify the result
3. preserve observations and traces
4. promote or candidate new lessons/cookbooks
5. create issues for real missing infrastructure
6. rerun after fixes
7. compare baseline and candidate tranches

This is the beginning of a real platform-learning loop.

## What The Loop Still Does Not Do Automatically

The feedback loop is much stronger than before, but it still stops short of
true autonomous library development.

Today it can:

- identify blockers
- preserve evidence
- generate tracked work
- promote knowledge
- make reruns more informed

It cannot yet reliably:

- implement a missing foundational primitive on its own
- validate and promote that implementation end to end without supervision

That is why the "Library Developer" role is still effectively the human plus
the coding agent, even though the surrounding platform is becoming more
autonomous.

## The Main Lesson

The main lesson from this journey is that learning systems are only as good as
their auditability.

Trellis improved when it stopped treating learning as:

- "store another lesson"

and started treating it as:

- run
- trace
- classify
- promote
- rerun
- compare

That is a much stronger development loop.

## Why This Matters For The Next Milestones

The next milestones in autonomous library development rely on this loop:

- rerun stale failures until the current failure buckets are trustworthy
- eliminate provider/config noise
- close remaining mock market-data gaps
- pick real missing primitives from blocker reports
- promote reusable knowledge from successful reruns

Those are not side tasks. They are the mechanism by which the platform gets
from "interesting prototype" to "guarded self-improving pricing system."
