Task Diagnostics
=================

This page explains the canonical per-task diagnosis packet and dossier that
Trellis now writes for every task run.

Why it exists
-------------

Before the packet, diagnosis was spread across:

- the raw batch JSON from ``scripts/run_tasks.py``
- the latest/history run records in ``task_runs/``
- platform and analytical traces
- remediation summaries
- learning artifacts and lessons

That made both human and agent diagnosis expensive. The packet collects the
important pieces into one structured artifact, and the dossier renders the
same data in a readable Markdown form.

What gets written
-----------------

For each task run, Trellis writes:

- a canonical JSON packet
- a human-readable Markdown dossier
- the existing run record in ``task_runs/history`` and ``task_runs/latest``

The packet and dossier live under:

- ``task_runs/diagnostics/history/<task_id>/<run_id>.json``
- ``task_runs/diagnostics/history/<task_id>/<run_id>.md``
- ``task_runs/diagnostics/latest/<task_id>.json``
- ``task_runs/diagnostics/latest/<task_id>.md``

The batch runner also surfaces those paths on the task result itself so a
caller can jump directly to the packet after a batch finishes.

How to read one
---------------

Start with the Markdown dossier. It is ordered for diagnosis:

1. Summary
2. Primary diagnosis
3. Comparison summary, if present
4. Method outcomes
5. Trace index
6. Learning summary
7. Evidence
8. Workflow and storage paths

If the dossier is still not enough, open the JSON packet next. The packet is
the canonical structured record and should contain the same evidence in a
machine-friendly shape.

Operational use
---------------

The packet is meant to shorten the feedback loop after a pricing batch:

- ``scripts/run_tasks.py`` writes the packet paths into the batch results
- ``scripts/remediate.py --analyze-only`` can be used alongside the packet to
  bucket failures
- future diagnosis work should refine the packet schema rather than creating
  another parallel report

