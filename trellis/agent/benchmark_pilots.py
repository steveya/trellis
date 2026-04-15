"""Cross-corpus benchmark pilot registry.

A *pilot* is a bounded, representative subset of tasks from a benchmark corpus
that opts in to the fresh-generated methodology (QUA-864, QUA-866 boundary +
QUA-867 admission + QUA-868 scorecard).  This module is the single source of
truth for which task ids belong to which corpus's pilot so downstream callers
(the runner, the boundary enforcer, the scorecard generator, and any future
per-corpus tooling) cannot drift out of sync.

Refs: QUA-870 (epic QUA-869).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BenchmarkPilot:
    """Immutable registry entry for one benchmark corpus's pilot subset."""

    corpus: str
    task_ids: tuple[str, ...]
    execution_policy: str = "fresh_generated"
    description: str = ""


_FINANCEPY_PILOT = BenchmarkPilot(
    corpus="financepy",
    task_ids=("F001", "F002", "F003", "F007", "F009", "F012"),
    execution_policy="fresh_generated",
    description=(
        "Fresh-generated FinancePy parity pilot covering plain analytical "
        "equity, FX analytical, schedule/rates, credit conventions, barrier "
        "structure, and analytical exotic assembly."
    ),
)


PILOT_REGISTRY: Mapping[str, BenchmarkPilot] = {
    _FINANCEPY_PILOT.corpus: _FINANCEPY_PILOT,
}


def _normalize_corpus(corpus: str) -> str:
    return str(corpus or "").strip().lower()


def get_pilot(corpus: str) -> BenchmarkPilot:
    """Return the pilot registry entry for ``corpus`` or raise ``KeyError``."""
    key = _normalize_corpus(corpus)
    if key not in PILOT_REGISTRY:
        raise KeyError(f"no benchmark pilot registered for corpus {corpus!r}")
    return PILOT_REGISTRY[key]


def get_pilot_task_ids(corpus: str) -> tuple[str, ...]:
    """Return the sorted pilot task ids for ``corpus``."""
    return tuple(sorted(get_pilot(corpus).task_ids))


def is_pilot_task(corpus: str, task_id: str) -> bool:
    """Return whether ``task_id`` belongs to ``corpus``'s pilot subset."""
    normalized = str(task_id or "").strip()
    if not normalized:
        return False
    try:
        pilot = get_pilot(corpus)
    except KeyError:
        return False
    return normalized in pilot.task_ids


def pilot_corpora() -> tuple[str, ...]:
    """Return the registered pilot corpus identifiers."""
    return tuple(sorted(PILOT_REGISTRY))


__all__ = (
    "BenchmarkPilot",
    "PILOT_REGISTRY",
    "get_pilot",
    "get_pilot_task_ids",
    "is_pilot_task",
    "pilot_corpora",
)
