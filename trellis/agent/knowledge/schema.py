"""Schema definitions for the Trellis knowledge system.

All knowledge types — features, decompositions, lessons, benchmarks,
failure signatures, and retrieval specs — are defined here as frozen
dataclasses.  The feature taxonomy is the core abstraction: every product
decomposes into features, and retrieval is feature-based (union, not
instrument-name lookup).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Severity bucket used for lessons, failures, and retrieval prioritization."""
    CRITICAL = "critical"   # >10x pricing error
    HIGH = "high"           # >100bp
    MEDIUM = "medium"       # 10-100bp
    LOW = "low"             # <10bp


class LessonStatus(str, Enum):
    """Lifecycle state for captured lessons inside the knowledge system."""
    CANDIDATE = "candidate"     # Just captured, not yet validated
    VALIDATED = "validated"     # Confirmed by successful test
    PROMOTED = "promoted"       # Accepted into stable lessons
    ARCHIVED = "archived"       # Superseded or merged


# ---------------------------------------------------------------------------
# Feature taxonomy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    """A single feature in the taxonomy.  Features are the atoms;
    instruments are molecules built from feature sets."""

    id: str
    description: str
    implies: tuple[str, ...] = ()       # transitive dependencies
    method_hint: str | None = None      # suggests a method, doesn't mandate
    market_data: tuple[str, ...] = ()   # required market data capabilities


# ---------------------------------------------------------------------------
# Product decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductDecomposition:
    """A product broken into features — static or LLM-generated."""

    instrument: str                             # normalised key, e.g. "callable_range_accrual"
    features: tuple[str, ...]                   # ["callable", "range_condition", ...]
    method: str                                 # primary pricing method
    method_modules: tuple[str, ...] = ()
    required_market_data: frozenset[str] = frozenset()
    modeling_requirements: tuple[str, ...] = ()
    reasoning: str = ""
    notes: str = ""                             # known complexities
    learned: bool = False                       # True if auto-discovered


@dataclass(frozen=True)
class ProductIR:
    """Typed deterministic representation of a product decomposition.

    This is a higher-level, more structured view than ``ProductDecomposition``.
    It is designed for routing, validation, and assembly-first code generation.
    The IR should remain deterministic for known products and conservative for
    unsupported composites.
    """

    instrument: str
    payoff_family: str
    payoff_traits: tuple[str, ...] = ()
    exercise_style: str = "none"
    state_dependence: str = "terminal_markov"
    schedule_dependence: bool = False
    model_family: str = "generic"
    candidate_engine_families: tuple[str, ...] = ()
    required_market_data: frozenset[str] = frozenset()
    reusable_primitives: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()
    supported: bool = True


# ---------------------------------------------------------------------------
# Retrieval filter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppliesWhen:
    """Structured filter for when a lesson / signature is relevant."""

    method: tuple[str, ...] = ()
    features: tuple[str, ...] = ()          # PRIMARY retrieval axis
    instrument: tuple[str, ...] = ()        # backward-compat, secondary
    error_signature: str | None = None      # regex on error message


# ---------------------------------------------------------------------------
# Lesson (Store C)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LessonIndex:
    """Lightweight index entry — always loaded (hot tier)."""

    id: str
    title: str
    severity: Severity
    category: str
    applies_when: AppliesWhen
    status: LessonStatus = LessonStatus.PROMOTED


@dataclass(frozen=True)
class Lesson:
    """Full lesson content — loaded on demand (warm tier)."""

    id: str
    title: str
    severity: Severity
    category: str
    applies_when: AppliesWhen

    symptom: str
    root_cause: str
    fix: str
    validation: str

    confidence: float = 1.0
    status: LessonStatus = LessonStatus.PROMOTED
    version: str = ""
    created: str = ""
    source_trace: str | None = None
    supersedes: tuple[str, ...] = ()
    derived_principle: str | None = None


# ---------------------------------------------------------------------------
# Canonical knowledge (Store A)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principle:
    """A distilled rule — always injected (hot tier)."""

    id: str                             # "P1", "P2", …
    rule: str                           # One-line rule
    derived_from: tuple[str, ...] = ()  # Lesson IDs
    category: str = ""


@dataclass(frozen=True)
class CookbookEntry:
    """A pricing-method template."""

    method: str
    template: str
    description: str = ""
    applicable_instruments: tuple[str, ...] = ()
    version: str = ""


@dataclass(frozen=True)
class DataContractEntry:
    """Data contract — unit conventions and conversion rules."""

    name: str
    method: str
    source: str
    convention: str
    typical_range: str
    model_expects: str
    conversion: str
    model_range: str
    warning: str = ""


@dataclass(frozen=True)
class MethodRequirements:
    """Per-method modelling constraints."""

    method: str
    requirements: tuple[str, ...]


# ---------------------------------------------------------------------------
# Failure signatures (Store D / retrieval index)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureSignature:
    """An indexed error pattern for retrieval."""

    pattern: str                            # regex on error message
    magnitude: str                          # "catastrophic", "significant", "minor"
    category: str
    probable_causes: tuple[str, ...] = ()   # lesson IDs
    features: tuple[str, ...] = ()          # related features
    diagnostic_hint: str = ""


# ---------------------------------------------------------------------------
# Benchmarks (Store D)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark test case with reference value."""

    params: dict[str, Any]
    reference: float
    tolerance_pct: float
    tolerance_abs: float = 0.0
    source: str = ""


@dataclass(frozen=True)
class BenchmarkSuite:
    """A collection of benchmark cases for one instrument/method."""

    id: str
    title: str
    features: tuple[str, ...]       # matched by features
    method: str
    source_citation: str
    cases: tuple[BenchmarkCase, ...]
    setup: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run traces (Store B — episodic / cold)
# ---------------------------------------------------------------------------

@dataclass
class RunTrace:
    """Raw trace of a single build attempt."""

    timestamp: str
    instrument: str
    method: str
    description: str
    pricing_plan: dict[str, Any]
    attempt: int
    code_hash: str
    validation_failures: list[str]
    diagnosis: dict[str, Any] | None
    resolved: bool
    lesson_id: str | None
    duration_seconds: float


# ---------------------------------------------------------------------------
# Retrieval specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalSpec:
    """What to retrieve for a given task."""

    method: str | None = None
    features: list[str] = field(default_factory=list)
    instrument: str | None = None
    exercise_style: str | None = None
    state_dependence: str | None = None
    schedule_dependence: bool | None = None
    model_family: str | None = None
    candidate_engine_families: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()
    error_signatures: list[str] = field(default_factory=list)
    max_lessons: int = 7
    include_benchmarks: bool = False
