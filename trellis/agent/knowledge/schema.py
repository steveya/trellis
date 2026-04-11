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


class AdapterLifecycleStatus(str, Enum):
    """Lifecycle state for checked-in adapters and fresh-build replacements."""

    FRESH = "fresh"             # Current validated adapter or fresh replacement
    STALE = "stale"             # Older than the validated replacement
    DEPRECATED = "deprecated"   # Kept only for compatibility
    ARCHIVED = "archived"       # Removed from normal retrieval


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
    route_families: tuple[str, ...] = ()   # exact route-family labels used by planner/validator
    required_market_data: frozenset[str] = frozenset()
    reusable_primitives: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()
    supported: bool = True
    event_machine: object | None = None  # EventMachine when typed, None for legacy


@dataclass(frozen=True)
class SimilarProductMatch:
    """One deterministic near-match to a sparse or novel product request."""

    instrument: str
    method: str
    score: float
    shared_features: tuple[str, ...] = ()
    query_only_features: tuple[str, ...] = ()
    candidate_only_features: tuple[str, ...] = ()
    promoted_routes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Build gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildGateThresholds:
    """Configurable thresholds for the build gate checkpoint.

    Controls when the pipeline blocks, narrows routes, or proceeds.
    """

    block_below: float = 0.3           # confidence < this → block
    narrow_below: float = 0.55         # confidence < this → narrow_route
    max_unresolved_conflicts: int = 0  # any conflicts → clarify
    require_promoted_route: bool = False  # if True, no promoted route → block


@dataclass(frozen=True)
class BuildGateDecision:
    """Result of a build gate evaluation.

    Emitted between validation (Layer 2) and compilation (Layer 3)
    to prevent wasted LLM round-trips on doomed builds.
    """

    decision: str          # "proceed" | "narrow_route" | "clarify" | "block"
    reason: str            # human-readable explanation
    gap_confidence: float  # from GapReport
    unresolved_conflicts: tuple[str, ...] = ()   # conflict summaries
    missing_required_inputs: tuple[str, ...] = ()  # missing market inputs
    route_admissibility_failures: tuple[str, ...] = ()
    suggested_fallback_route: str | None = None
    gate_source: str = ""  # "pre_flight" or "pre_generation"


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
    supersedes: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class LessonRegressionTemplate:
    """Deterministic regression-template metadata for one lesson family."""

    family: str
    target_test_file: str
    description: str = ""
    assertion_focus: tuple[str, ...] = ()
    fixture_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LessonRegressionPayload:
    """Materialized regression payload derived from a validated or promoted lesson."""

    lesson_id: str
    lesson_title: str
    lesson_category: str
    lesson_status: LessonStatus
    template_family: str
    target_test_file: str
    applies_when: AppliesWhen
    rationale: str
    assertion_focus: tuple[str, ...] = ()
    fixture_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source_trace: str | None = None
    rendered_fragment: str = ""


@dataclass(frozen=True)
class AdapterLifecycleRecord:
    """Lifecycle metadata for an adapter family or fresh-build replacement."""

    adapter_id: str
    status: AdapterLifecycleStatus
    module_path: str
    validated_against_repo_revision: str = ""
    supersedes: tuple[str, ...] = ()
    replacement: str = ""
    reason: str = ""
    code_hash: str = ""


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


@dataclass(frozen=True)
class ModelGrammarEntry:
    """Canonical calibration-layer model-grammar entry used for planner lookup."""

    id: str
    title: str
    methods: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    model_families: tuple[str, ...] = ()
    engine_families: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    model_name: str = ""
    state_semantics: tuple[str, ...] = ()
    quote_families: tuple[str, ...] = ()
    calibration_workflows: tuple[str, ...] = ()
    runtime_materialization_kind: str = ""
    runtime_materialization_targets: tuple[str, ...] = ()
    rates_curve_roles: tuple[str, ...] = ()
    required_market_data: tuple[str, ...] = ()
    authority_surfaces: tuple[str, ...] = ()
    deferred_scope: tuple[str, ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# Structured memory / repo state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRule:
    """A stable policy statement that should be treated as always-on guidance."""

    id: str
    rule: str
    scope: tuple[str, ...] = ()
    rationale: str = ""
    priority: int = 0


@dataclass(frozen=True)
class WorkflowTemplate:
    """A reusable workflow skeleton for build, review, or repair flows."""

    id: str
    name: str
    steps: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class FailureCase:
    """A normalized failure example extracted from a trace or task run."""

    id: str
    failure_type: str
    signature: str
    description: str = ""
    task_id: str = ""
    trace_path: str | None = None
    repo_revision: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairRecipe:
    """A reusable repair pattern distilled from successful recovery."""

    id: str
    failure_type: str
    recipe: str
    prerequisites: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    scope: tuple[str, ...] = ()
    reusable: bool = True


@dataclass(frozen=True)
class EvalSpec:
    """An explicit evaluation definition for a benchmark or task tranche."""

    id: str
    title: str
    description: str = ""
    benchmark_ids: tuple[str, ...] = ()
    grader_ids: tuple[str, ...] = ()
    hard_gates: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraderSpec:
    """A single deterministic grader and the signals it cares about."""

    id: str
    category: str
    description: str = ""
    hard: bool = True
    applies_to: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryExample:
    """A compact execution trajectory that can be reused as memory."""

    id: str
    task_id: str
    repo_revision: str
    outcome: str
    steps: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()
    trace_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepoFact:
    """A live repository fact keyed by revision."""

    kind: str
    key: str
    value: str
    repo_revision: str
    source_path: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class SymbolMap:
    """Revision-scoped map of modules to exported public symbols."""

    repo_revision: str
    module_to_symbols: dict[str, tuple[str, ...]]
    symbol_to_modules: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class PackageMap:
    """Revision-scoped map of package roots to the modules they contain."""

    repo_revision: str
    package_to_modules: dict[str, tuple[str, ...]]
    module_to_package: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TestMap:
    """Revision-scoped map of test directories and likely test targets."""

    repo_revision: str
    directory_to_tests: dict[str, tuple[str, ...]]
    symbol_to_tests: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContract:
    """A structured contract for a platform or agent tool."""

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    guarantees: tuple[str, ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# Generated skill layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillRecord:
    """A generated reusable-guidance record projected from source artifacts."""

    skill_id: str
    kind: str
    title: str
    summary: str = ""
    source_artifact: str = ""
    source_path: str = ""
    instrument_types: tuple[str, ...] = ()
    method_families: tuple[str, ...] = ()
    route_families: tuple[str, ...] = ()
    failure_buckets: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    origin: str = ""
    parents: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    status: str = ""
    confidence: float = 1.0
    updated_at: str = ""
    precedence_rank: int = 0
    instruction_type: str = ""
    source_kind: str = ""
    lineage_status: str = ""
    lineage_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillIndexManifest:
    """Provenance metadata for one generated skill index snapshot."""

    repo_revision: str
    source_paths: tuple[str, ...] = ()
    source_fingerprints: tuple[str, ...] = ()
    record_count: int = 0
    kind_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedSkillIndex:
    """Deterministic generated view over reusable guidance artifacts."""

    manifest: SkillIndexManifest
    records: tuple[SkillRecord, ...] = ()


# ---------------------------------------------------------------------------
# Instruction lifecycle / route guidance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstructionConflict:
    """A conflict between two or more instruction records."""

    reason: str
    conflicting_ids: tuple[str, ...] = ()
    winner_id: str | None = None


@dataclass(frozen=True)
class InstructionRecord:
    """A versioned route guidance record with explicit scope and precedence."""

    id: str
    title: str
    instruction_type: str
    status: str = "active"
    source_kind: str = "canonical"
    source_id: str = ""
    source_revision: str = ""
    scope_methods: tuple[str, ...] = ()
    scope_instruments: tuple[str, ...] = ()
    scope_routes: tuple[str, ...] = ()
    scope_modules: tuple[str, ...] = ()
    scope_features: tuple[str, ...] = ()
    precedence_rank: int = 0
    supersedes: tuple[str, ...] = ()
    conflict_policy: str = "prefer_newer"
    statement: str = ""
    rationale: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class ResolvedInstructionSet:
    """The precedence-resolved instruction set for one route."""

    route: str
    effective_instructions: tuple[InstructionRecord, ...] = ()
    dropped_instructions: tuple[InstructionRecord, ...] = ()
    conflicts: tuple[InstructionConflict, ...] = ()


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
    semantic_text_markers: tuple[str, ...] = ()
    reusable_primitives: tuple[str, ...] = ()
    unresolved_primitives: tuple[str, ...] = ()
    error_signatures: list[str] = field(default_factory=list)
    max_lessons: int = 7
    include_benchmarks: bool = False
