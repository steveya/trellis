"""Deterministic semantic validation for generated payoff modules.

This layer sits between import validation and file write. It performs a
Trellis-aware static analysis pass over generated code and rejects code that is
syntactically valid but semantically inconsistent with the intended pricing
contract.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from trellis.agent.codegen_guardrails import GenerationPlan
from trellis.agent.early_exercise_policy import (
    canonicalize_early_exercise_policy,
    render_early_exercise_policy_summary,
    render_implemented_early_exercise_policy_summary,
)
from trellis.agent.knowledge.schema import ProductIR

_RAW_STRING_SCHEDULE_FIELD_NAMES = frozenset({
    "call_dates",
    "put_dates",
    "exercise_dates",
    "observation_dates",
})
_RAW_STRING_SCHEDULE_ANNOTATIONS = frozenset({
    "str",
    "str|None",
    "Optional[str]",
    "typing.Optional[str]",
})
_ROUTE_HELPER_SUBSUMED_PRIMITIVE_ROLES = frozenset({
    "array_backend",
    "event_probability",
    "loss_distribution",
    "market_binding",
    "path_simulation",
    "pricing_kernel",
    "time_measure",
})


@dataclass(frozen=True)
class SemanticIssue:
    """Single semantic validation issue."""

    code: str
    message: str


@dataclass(frozen=True)
class SemanticSignals:
    """Key structural facts extracted from generated code by static analysis.

    Captures which pricing engines, Monte Carlo methods, exercise-control
    primitives, and other patterns appear in the code so that downstream
    validators can check consistency without running the code.
    """

    engine_families: tuple[str, ...]
    resolved_calls: tuple[str, ...]
    monte_carlo_methods: tuple[str, ...]
    exercise_control_primitives: tuple[str, ...]
    laguerre_import_modules: tuple[str, ...]
    transform_pricers: tuple[str, ...]
    scalar_math_functions: tuple[str, ...]
    path_matrix_callbacks: tuple[str, ...]
    lattice_exercise_types: tuple[str, ...]
    lattice_has_exercise_steps: bool
    lattice_exercise_functions: tuple[str, ...]
    lattice_exercise_styles: tuple[str, ...]
    lattice_invalid_policy_kwargs: tuple[str, ...]

    @property
    def uses_longstaff_schwartz(self) -> bool:
        """Backward-compatible convenience for the currently implemented primitive."""
        return "longstaff_schwartz" in self.exercise_control_primitives


@dataclass(frozen=True)
class SemanticValidationReport:
    """Outcome of semantic validation."""

    issues: tuple[SemanticIssue, ...]
    signals: SemanticSignals

    @property
    def ok(self) -> bool:
        """Return ``True`` when semantic validation produced no issues."""
        return not self.issues

    @property
    def errors(self) -> tuple[str, ...]:
        """Return human-readable ``code: message`` strings for all issues."""
        return tuple(f"{issue.code}: {issue.message}" for issue in self.issues)


class _SemanticVisitor(ast.NodeVisitor):
    """Extract semantic signals from generated code."""

    def __init__(self) -> None:
        """Initialize the semantic fingerprint accumulators used during AST traversal."""
        self.aliases: dict[str, str] = {}
        self.function_defs: dict[str, ast.FunctionDef] = {}
        self.engine_families: set[str] = set()
        self.resolved_calls: set[str] = set()
        self.monte_carlo_methods: list[str] = []
        self.exercise_control_primitives: set[str] = set()
        self.laguerre_import_modules: list[str] = []
        self.transform_pricers: list[str] = []
        self.scalar_math_functions: set[str] = set()
        self.path_matrix_callbacks: set[str] = set()
        self.lattice_exercise_types: list[str] = []
        self.lattice_has_exercise_steps = False
        self.lattice_exercise_functions: list[str] = []
        self.lattice_exercise_styles: list[str] = []
        self.lattice_invalid_policy_kwargs: list[str] = []
        self._lattice_policy_bindings: dict[str, _LatticePolicyBinding] = {}
        self._mc_engine_vars: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Record import aliases and infer broad engine families from module imports."""
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[-1]
            self.aliases[local] = alias.name
            self._record_engine_family(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record imported symbols, aliases, and special basis-function imports."""
        if node.level != 0 or not node.module:
            return
        for alias in node.names:
            local = alias.asname or alias.name
            self.aliases[local] = f"{node.module}.{alias.name}"
            self._record_engine_family(node.module, alias.name)
            if alias.name == "LaguerreBasis":
                self.laguerre_import_modules.append(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Index local functions and flag characteristic functions using scalar math."""
        self.function_defs[node.name] = node
        if node.name in {"char_fn", "psi"} and _function_uses_scalar_math(node, self.aliases):
            self.scalar_math_functions.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track variables bound to Monte Carlo engines for later call-site analysis."""
        call_name = _resolve_call_name(node.value, self.aliases)
        if call_name.endswith("MonteCarloEngine"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._mc_engine_vars.add(target.id)
        lattice_policy = _extract_lattice_policy_binding(node.value, self.aliases)
        if lattice_policy is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._lattice_policy_bindings[target.id] = lattice_policy
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Record semantic signals implied by function calls and keyword usage."""
        call_name = _resolve_call_name(node.func, self.aliases)
        if call_name:
            self.resolved_calls.add(call_name)

        if call_name.endswith("MonteCarloEngine"):
            self.engine_families.add("monte_carlo")
            method = _literal_keyword(node, "method")
            if isinstance(method, str):
                self.monte_carlo_methods.append(method)

        exercise_control = canonicalize_early_exercise_policy(call_name)
        if exercise_control is not None:
            self.exercise_control_primitives.add(exercise_control)
            self.engine_families.add("exercise")

        if call_name.endswith("fft_price"):
            self.transform_pricers.append("fft_price")
            self.engine_families.add("fft_pricing")
        elif call_name.endswith("cos_price"):
            self.transform_pricers.append("cos_price")
            self.engine_families.add("fft_pricing")

        if call_name.endswith("black76_call") or call_name.endswith("black76_put"):
            self.engine_families.add("analytical")

        if call_name.endswith("lattice_backward_induction"):
            self.engine_families.add("lattice")
            exercise_type = _literal_keyword(node, "exercise_type")
            if isinstance(exercise_type, str):
                self.lattice_exercise_types.append(exercise_type)
            if any(keyword.arg == "exercise_steps" for keyword in node.keywords):
                self.lattice_has_exercise_steps = True
            exercise_fn = _name_keyword(node, "exercise_fn", self.aliases)
            if exercise_fn:
                self.lattice_exercise_functions.append(exercise_fn)
            exercise_policy = _policy_keyword(node, "exercise_policy", self.aliases, self._lattice_policy_bindings)
            if exercise_policy is not None:
                self.lattice_exercise_styles.append(exercise_policy.exercise_style)
                self.lattice_exercise_types.append(exercise_policy.exercise_type)
                self.lattice_has_exercise_steps = (
                    self.lattice_has_exercise_steps
                    or bool(exercise_policy.has_exercise_steps)
                )
                self.lattice_exercise_functions.append(exercise_policy.exercise_objective)

        if call_name.endswith("resolve_lattice_exercise_policy"):
            invalid_kwargs = sorted(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg in {"exercise_fn", "exercise_type"}
            )
            self.lattice_invalid_policy_kwargs.extend(invalid_kwargs)

        if isinstance(node.func, ast.Attribute) and node.func.attr == "price":
            engine_name = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            if engine_name in self._mc_engine_vars:
                self.engine_families.add("monte_carlo")
                callback_name = _callback_name(node)
                if callback_name and callback_name in self.function_defs:
                    if _function_returns_input_shape(self.function_defs[callback_name]):
                        self.path_matrix_callbacks.add(callback_name)

        self.generic_visit(node)

    def _record_engine_family(self, module_name: str, symbol_name: str | None = None) -> None:
        """Map imported module paths onto the exact route-family taxonomy when possible."""
        if module_name.startswith("trellis.models.black"):
            self.engine_families.add("analytical")
        elif module_name.startswith("trellis.models.rate_style_swaption"):
            self.engine_families.add("analytical")
        elif module_name.startswith("trellis.models.transforms"):
            self.engine_families.add("fft_pricing")
        elif module_name.startswith("trellis.models.trees.binomial"):
            self.engine_families.add("equity_tree")
        elif module_name.startswith("trellis.models.trees.backward_induction"):
            self.engine_families.add("equity_tree")
        elif module_name.startswith("trellis.models.trees.trinomial"):
            self.engine_families.add("equity_tree")
        elif module_name.startswith("trellis.models.equity_option_tree"):
            self.engine_families.add("equity_tree")
        elif module_name.startswith("trellis.models.bermudan_swaption_tree"):
            self.engine_families.add("rate_lattice")
        elif module_name.startswith("trellis.models.equity_option_pde"):
            self.engine_families.add("pde_solver")
        elif module_name.startswith("trellis.models.trees.lattice"):
            self.engine_families.add("rate_lattice")
        elif module_name == "trellis.models.trees":
            if symbol_name in {"BinomialTree", "TrinomialTree", "backward_induction"}:
                self.engine_families.add("equity_tree")
            elif symbol_name in {"build_rate_lattice", "lattice_backward_induction", "RecombiningLattice"}:
                self.engine_families.add("rate_lattice")
            else:
                self.engine_families.update({"equity_tree", "rate_lattice"})
        elif module_name.startswith("trellis.models.pde"):
            self.engine_families.add("pde_solver")
        elif module_name.startswith("trellis.models.copulas"):
            self.engine_families.add("copula")
        elif module_name.startswith("trellis.models.cashflow_engine"):
            self.engine_families.add("waterfall")
        elif module_name.startswith("trellis.models.qmc"):
            self.engine_families.add("qmc")
        elif module_name.startswith("trellis.models.monte_carlo"):
            self.engine_families.add("monte_carlo")


def extract_semantic_signals(source: str) -> SemanticSignals:
    """Extract deterministic semantic signals from generated source code."""
    tree = ast.parse(source)
    visitor = _SemanticVisitor()
    visitor.visit(tree)

    return SemanticSignals(
        engine_families=tuple(sorted(visitor.engine_families)),
        resolved_calls=tuple(sorted(visitor.resolved_calls)),
        monte_carlo_methods=tuple(visitor.monte_carlo_methods),
        exercise_control_primitives=tuple(sorted(visitor.exercise_control_primitives)),
        laguerre_import_modules=tuple(visitor.laguerre_import_modules),
        transform_pricers=tuple(visitor.transform_pricers),
        scalar_math_functions=tuple(sorted(visitor.scalar_math_functions)),
        path_matrix_callbacks=tuple(sorted(visitor.path_matrix_callbacks)),
        lattice_exercise_types=tuple(visitor.lattice_exercise_types),
        lattice_has_exercise_steps=visitor.lattice_has_exercise_steps,
        lattice_exercise_functions=tuple(visitor.lattice_exercise_functions),
        lattice_exercise_styles=tuple(visitor.lattice_exercise_styles),
        lattice_invalid_policy_kwargs=tuple(visitor.lattice_invalid_policy_kwargs),
    )


def validate_semantics(
    source: str,
    *,
    product_ir: ProductIR | None = None,
    generation_plan: GenerationPlan | None = None,
) -> SemanticValidationReport:
    """Validate generated code against Trellis semantic contracts."""
    try:
        signals = extract_semantic_signals(source)
    except SyntaxError as exc:
        issue = SemanticIssue(
            code="semantic.parse_error",
            message=f"cannot analyze generated code because it is not valid Python: {exc}",
        )
        return SemanticValidationReport(issues=(issue,), signals=_empty_signals())

    issues: list[SemanticIssue] = []

    for field_name, annotation in _raw_string_schedule_fields(source):
        issues.append(
            SemanticIssue(
                code="schedule.raw_string_field",
                message=(
                    f"Schedule-bearing spec field `{field_name}` is annotated as `{annotation}`. "
                    "Use `tuple[date, ...]` or `tuple[date, ...] | None` and normalize those "
                    "explicit dates onto `ContractTimeline` helpers instead of comma-separated strings."
                ),
            )
        )

    primitive_plan = generation_plan.primitive_plan if generation_plan is not None else None
    required_route_helpers: tuple[str, ...] = ()
    helper_only_required_route = False
    helper_route_calls_present = False
    if primitive_plan is not None:
        required_route_helpers = tuple(
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitive_plan.primitives
            if primitive.required and primitive.role == "route_helper"
        )
        required_non_helper_primitives = tuple(
            primitive
            for primitive in primitive_plan.primitives
            if primitive.required and primitive.role != "route_helper"
        )
        helper_only_required_route = bool(required_route_helpers) and not required_non_helper_primitives
        helper_route_calls_present = any(
            helper_name in signals.resolved_calls
            for helper_name in required_route_helpers
        )
    if primitive_plan is not None and primitive_plan.blockers:
        blocker_text = ", ".join(primitive_plan.blockers)
        if generation_plan is not None and generation_plan.blocker_report is not None:
            blocker_text = generation_plan.blocker_report.summary
        issues.append(SemanticIssue(
            code="assembly.route_has_blockers",
            message=(
                "The selected primitive route is blocked by unresolved prerequisites: "
                f"{blocker_text}. Do not generate speculative "
                "code for this product until the blockers are resolved."
            ),
        ))

    thin_adapter_calls = any(call.startswith("trellis.instruments.") for call in signals.resolved_calls)
    helper_backed_thin_adapter = thin_adapter_calls or (
        helper_only_required_route and helper_route_calls_present
    )
    if primitive_plan is not None and not primitive_plan.blockers and not thin_adapter_calls:
        required_primitives = tuple(
            primitive
            for primitive in primitive_plan.primitives
            if primitive.required
            and not (
                helper_route_calls_present
                and primitive.role in _ROUTE_HELPER_SUBSUMED_PRIMITIVE_ROLES
            )
        )
        called_primitives = {
            f"{primitive.module}.{primitive.symbol}"
            for primitive in required_primitives
            if f"{primitive.module}.{primitive.symbol}" in signals.resolved_calls
        }
        role_to_primitives: dict[str, list[object]] = {}
        for primitive in required_primitives:
            role_to_primitives.setdefault(primitive.role, []).append(primitive)

        missing_primitives: list[str] = []
        for primitives_for_role in role_to_primitives.values():
            role_refs = {
                f"{primitive.module}.{primitive.symbol}"
                for primitive in primitives_for_role
            }
            if called_primitives.intersection(role_refs):
                continue
            missing_primitives.extend(sorted(role_refs))
        if missing_primitives:
            issues.append(SemanticIssue(
                code="assembly.required_primitive_missing",
                message=(
                    "Generated code did not use the required primitives from the "
                    "selected assembly route: "
                    + ", ".join(missing_primitives)
                    + ". Rebuild the payoff as a thin adapter around those primitives."
                ),
            ))

        excluded_primitives = [
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitive_plan.primitives
            if primitive.excluded
            and f"{primitive.module}.{primitive.symbol}" in signals.resolved_calls
        ]
        if excluded_primitives:
            issues.append(SemanticIssue(
                code="assembly.excluded_primitive_used",
                message=(
                    "Generated code called a primitive that is explicitly excluded "
                    "from the selected assembly route: "
                    + ", ".join(excluded_primitives)
                    + ". This primitive is incompatible with this route's engine family."
                ),
            ))

    if any(method not in {"euler", "milstein", "exact"} for method in signals.monte_carlo_methods):
        issues.append(SemanticIssue(
            code="mc.invalid_method_mode",
            message=(
                "MonteCarloEngine only supports method strings 'euler', "
                "'milstein', or 'exact'. Use an approved early-exercise "
                "control primitive instead of inventing method='lsm'. "
                f"Approved policy classes: {render_early_exercise_policy_summary()}."
            ),
        ))

    if signals.lattice_invalid_policy_kwargs:
        invalid_kwargs = ", ".join(f"`{name}`" for name in sorted(set(signals.lattice_invalid_policy_kwargs)))
        issues.append(SemanticIssue(
            code="lattice.invalid_policy_kwarg",
            message=(
                "resolve_lattice_exercise_policy(...) only accepts product exercise semantics "
                f"and schedule steps. Remove unsupported keyword(s) {invalid_kwargs} and "
                "let the checked-in policy object carry the lattice objective/type."
            ),
        ))

    if any(module == "trellis.models.monte_carlo.lsm" for module in signals.laguerre_import_modules):
        issues.append(SemanticIssue(
            code="exercise.invalid_basis_import",
            message=(
                "Import LaguerreBasis from "
                "`trellis.models.monte_carlo.schemes`, not from "
                "`trellis.models.monte_carlo.lsm`."
            ),
        ))

    if (
        not helper_backed_thin_adapter
        and product_ir is not None
        and product_ir.exercise_style in {
        "american",
        "bermudan",
        "issuer_call",
        "holder_put",
    }
    ):
        if (
            ("monte_carlo" in signals.engine_families or (generation_plan and generation_plan.method == "monte_carlo"))
            and not signals.exercise_control_primitives
            and "lattice" not in signals.engine_families
        ):
            issues.append(SemanticIssue(
                code="exercise.missing_control_primitive",
                message=(
                    "Early-exercise products priced with Monte Carlo must use a "
                    "real control primitive instead of plain "
                    "MonteCarloEngine.price(...). Approved policy classes are "
                    f"{render_early_exercise_policy_summary()}. Currently "
                    "implemented in Trellis: "
                    f"{render_implemented_early_exercise_policy_summary()}."
                ),
            ))

    uses_lattice_backward_induction = any(
        call.endswith("lattice_backward_induction")
        for call in signals.resolved_calls
    )
    if (
        not helper_backed_thin_adapter
        and product_ir is not None
        and product_ir.exercise_style in {"bermudan", "issuer_call", "holder_put"}
    ):
        lattice_expected = uses_lattice_backward_induction
        if generation_plan is not None and generation_plan.primitive_plan is not None:
            lattice_expected = lattice_expected or generation_plan.primitive_plan.route == "exercise_lattice"
        if lattice_expected:
            if "bermudan" not in signals.lattice_exercise_types:
                issues.append(SemanticIssue(
                    code="lattice.exercise_type_mismatch",
                    message=(
                        "Schedule-dependent lattice exercise products must use "
                        "`exercise_type=\"bermudan\"` in lattice_backward_induction(...)."
                    ),
                ))
            if product_ir.schedule_dependence and not signals.lattice_has_exercise_steps:
                issues.append(SemanticIssue(
                    code="lattice.exercise_schedule_missing",
                    message=(
                        "Schedule-dependent lattice exercise products must pass "
                        "`exercise_steps=...` to lattice_backward_induction(...)."
                    ),
                ))
            expected_objective = "min" if product_ir.exercise_style == "issuer_call" else "max"
            if expected_objective not in signals.lattice_exercise_functions:
                issues.append(SemanticIssue(
                    code="lattice.exercise_objective_mismatch",
                    message=(
                        "Lattice exercise objective does not match the product semantics. "
                        f"Expected `exercise_fn={expected_objective}` for "
                        f"`{product_ir.exercise_style}` products."
                    ),
                ))

    if signals.path_matrix_callbacks:
        callbacks = ", ".join(f"`{name}`" for name in signals.path_matrix_callbacks)
        issues.append(SemanticIssue(
            code="mc.invalid_payoff_shape",
            message=(
                f"MonteCarloEngine.price(...) expects payoff callbacks that map "
                f"full paths to a `(n_paths,)` payoff vector; {callbacks} "
                "appears to use the full path matrix in arithmetic directly."
            ),
        ))

    if signals.transform_pricers and signals.scalar_math_functions:
        funcs = ", ".join(f"`{name}`" for name in signals.scalar_math_functions)
        issues.append(SemanticIssue(
            code="transform.scalar_char_fn",
            message=(
                "Transform pricing helpers require vector-safe characteristic "
                f"functions. {funcs} uses scalar math/cmath calls; use "
                "array-safe numpy operations so char_fn can accept vector u."
            ),
        ))

    if product_ir is not None and signals.engine_families and not helper_backed_thin_adapter:
        allowed = set(product_ir.candidate_engine_families)
        allowed.update(getattr(product_ir, "route_families", ()))
        if generation_plan is not None:
            allowed.add(_plan_method_to_ir_family(generation_plan.method))
        if allowed and not set(signals.engine_families).intersection(allowed):
            issues.append(SemanticIssue(
                code="engine.family_incompatible_with_ir",
                message=(
                    f"Generated code uses engine families {sorted(signals.engine_families)!r}, "
                    f"but the product IR only allows {sorted(allowed)!r}."
                ),
            ))

    return SemanticValidationReport(issues=tuple(issues), signals=signals)


def _resolve_call_name(node: ast.AST, aliases: dict[str, str]) -> str:
    """Resolve an AST call target into a dotted name using known aliases."""
    if isinstance(node, ast.Call):
        return _resolve_call_name(node.func, aliases)
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_call_name(node.value, aliases)
        return f"{base}.{node.attr}"
    return ""


def _literal_keyword(node: ast.Call, name: str) -> str | None:
    """Return a literal string keyword argument when it is statically available."""
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value if isinstance(keyword.value.value, str) else None
    return None


def _name_keyword(node: ast.Call, name: str, aliases: dict[str, str]) -> str | None:
    """Resolve a keyword argument that names a symbol or dotted attribute."""
    for keyword in node.keywords:
        if keyword.arg != name:
            continue
        return _resolve_name(keyword.value, aliases)
    return None


def _callback_name(node: ast.Call) -> str | None:
    """Extract the payoff callback name from supported Monte Carlo call signatures."""
    if len(node.args) >= 3 and isinstance(node.args[2], ast.Name):
        return node.args[2].id
    for keyword in node.keywords:
        if keyword.arg == "payoff_fn" and isinstance(keyword.value, ast.Name):
            return keyword.value.id
    return None


@dataclass(frozen=True)
class _LatticePolicyBinding:
    """Static fingerprint for a resolved lattice exercise policy binding."""

    exercise_style: str
    exercise_type: str
    exercise_objective: str
    has_exercise_steps: bool


def _extract_lattice_policy_binding(
    node: ast.AST,
    aliases: dict[str, str],
) -> _LatticePolicyBinding | None:
    """Extract a static lattice exercise policy binding from a helper call."""
    if not isinstance(node, ast.Call):
        return None
    call_name = _resolve_call_name(node.func, aliases)
    if not call_name.endswith("resolve_lattice_exercise_policy"):
        return None
    exercise_style = _literal_keyword(node, "exercise_style")
    if exercise_style is None and node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        exercise_style = value if isinstance(value, str) else None
    if exercise_style is None:
        return None
    exercise_style = exercise_style.strip().lower()
    exercise_type = "bermudan" if exercise_style in {"bermudan", "issuer_call", "holder_put"} else exercise_style
    exercise_objective = "min" if exercise_style == "issuer_call" else "max"
    has_exercise_steps = any(keyword.arg == "exercise_steps" for keyword in node.keywords)
    if len(node.args) >= 2:
        has_exercise_steps = True
    return _LatticePolicyBinding(
        exercise_style=exercise_style,
        exercise_type=exercise_type,
        exercise_objective=exercise_objective,
        has_exercise_steps=has_exercise_steps,
    )


def _policy_keyword(
    node: ast.Call,
    name: str,
    aliases: dict[str, str],
    bindings: dict[str, _LatticePolicyBinding],
) -> _LatticePolicyBinding | None:
    """Resolve a lattice policy keyword to a tracked binding when possible."""
    for keyword in node.keywords:
        if keyword.arg != name:
            continue
        if isinstance(keyword.value, ast.Name):
            return bindings.get(keyword.value.id)
        return _extract_lattice_policy_binding(keyword.value, aliases)
    return None


def _resolve_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a bare name or attribute expression into a dotted symbol reference."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_name(node.value, aliases)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _raw_string_schedule_fields(source: str) -> tuple[tuple[str, str], ...]:
    """Return schedule-bearing spec fields that still use raw string annotations."""
    tree = ast.parse(source)
    flagged: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        field_name = node.target.id
        if field_name not in _RAW_STRING_SCHEDULE_FIELD_NAMES:
            continue
        annotation = ast.unparse(node.annotation).replace(" ", "")
        if annotation in _RAW_STRING_SCHEDULE_ANNOTATIONS:
            flagged.append((field_name, ast.unparse(node.annotation)))
    return tuple(flagged)


def _function_uses_scalar_math(node: ast.FunctionDef, aliases: dict[str, str]) -> bool:
    """Return whether a function calls scalar-only ``math`` or ``cmath`` helpers."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        call_name = _resolve_call_name(child.func, aliases)
        if call_name.startswith("math.") or call_name.startswith("cmath."):
            return True
    return False


def _function_returns_input_shape(node: ast.FunctionDef) -> bool:
    """Heuristically detect callbacks that return path-matrix-shaped expressions."""
    if not node.args.args:
        return False
    parameter = node.args.args[0].arg
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            if _expression_uses_path_matrix_arithmetic(child.value, parameter):
                return True
    return False


def _expression_uses_path_matrix_arithmetic(expr: ast.AST, parameter: str) -> bool:
    """Return whether an expression performs arithmetic on the raw path matrix input."""
    if isinstance(expr, ast.Name) and expr.id == parameter:
        return True

    parents = _parent_map(expr)
    for child in ast.walk(expr):
        if not isinstance(child, ast.Name) or child.id != parameter:
            continue
        parent = parents.get(child)
        if isinstance(parent, ast.Subscript) and parent.value is child:
            continue
        current = parent
        while current is not None and current is not expr:
            if isinstance(current, (ast.BinOp, ast.Compare, ast.BoolOp)):
                return True
            current = parents.get(current)
    return False


def _parent_map(node: ast.AST) -> dict[ast.AST, ast.AST]:
    """Build parent pointers for an AST subtree to support upward inspection."""
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _plan_method_to_ir_family(method: str) -> str:
    """Normalize method labels so plan methods align with IR engine-family names."""
    if method == "rate_tree":
        return "lattice"
    return method


def _empty_signals() -> SemanticSignals:
    """Return the neutral semantic-signal bundle used after parse failure."""
    return SemanticSignals(
        engine_families=(),
        resolved_calls=(),
        monte_carlo_methods=(),
        exercise_control_primitives=(),
        laguerre_import_modules=(),
        transform_pricers=(),
        scalar_math_functions=(),
        path_matrix_callbacks=(),
        lattice_exercise_types=(),
        lattice_has_exercise_steps=False,
        lattice_exercise_functions=(),
        lattice_exercise_styles=(),
        lattice_invalid_policy_kwargs=(),
    )
