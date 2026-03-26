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


@dataclass(frozen=True)
class SemanticIssue:
    """Single semantic validation issue."""

    code: str
    message: str


@dataclass(frozen=True)
class SemanticSignals:
    """Static semantic fingerprint extracted from generated code."""

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
            self._record_engine_family(node.module)
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

        if isinstance(node.func, ast.Attribute) and node.func.attr == "price":
            engine_name = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            if engine_name in self._mc_engine_vars:
                self.engine_families.add("monte_carlo")
                callback_name = _callback_name(node)
                if callback_name and callback_name in self.function_defs:
                    if _function_returns_input_shape(self.function_defs[callback_name]):
                        self.path_matrix_callbacks.add(callback_name)

        self.generic_visit(node)

    def _record_engine_family(self, module_name: str) -> None:
        """Map imported module paths onto the coarse pricing-engine taxonomy."""
        if module_name.startswith("trellis.models.black"):
            self.engine_families.add("analytical")
        elif module_name.startswith("trellis.models.transforms"):
            self.engine_families.add("fft_pricing")
        elif module_name.startswith("trellis.models.trees"):
            self.engine_families.add("lattice")
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

    primitive_plan = generation_plan.primitive_plan if generation_plan is not None else None
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

    if primitive_plan is not None and not primitive_plan.blockers:
        missing_primitives = [
            f"{primitive.module}.{primitive.symbol}"
            for primitive in primitive_plan.primitives
            if primitive.required
            and f"{primitive.module}.{primitive.symbol}" not in signals.resolved_calls
        ]
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

    if any(module == "trellis.models.monte_carlo.lsm" for module in signals.laguerre_import_modules):
        issues.append(SemanticIssue(
            code="exercise.invalid_basis_import",
            message=(
                "Import LaguerreBasis from "
                "`trellis.models.monte_carlo.schemes`, not from "
                "`trellis.models.monte_carlo.lsm`."
            ),
        ))

    if product_ir is not None and product_ir.exercise_style in {
        "american",
        "bermudan",
        "issuer_call",
        "holder_put",
    }:
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
    if product_ir is not None and product_ir.exercise_style in {"bermudan", "issuer_call", "holder_put"}:
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

    if product_ir is not None and signals.engine_families:
        allowed = set(product_ir.candidate_engine_families)
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
    )
