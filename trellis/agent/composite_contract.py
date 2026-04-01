"""Composition algebra — express structured products as sub-contract DAGs.

A ``CompositeSemanticContract`` holds a DAG of ``SubContractRef`` nodes
connected by ``ContractEdge`` edges.  The compiler flattens the DAG into
an ordered build sequence (topological sort) and emits proven primitive
references for proven sub-contracts, leaving only unproven stubs for the
agent to generate.

Initial scope (per S10): only ``sequential`` and ``calibrate_then_price``
edge types.  Conditional and parallel edges are deferred until a real
product needs them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractEdge:
    """A directed edge between two sub-contracts in the composition DAG."""

    from_contract: str
    to_contract: str
    edge_type: str = "sequential"  # "sequential" | "calibrate_then_price"
    data_flow: tuple[str, ...] = ()  # what flows: ("calibrated_lattice", "discount_curve")
    description: str = ""
    condition: str = ""  # reserved for future conditional edges


@dataclass(frozen=True)
class SubContractRef:
    """A reference to one sub-contract in the composition."""

    contract_id: str
    contract: object  # SemanticContract | CalibrationContract | CompositeSemanticContract
    proven: bool = False
    primitive_ref: str = ""  # module path to proven implementation


@dataclass(frozen=True)
class CompositeSemanticContract:
    """A structured product expressed as a DAG of sub-contracts."""

    composite_id: str
    description: str
    sub_contracts: tuple[SubContractRef, ...]
    edges: tuple[ContractEdge, ...]
    root_contract: str
    terminal_contracts: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_composite_contract(
    composite: CompositeSemanticContract,
) -> tuple[str, ...]:
    """Validate a ``CompositeSemanticContract`` for structural correctness.

    Returns a tuple of error strings (empty if valid).
    """
    errors: list[str] = []
    contract_ids = {sc.contract_id for sc in composite.sub_contracts}

    # 1. Root must exist
    if composite.root_contract not in contract_ids:
        errors.append(
            f"Root contract '{composite.root_contract}' not in sub-contracts: "
            f"{sorted(contract_ids)}"
        )

    # 2. Terminals must exist
    for tc in composite.terminal_contracts:
        if tc not in contract_ids:
            errors.append(
                f"Terminal contract '{tc}' not in sub-contracts: {sorted(contract_ids)}"
            )

    # 3. Must have at least one terminal
    if not composite.terminal_contracts:
        errors.append("No terminal contracts defined")

    # 4. Edge endpoints must reference defined sub-contracts
    for edge in composite.edges:
        if edge.from_contract not in contract_ids:
            errors.append(
                f"Edge references undefined from_contract '{edge.from_contract}'"
            )
        if edge.to_contract not in contract_ids:
            errors.append(
                f"Edge references undefined to_contract '{edge.to_contract}'"
            )

    # 5. DAG must be acyclic (topological sort)
    if contract_ids:
        cycle = _detect_cycle(contract_ids, composite.edges)
        if cycle:
            errors.append(f"Cycle detected in composition DAG: {' -> '.join(cycle)}")

    # 6. Terminals must be reachable from root
    if composite.root_contract in contract_ids and composite.terminal_contracts:
        adj: dict[str, list[str]] = {cid: [] for cid in contract_ids}
        for edge in composite.edges:
            if edge.from_contract in adj:
                adj[edge.from_contract].append(edge.to_contract)
        reachable = _reachable_from(composite.root_contract, adj)
        terminal_set = set(composite.terminal_contracts)
        unreachable_terminals = terminal_set - reachable
        if unreachable_terminals:
            errors.append(
                f"Terminal contract(s) not reachable from root: "
                f"{sorted(unreachable_terminals)}"
            )

    return tuple(errors)


def _detect_cycle(
    nodes: set[str],
    edges: tuple[ContractEdge, ...],
) -> list[str] | None:
    """Return a cycle path if the graph has a cycle, else None."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in edges:
        if edge.from_contract in adj:
            adj[edge.from_contract].append(edge.to_contract)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}
    parent: dict[str, str | None] = {n: None for n in nodes}

    def dfs(node: str) -> list[str] | None:
        color[node] = GRAY
        for neighbor in adj.get(node, ()):
            if color.get(neighbor) == GRAY:
                # Reconstruct cycle
                cycle = [neighbor, node]
                cur = node
                while parent.get(cur) and parent[cur] != neighbor:
                    cur = parent[cur]  # type: ignore[assignment]
                    cycle.append(cur)
                cycle.reverse()
                return cycle
            if color.get(neighbor) == WHITE:
                parent[neighbor] = node
                result = dfs(neighbor)
                if result:
                    return result
        color[node] = BLACK
        return None

    for node in nodes:
        if color[node] == WHITE:
            result = dfs(node)
            if result:
                return result
    return None


def _reachable_from(start: str, adj: dict[str, list[str]]) -> set[str]:
    """BFS reachability from *start*."""
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, ()):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


# ---------------------------------------------------------------------------
# Market data and method analysis
# ---------------------------------------------------------------------------

def union_market_data_requirements(
    composite: CompositeSemanticContract,
) -> frozenset[str]:
    """Return the union of all sub-contracts' required market data."""
    all_data: set[str] = set()
    for ref in composite.sub_contracts:
        contract = ref.contract
        # SemanticContract path
        market_data = getattr(contract, "market_data", None)
        if market_data is not None:
            for inp in getattr(market_data, "required_inputs", ()):
                input_id = getattr(inp, "input_id", None) or getattr(inp, "capability", "")
                if input_id:
                    all_data.add(input_id)
        # CalibrationContract path
        target = getattr(contract, "target", None)
        if target is not None and hasattr(contract, "fitting_instruments"):
            for fi in getattr(contract, "fitting_instruments", ()):
                all_data.add(getattr(fi, "instrument_type", ""))
    return frozenset(all_data)


def topological_sort(
    composite: CompositeSemanticContract,
) -> tuple[str, ...]:
    """Return sub-contract IDs in topological order (dependencies first).

    Raises ``ValueError`` if the graph has a cycle.
    """
    contract_ids = {sc.contract_id for sc in composite.sub_contracts}
    in_degree: dict[str, int] = {cid: 0 for cid in contract_ids}
    adj: dict[str, list[str]] = {cid: [] for cid in contract_ids}

    for edge in composite.edges:
        if edge.from_contract in adj and edge.to_contract in in_degree:
            adj[edge.from_contract].append(edge.to_contract)
            in_degree[edge.to_contract] += 1

    queue: deque[str] = deque(
        cid for cid, deg in in_degree.items() if deg == 0
    )
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj.get(node, ()):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(contract_ids):
        raise ValueError("Cycle in composite contract DAG — cannot topologically sort")

    return tuple(order)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_composite_contract(
    composite: CompositeSemanticContract,
) -> "CompositeBlueprint":
    """Compile a composite contract into an ordered build sequence.

    For each sub-contract in topological order:
    - If proven: emit a primitive reference
    - If not: emit a generation stub
    """
    errors = validate_composite_contract(composite)
    if errors:
        raise ValueError(
            f"Cannot compile invalid composite contract: {'; '.join(errors)}"
        )

    order = topological_sort(composite)
    ref_by_id = {sc.contract_id: sc for sc in composite.sub_contracts}

    steps: list[CompositeStep] = []
    for contract_id in order:
        ref = ref_by_id[contract_id]
        # Find incoming edges (what this step consumes)
        incoming = tuple(
            edge for edge in composite.edges
            if edge.to_contract == contract_id
        )
        consumes = tuple(
            item
            for edge in incoming
            for item in edge.data_flow
        )
        # Find outgoing edges (what this step produces)
        outgoing = tuple(
            edge for edge in composite.edges
            if edge.from_contract == contract_id
        )
        produces = tuple(
            item
            for edge in outgoing
            for item in edge.data_flow
        )

        steps.append(CompositeStep(
            contract_id=contract_id,
            proven=ref.proven,
            primitive_ref=ref.primitive_ref,
            consumes=consumes,
            produces=produces,
        ))

    return CompositeBlueprint(
        composite_id=composite.composite_id,
        steps=tuple(steps),
        market_data_requirements=union_market_data_requirements(composite),
        description=composite.description,
    )


@dataclass(frozen=True)
class CompositeStep:
    """One step in the flattened composition sequence."""

    contract_id: str
    proven: bool = False
    primitive_ref: str = ""
    consumes: tuple[str, ...] = ()  # data consumed from upstream
    produces: tuple[str, ...] = ()  # data produced for downstream


@dataclass(frozen=True)
class CompositeBlueprint:
    """Compiled output of a composite contract."""

    composite_id: str
    steps: tuple[CompositeStep, ...]
    market_data_requirements: frozenset[str] = frozenset()
    description: str = ""


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def callable_bond_composite(
    *,
    description: str = "Callable bond: HW calibration → bond cashflows → backward induction",
    coupon: float = 0.05,
    call_dates: tuple[str, ...] = ("2025-06-30", "2026-06-30"),
) -> CompositeSemanticContract:
    """Build the canonical callable bond as a linear composition.

    calibration → cashflows → backward_induction
    """
    from trellis.agent.calibration_contract import hull_white_calibration_contract

    hw_cal = hull_white_calibration_contract(fitting="swaption")

    return CompositeSemanticContract(
        composite_id="callable_bond_composite",
        description=description,
        sub_contracts=(
            SubContractRef(
                contract_id="hw_calibration",
                contract=hw_cal,
                proven=True,
                primitive_ref="trellis.models.trees.lattice.build_rate_lattice",
            ),
            SubContractRef(
                contract_id="bond_cashflows",
                contract=None,  # placeholder — agent generates
                proven=False,
            ),
            SubContractRef(
                contract_id="backward_induction",
                contract=None,  # placeholder — agent generates
                proven=False,
            ),
        ),
        edges=(
            ContractEdge(
                from_contract="hw_calibration",
                to_contract="bond_cashflows",
                edge_type="calibrate_then_price",
                data_flow=("calibrated_lattice", "mean_reversion", "sigma_hw"),
            ),
            ContractEdge(
                from_contract="bond_cashflows",
                to_contract="backward_induction",
                edge_type="sequential",
                data_flow=("coupon_schedule", "call_schedule", "terminal_payoff"),
            ),
        ),
        root_contract="hw_calibration",
        terminal_contracts=("backward_induction",),
    )
