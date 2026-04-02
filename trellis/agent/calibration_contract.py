"""Calibration contract — typed calibration step as a DSL primitive.

Declares what needs to be calibrated, routes to a proven calibration
primitive, and produces a typed ``CalibrationResult`` that the pricing
code consumes.  The agent never generates calibration code inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationTarget:
    """What parameter to fit."""

    parameter: str        # "short_rate_mean_reversion", "vol_surface", "hazard_rate", "sabr_alpha_rho_nu"
    model_family: str     # "hull_white", "sabr", "local_vol", "black76"
    description: str = ""


@dataclass(frozen=True)
class FittingInstrument:
    """What market instrument to match during calibration."""

    instrument_type: str  # "swaption", "cap", "cds_spread", "option"
    data_source: str      # "market_quote", "implied_from_surface", "bootstrap"
    parameters: tuple[str, ...] = ()  # e.g. ("expiry", "tenor", "strike")


@dataclass(frozen=True)
class CalibrationMethod:
    """How to perform the calibration."""

    optimizer: str = "analytical"      # "analytical", "least_squares", "differential_evolution", "brent"
    max_iterations: int = 1000
    convergence_tol: float = 1e-8
    stability_checks: tuple[str, ...] = ()  # "repricing_residual", "parameter_bounds", "gradient_norm"


@dataclass(frozen=True)
class OutputBinding:
    """How calibrated parameters flow into the pricing step."""

    target_path: str                       # "market_state.vol_surface", "lattice.mean_reversion"
    parameter_names: tuple[str, ...] = ()  # what the calibration produces
    consumption_pattern: str = "inject_parameter"  # "replace_field", "inject_parameter", "build_lattice"


@dataclass(frozen=True)
class CalibrationContract:
    """Declares a calibration step as a DSL primitive."""

    target: CalibrationTarget
    fitting_instruments: tuple[FittingInstrument, ...]
    method: CalibrationMethod = field(default_factory=CalibrationMethod)
    acceptance_criteria: tuple[str, ...] = ("repricing_residual < 1e-6",)
    output: OutputBinding = field(default_factory=lambda: OutputBinding(target_path=""))
    proven_primitive: str = ""   # e.g. "calibrate_swaption_black_vol", "build_lattice"
    depends_on: str = ""         # another CalibrationContract ID for chaining
    description: str = ""


@dataclass(frozen=True)
class CalibrationResult:
    """Typed output of a calibration step."""

    target: CalibrationTarget
    calibrated_parameters: dict[str, float] = field(default_factory=dict)
    residual: float = 0.0
    provenance: dict[str, object] = field(default_factory=dict)
    accepted: bool = True

    def __post_init__(self):
        """Freeze mutable dicts for safety."""
        from types import MappingProxyType
        if isinstance(self.calibrated_parameters, dict):
            object.__setattr__(self, "calibrated_parameters", MappingProxyType(self.calibrated_parameters))
        if isinstance(self.provenance, dict):
            object.__setattr__(self, "provenance", MappingProxyType(self.provenance))


# ---------------------------------------------------------------------------
# Known targets and primitives
# ---------------------------------------------------------------------------

_KNOWN_TARGETS = {
    "short_rate_mean_reversion": "hull_white",
    "short_rate_sigma": "hull_white",
    "vol_surface": "black76",
    "flat_vol": "black76",
    "sabr_alpha_rho_nu": "sabr",
    "local_vol_surface": "local_vol",
    "hazard_rate": "credit",
}

_KNOWN_PRIMITIVES = {
    "build_lattice": "trellis.models.trees.algebra",
    "build_rate_lattice": "trellis.models.trees.lattice",
    "calibrate_swaption_black_vol": "trellis.models.calibration",
    "calibrate_cap_floor_black_vol": "trellis.models.calibration",
    "calibrate_sabr": "trellis.models.calibration.sabr_fit",
    "dupire_local_vol": "trellis.models.calibration.local_vol",
}

_KNOWN_OPTIMIZERS = frozenset({
    "analytical", "least_squares", "differential_evolution", "brent", "lbfgsb",
})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_calibration_contract(contract: CalibrationContract) -> tuple[str, ...]:
    """Validate a ``CalibrationContract`` for structural correctness.

    Returns a tuple of error strings (empty if valid).
    """
    errors: list[str] = []

    # Target parameter must be known
    if contract.target.parameter not in _KNOWN_TARGETS:
        errors.append(
            f"Unknown calibration target parameter: '{contract.target.parameter}'. "
            f"Known: {sorted(_KNOWN_TARGETS)}"
        )
    elif contract.target.model_family != _KNOWN_TARGETS[contract.target.parameter]:
        expected = _KNOWN_TARGETS[contract.target.parameter]
        errors.append(
            f"Target parameter '{contract.target.parameter}' belongs to model family "
            f"'{expected}', not '{contract.target.model_family}'"
        )

    # Must have at least one fitting instrument
    if not contract.fitting_instruments:
        errors.append("CalibrationContract requires at least one fitting instrument")

    # Optimizer must be known
    if contract.method.optimizer not in _KNOWN_OPTIMIZERS:
        errors.append(
            f"Unknown optimizer: '{contract.method.optimizer}'. "
            f"Known: {sorted(_KNOWN_OPTIMIZERS)}"
        )

    # Proven primitive must be known (if specified)
    if contract.proven_primitive and contract.proven_primitive not in _KNOWN_PRIMITIVES:
        errors.append(
            f"Unknown proven primitive: '{contract.proven_primitive}'. "
            f"Known: {sorted(_KNOWN_PRIMITIVES)}"
        )

    # Output binding must have a target path
    if contract.output.target_path == "":
        errors.append("OutputBinding.target_path must be non-empty")

    return tuple(errors)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def hull_white_calibration_contract(
    *,
    fitting: str = "swaption",
) -> CalibrationContract:
    """Build the canonical Hull-White rate-tree calibration contract.

    Maps to the unified ``build_lattice()`` surface with a short-rate lattice
    model specification and term-structure calibration target.
    """
    if fitting == "swaption":
        instruments = (
            FittingInstrument(
                instrument_type="swaption",
                data_source="market_quote",
                parameters=("expiry", "tenor", "strike"),
            ),
        )
    elif fitting == "cap":
        instruments = (
            FittingInstrument(
                instrument_type="cap",
                data_source="market_quote",
                parameters=("expiry", "strike"),
            ),
        )
    else:
        instruments = (
            FittingInstrument(
                instrument_type=fitting,
                data_source="market_quote",
            ),
        )

    return CalibrationContract(
        target=CalibrationTarget(
            parameter="short_rate_mean_reversion",
            model_family="hull_white",
            description="Hull-White short-rate mean reversion and volatility",
        ),
        fitting_instruments=instruments,
        method=CalibrationMethod(
            optimizer="analytical",
            stability_checks=("parameter_bounds",),
        ),
        acceptance_criteria=("repricing_residual < 1e-6",),
        output=OutputBinding(
            target_path="lattice",
            parameter_names=("mean_reversion", "sigma_hw"),
            consumption_pattern="build_lattice",
        ),
        proven_primitive="build_lattice",
        description=f"Hull-White calibration via {fitting} fitting (Brigo-Mercurio analytical)",
    )


def sabr_smile_calibration_contract() -> CalibrationContract:
    """Build the canonical SABR smile calibration contract.

    Maps to ``calibrate_sabr()`` which uses L-BFGS-B optimization.
    """
    return CalibrationContract(
        target=CalibrationTarget(
            parameter="sabr_alpha_rho_nu",
            model_family="sabr",
            description="SABR stochastic volatility parameters (alpha, rho, nu)",
        ),
        fitting_instruments=(
            FittingInstrument(
                instrument_type="option",
                data_source="implied_from_surface",
                parameters=("strike", "expiry", "implied_vol"),
            ),
        ),
        method=CalibrationMethod(
            optimizer="lbfgsb",
            stability_checks=("parameter_bounds", "gradient_norm"),
        ),
        acceptance_criteria=("repricing_residual < 1e-4",),
        output=OutputBinding(
            target_path="process",
            parameter_names=("alpha", "rho", "nu"),
            consumption_pattern="inject_parameter",
        ),
        proven_primitive="calibrate_sabr",
        description="SABR smile calibration via L-BFGS-B (Hagan et al.)",
    )


def black76_flat_vol_calibration_contract(
    *,
    fitting: str = "cap",
) -> CalibrationContract:
    """Build the canonical Black76 flat-vol calibration contract.

    Maps to ``calibrate_cap_floor_black_vol()`` or
    ``calibrate_swaption_black_vol()``.
    """
    primitive = (
        "calibrate_cap_floor_black_vol" if fitting == "cap"
        else "calibrate_swaption_black_vol"
    )
    return CalibrationContract(
        target=CalibrationTarget(
            parameter="flat_vol",
            model_family="black76",
            description="Black76 flat implied volatility",
        ),
        fitting_instruments=(
            FittingInstrument(
                instrument_type=fitting,
                data_source="market_quote",
                parameters=("expiry", "strike", "target_price"),
            ),
        ),
        method=CalibrationMethod(
            optimizer="brent",
            stability_checks=("repricing_residual",),
        ),
        acceptance_criteria=("repricing_residual < 1e-8",),
        output=OutputBinding(
            target_path="market_state.vol_surface",
            parameter_names=("calibrated_vol",),
            consumption_pattern="replace_field",
        ),
        proven_primitive=primitive,
        description=f"Black76 flat-vol calibration via Brent root-finding ({fitting} fitting)",
    )
