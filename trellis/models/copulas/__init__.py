"""Copula methods for portfolio credit modeling."""

from trellis.models.copulas.correlation import equicorrelation_matrix
from trellis.models.copulas.factor import FactorCopula
from trellis.models.copulas.gaussian import GaussianCopula
from trellis.models.copulas.student_t import StudentTCopula

__all__ = [
    "equicorrelation_matrix",
    "FactorCopula",
    "GaussianCopula",
    "StudentTCopula",
]
