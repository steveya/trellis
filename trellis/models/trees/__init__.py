"""Lattice methods: generic lattice, tree construction, and backward induction."""

from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.trinomial import TrinomialTree
from trellis.models.trees.backward_induction import backward_induction
from trellis.models.trees.lattice import (
    RecombiningLattice,
    lattice_backward_induction,
    build_rate_lattice,
    build_spot_lattice,
)
