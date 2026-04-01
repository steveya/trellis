"""Lattice methods: generic lattice, tree construction, and backward induction."""

from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.trinomial import TrinomialTree
from trellis.models.trees.backward_induction import backward_induction
from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_generic_lattice,
    lattice_backward_induction,
    build_rate_lattice,
    build_spot_lattice,
)
from trellis.models.trees.control import (
    ExerciseObjective,
    LatticeExercisePolicy,
    lattice_step_from_time,
    lattice_steps_from_timeline,
    resolve_lattice_exercise_policy,
)
