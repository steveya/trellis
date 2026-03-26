# Migration Notes

## Canonical Pricing-Family Imports

Trellis now prefers package-level family imports for the public surface.

- Core domain types: `trellis.core`
- Main model families: `trellis.models`
- QMC accelerators: `trellis.models.qmc`

## QMC Canonical Path

QMC helpers now have a dedicated canonical package:

```python
from trellis.models.qmc import sobol_normals, brownian_bridge
```

The previous low-level implementation imports remain supported:

```python
from trellis.models.monte_carlo.variance_reduction import sobol_normals
from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
```

No behavior changed in those low-level modules. The new package exists to make
QMC a first-class method family in the public API, capability registry, and
agent knowledge system.

## Intentionally Unchanged

- `trellis.models.monte_carlo` remains the home for simulation engines,
  discretization schemes, variance-reduction internals, and exercise logic.
- Existing low-level import paths are still valid.
- No MLMC, BSDE, quantization, or surrogate family was added in this pass.
