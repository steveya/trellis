# Final Refactor Review

## Review Finding

The canonical Monte Carlo-facing cookbook templates still teach direct
`import numpy as np` usage even though the repository guidance for pricing code
prefers:

```python
from trellis.core.differentiable import get_numpy
np = get_numpy()
```

This is not a runtime bug in the library itself, but it is a consistency gap in
the agent-facing policy layer. Since launched agents learn from cookbook
templates, the canonical templates should match the repo's own autograd-friendly
practice.

## Refactor Plan

1. Add one safety test that locks the Monte Carlo and QMC cookbooks to the
   `get_numpy` pattern.
2. Update only the canonical YAML templates.
3. Rerun the cookbook-specific tests and the broader agent suite.
