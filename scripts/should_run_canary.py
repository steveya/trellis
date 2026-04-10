"""Decide whether the focused canary gate should run for local changes."""

from __future__ import annotations

import sys

from trellis.testing.gates import main


if __name__ == "__main__":
    sys.exit(main())
