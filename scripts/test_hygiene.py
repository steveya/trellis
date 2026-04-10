"""Report stale skip/xfail/quarantine markers in the test suite."""

from __future__ import annotations

import sys

from trellis.testing.hygiene import main


if __name__ == "__main__":
    sys.exit(main())

