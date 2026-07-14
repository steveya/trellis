"""Report canonical helper authority and checked-in adapter delegation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trellis.agent.helper_authority_audit import main


if __name__ == "__main__":
    sys.exit(main())
