"""Put the tools dir on sys.path so tests import the ``_peloton_tools`` library
the same way the CLIs do (they run standalone, no installed package needed)."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "src" / "peloton" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
