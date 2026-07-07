"""Bridge from the handlers to the internal ``_peloton_tools`` library.

``tools/`` is deliberately not a Python package (the CLIs run it standalone), so we
put it on ``sys.path`` and import ``_peloton_tools`` by its package-unique name — the
same way the CLIs do. Handlers import their tool functions from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from _peloton_tools import copytree, images, pipeline  # noqa: E402,F401
import convert_photos  # noqa: E402,F401  (tools-level module, not under _peloton_tools)

__all__ = ["copytree", "images", "pipeline", "convert_photos"]
