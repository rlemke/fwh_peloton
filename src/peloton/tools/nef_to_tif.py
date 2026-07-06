#!/usr/bin/env python3
"""Back-compat alias for **convert-photos**, defaulting to RAW → 16-bit TIFF.

The general converter (RAW/TIFF/JPEG in → TIFF/JPEG out, ``--resize``, ``--format``)
now lives in ``convert_photos.py``; this thin shim keeps the old ``nef-to-tif`` name
and re-exports its functions so existing callers keep working.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_photos import (  # noqa: E402,F401
    convert_dir, convert_one, convert_tree, main, _resolve_workers)

if __name__ == "__main__":
    sys.exit(main())
