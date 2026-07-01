"""peloton — split group cycling photos into per-rider enhanced portraits.

Pipeline (each stage a standalone ``_peloton_tools`` primitive + CLI):

    load  →  detect riders (YOLO)  →  crop per-rider  →  upscale (Real-ESRGAN)
          →  face-restore (CodeFormer/GFPGAN)  →  per-rider output images

This first cut ships the reusable **library + tools** only. The
``facetwork.domains`` entry point, ``handlers/`` and ``ffl/`` (which turn these
tools into an FFL workflow) are the next phase — see ``tools/README.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"
