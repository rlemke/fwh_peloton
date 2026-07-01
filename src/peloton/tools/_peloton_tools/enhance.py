"""Upscale + face restoration, with graceful degradation.

Both stages are *best-effort*: if the ML backend isn't installed the pipeline
still produces a usable output (a Lanczos upscale / the unmodified crop) instead
of failing. Each function returns ``(image, backend_used)`` so callers can record
what actually ran.

Upscale backends (tried in order for backend="auto"):
    realesrgan-ncnn   — the standalone `realesrgan-ncnn-vulkan` binary (no torch)
    realesrgan        — the Python package (needs torch + weights)
    lanczos           — PIL high-quality resample (always available)

Face-restore backends:
    gfpgan / codeformer — if importable
    none                — passthrough (always available)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("peloton.enhance")

_NCNN_BIN = "realesrgan-ncnn-vulkan"


def upscale(img: Any, scale: int = 4, backend: str = "auto") -> tuple[Any, str]:
    """Return an upscaled copy of ``img`` and the backend that produced it."""
    order = ([backend] if backend != "auto"
             else ["realesrgan-ncnn", "realesrgan", "lanczos"])
    for b in order:
        try:
            if b == "realesrgan-ncnn":
                out = _upscale_ncnn(img, scale)
            elif b == "realesrgan":
                out = _upscale_realesrgan_py(img, scale)
            elif b == "lanczos":
                out = _upscale_lanczos(img, scale)
            else:
                raise ValueError(f"unknown upscale backend: {b!r}")
            if out is not None:
                log.info("upscaled x%d via %s (%dx%d → %dx%d)", scale, b,
                         img.width, img.height, out.width, out.height)
                return out, b
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            log.warning("upscale backend %s unavailable/failed: %s", b, exc)
    # order was explicit + failed; last resort so we never return None
    return _upscale_lanczos(img, scale), "lanczos"


def _upscale_lanczos(img: Any, scale: int) -> Any:
    from PIL import Image  # noqa: PLC0415
    return img.resize((img.width * scale, img.height * scale), Image.LANCZOS)


def _upscale_ncnn(img: Any, scale: int) -> Any | None:
    exe = shutil.which(_NCNN_BIN)
    if not exe:
        return None
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "in.png", Path(td) / "out.png"
        img.save(ip)
        # -n picks the x4 model; the tool always produces x4, we resize if needed.
        subprocess.run([exe, "-i", str(ip), "-o", str(op), "-s", "4"],
                       check=True, capture_output=True, timeout=600)
        out = Image.open(op).convert("RGB")
        if scale != 4:
            out = out.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        return out


def _upscale_realesrgan_py(img: Any, scale: int) -> Any | None:
    try:
        import numpy as np  # noqa: PLC0415
        from realesrgan import RealESRGANer  # noqa: PLC0415
    except ImportError:
        return None
    # The Python stack needs a configured model + weights; if the caller hasn't
    # set one up this raises and "auto" falls back. Kept minimal on purpose.
    raise RuntimeError("realesrgan python backend needs an explicit model setup")


def restore_faces(img: Any, fidelity: float = 0.7, backend: str = "auto") -> tuple[Any, str]:
    """Best-effort face restoration. ``fidelity`` (0..1) trades identity-fidelity
    (high) vs restoration-strength (low). Returns ``(image, backend_used)``.
    """
    order = ([backend] if backend != "auto" else ["gfpgan", "none"])
    for b in order:
        try:
            if b == "gfpgan":
                out = _restore_gfpgan(img, fidelity)
            elif b in ("none", "passthrough"):
                out = img
            else:
                raise ValueError(f"unknown face-restore backend: {b!r}")
            if out is not None:
                if b != "none":
                    log.info("face-restore via %s (fidelity=%.2f)", b, fidelity)
                return out, b
        except Exception as exc:  # noqa: BLE001
            log.warning("face-restore backend %s unavailable/failed: %s", b, exc)
    return img, "none"


def _restore_gfpgan(img: Any, fidelity: float) -> Any | None:
    try:
        import gfpgan  # noqa: F401,PLC0415
    except ImportError:
        return None
    # Real GFPGAN wiring (weights + GFPGANer) lands with the enhance extra; kept
    # a stub so "auto" degrades cleanly until it's configured.
    raise RuntimeError("gfpgan backend needs weights/model setup")
