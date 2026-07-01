"""Image load / save / normalize — a thin Pillow wrapper.

Pillow is a core dep, but imported lazily behind a clear error so the module
can be imported (for typing / the mock path) even in a stripped environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("peloton.images")


def _pil():
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Pillow is required. Install it: pip install pillow"
        ) from exc
    return Image


def load_image(path: str | Path) -> Any:
    """Load an image as an RGB ``PIL.Image``. EXIF-orientation is applied so
    portraits/landscapes come out upright."""
    Image = _pil()
    from PIL import ImageOps  # noqa: PLC0415

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)  # honour camera rotation
    return img.convert("RGB")


def save_image(img: Any, path: str | Path, quality: int = 92) -> Path:
    """Save an image, creating parent dirs. JPEG by extension, else PNG-ish."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {}
    if p.suffix.lower() in {".jpg", ".jpeg"}:
        params = {"quality": quality, "optimize": True}
        if img.mode != "RGB":
            img = img.convert("RGB")
    img.save(p, **params)
    log.debug("wrote %s (%dx%d)", p, img.width, img.height)
    return p


def size(img: Any) -> tuple[int, int]:
    """(width, height)."""
    return int(img.width), int(img.height)
