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


_heif_registered = False


def _ensure_heif() -> None:
    """Register the HEIF/HEIC opener with Pillow (once) so iPhone .heic photos
    open transparently. No-op if pillow-heif isn't installed (HEIC just won't
    load, with Pillow's normal UnidentifiedImageError)."""
    global _heif_registered
    if _heif_registered:
        return
    _heif_registered = True
    try:
        import pillow_heif  # noqa: PLC0415
        pillow_heif.register_heif_opener()
        log.debug("registered HEIF/HEIC opener")
    except ImportError:
        log.debug("pillow-heif not installed — .heic files won't open")


def load_image(path: str | Path) -> Any:
    """Load an image as an RGB ``PIL.Image``. EXIF-orientation is applied so
    portraits/landscapes come out upright."""
    Image = _pil()
    from PIL import ImageOps  # noqa: PLC0415

    _ensure_heif()  # so .heic / .heif open like any other format
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


def fit_to_size(img: Any, target: tuple[int, int], color: str = "white") -> Any:
    """Scale ``img`` to exactly ``target`` (w, h) preserving aspect — never
    distorting — padding any residual with ``color`` (a name/#hex, or ``blur``)."""
    Image = _pil()
    from PIL import ImageColor, ImageOps  # noqa: PLC0415

    tw, th = target
    if color == "blur":
        base = ImageOps.fit(img.convert("RGB"), (tw, th), method=Image.LANCZOS)
        base = base.filter(_gaussian(max(12, min(tw, th) // 12)))
        fitted = ImageOps.contain(img.convert("RGB"), (tw, th), method=Image.LANCZOS)
        base.paste(fitted, ((tw - fitted.width) // 2, (th - fitted.height) // 2))
        return base
    try:
        fill = ImageColor.getrgb(color)
    except ValueError:
        fill = (255, 255, 255)
    return ImageOps.pad(img.convert("RGB"), (tw, th), method=Image.LANCZOS,
                        color=fill, centering=(0.5, 0.5))


def _gaussian(radius: int):
    from PIL import ImageFilter  # noqa: PLC0415
    return ImageFilter.GaussianBlur(radius)
