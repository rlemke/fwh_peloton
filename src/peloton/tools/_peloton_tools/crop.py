"""Pure box geometry + cropping — no models, no I/O beyond the passed image.

A ``Box`` is an axis-aligned ``(x1, y1, x2, y2)`` in pixel coordinates with
``x1<=x2`` and ``y1<=y2``. All functions are pure and deterministic, so this is
the module that carries the crop-correctness unit tests.
"""

from __future__ import annotations

from typing import Any

Box = tuple[float, float, float, float]


def clamp_box(box: Box, width: int, height: int) -> Box:
    """Clamp a box to the image bounds and to integer pixels."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width))
    y1 = max(0, min(int(round(y1)), height))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def union(*boxes: Box) -> Box:
    """Smallest box enclosing all inputs. Ignores ``None`` entries."""
    real = [b for b in boxes if b is not None]
    if not real:
        raise ValueError("union() needs at least one box")
    xs1 = min(b[0] for b in real)
    ys1 = min(b[1] for b in real)
    xs2 = max(b[2] for b in real)
    ys2 = max(b[3] for b in real)
    return (xs1, ys1, xs2, ys2)


def area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes (0..1)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def pad_box(box: Box, pad_frac: float, width: int, height: int) -> Box:
    """Expand a box by ``pad_frac`` of its own size on each side, then clamp to
    the image. ``pad_frac=0.15`` adds 15% margin so heads/wheels aren't clipped.
    """
    x1, y1, x2, y2 = box
    dw = (x2 - x1) * pad_frac
    dh = (y2 - y1) * pad_frac
    return clamp_box((x1 - dw, y1 - dh, x2 + dw, y2 + dh), width, height)


def crop_box(img: Any, box: Box) -> Any:
    """Crop ``img`` (a ``PIL.Image``) to an integer, in-bounds box."""
    x1, y1, x2, y2 = clamp_box(box, img.width, img.height)
    return img.crop((x1, y1, x2, y2))


def cutout(img: Any, mask: Any, box: Box, *, bg: str = "white", feather: int = 2) -> Any:
    """Crop to ``box`` and knock out everything outside ``mask`` (a boolean
    ``HxW`` array over the full image).

    ``bg``: ``white`` / ``black`` / ``blur`` → composited RGB; ``transparent`` →
    RGBA with the mask as alpha. ``feather`` softens the mask edge (px).
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageFilter  # noqa: PLC0415

    x1, y1, x2, y2 = clamp_box(box, img.width, img.height)
    sub = img.crop((x1, y1, x2, y2)).convert("RGB")
    sub_mask = np.ascontiguousarray(mask[y1:y2, x1:x2]).astype("uint8") * 255
    alpha = Image.fromarray(sub_mask, mode="L")
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))

    if bg == "transparent":
        out = sub.convert("RGBA")
        out.putalpha(alpha)
        return out
    if bg == "blur":
        base = sub.filter(ImageFilter.GaussianBlur(12))
    else:
        color = (0, 0, 0) if bg == "black" else (255, 255, 255)
        base = Image.new("RGB", sub.size, color)
    return Image.composite(sub, base, alpha)
