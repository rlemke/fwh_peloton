"""Pure box geometry + cropping — no models, no I/O beyond the passed image.

A ``Box`` is an axis-aligned ``(x1, y1, x2, y2)`` in pixel coordinates with
``x1<=x2`` and ``y1<=y2``. All functions are pure and deterministic, so this is
the module that carries the crop-correctness unit tests.
"""

from __future__ import annotations

from typing import Any

Box = tuple[float, float, float, float]


def parse_aspect(s: str) -> float:
    """``'4:5'`` / ``'4/5'`` / ``'0.8'`` → aspect ratio (w/h)."""
    s = s.strip()
    for sep in (":", "/"):
        if sep in s:
            a, b = s.split(sep)
            return float(a) / float(b)
    return float(s)


def parse_size(s: str) -> tuple[int, int]:
    """``'1080x1350'`` → ``(1080, 1350)``."""
    w, h = s.lower().replace(" ", "").split("x")
    return int(w), int(h)


def parse_print_sizes(s: str) -> list[tuple[str, float, float]]:
    """``'4x6,8x10'`` → ``[('4x6', 4.0, 6.0), ('8x10', 8.0, 10.0)]`` (label, w_in, h_in)."""
    out: list[tuple[str, float, float]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        wi, hi = tok.lower().replace(" ", "").split("x")
        out.append((tok, float(wi), float(hi)))
    return out


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


def intersects(a: Box, b: Box) -> bool:
    """True if two boxes overlap with positive area."""
    return (max(a[0], b[0]) < min(a[2], b[2])
            and max(a[1], b[1]) < min(a[3], b[3]))


def context_box(base: Box, others: list[Box], width: int, height: int,
                *, reach: float = 0.6, margin: float = 0.08) -> Box:
    """A wider crop around ``base`` that also takes in nearby riders — no target
    aspect ratio, the box is simply whatever encloses the rider and their
    neighbours.

    Expand ``base`` by ``reach`` (fraction of its own size) into a neighbourhood,
    union in every ``others`` box that overlaps that neighbourhood, then add a
    small ``margin`` and clamp. With no neighbours nearby it degrades to just a
    wider, contextual view of the one rider (the neighbourhood itself).
    """
    neigh = pad_box(base, reach, width, height)
    boxes: list[Box] = [neigh]
    for o in others:
        if o is not None and intersects(neigh, o):
            boxes.append(o)
    return pad_box(union(*boxes), margin, width, height)


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


def aspect_box(box: Box, target_ar: float, width: int, height: int) -> tuple[Box, bool]:
    """Grow ``box`` OUTWARD to the aspect ratio ``target_ar`` (= w/h), centred on
    the box, then slid to stay inside the image — so a rider keeps their true
    proportions and the extra area is filled with real surrounding pixels (which
    may include other riders), not distortion.

    Returns ``(new_box, needs_pad)``. ``needs_pad`` is True when the image itself
    is too small to reach ``target_ar`` (the caller pads the remainder).
    """
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if bw / bh < target_ar:          # too tall → widen
        nw, nh = bh * target_ar, bh
    else:                            # too wide → heighten
        nw, nh = bw, bw / target_ar
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1, nx2, ny2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if nx1 < 0:
        nx2 -= nx1; nx1 = 0.0
    if ny1 < 0:
        ny2 -= ny1; ny1 = 0.0
    if nx2 > width:
        nx1 -= (nx2 - width); nx2 = float(width)
    if ny2 > height:
        ny1 -= (ny2 - height); ny2 = float(height)
    needs_pad = (nw > width + 0.5) or (nh > height + 0.5)
    return clamp_box((max(0.0, nx1), max(0.0, ny1), nx2, ny2), width, height), needs_pad


def cutout(img: Any, mask: Any, box: Box, *, bg: str = "white", feather: int = 2) -> Any:
    """Crop to ``box`` and knock out everything outside ``mask`` (a boolean
    ``HxW`` array over the full image).

    ``bg``:
      * ``transparent`` → RGBA with the mask as alpha (PNG)
      * ``blur`` / ``bokeh`` → the crop's own background, gaussian-blurred (bokeh
        is a stronger, size-relative blur for a portrait look)
      * ``white`` / ``black`` / a colour (``#RRGGBB`` or a PIL colour name) → solid
      * a path to an image file → that image, cover-fit behind the rider
    ``feather`` softens the mask edge (px).
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
    return Image.composite(sub, _background(sub, bg), alpha)


def _background(sub: Any, bg: str) -> Any:
    """The RGB backdrop for a cutout — see ``cutout``'s ``bg`` options."""
    import os  # noqa: PLC0415

    from PIL import Image, ImageColor, ImageFilter, ImageOps  # noqa: PLC0415

    if bg == "blur":
        return sub.filter(ImageFilter.GaussianBlur(12))
    if bg == "bokeh":
        radius = max(10, min(sub.size) // 12)      # heavier, size-relative
        return sub.filter(ImageFilter.GaussianBlur(radius))
    if isinstance(bg, str) and os.path.isfile(bg):
        return ImageOps.fit(Image.open(bg).convert("RGB"), sub.size,
                            method=Image.LANCZOS)   # cover-fit the replacement
    if bg == "white":
        color: Any = (255, 255, 255)
    elif bg == "black":
        color = (0, 0, 0)
    else:
        try:
            color = ImageColor.getrgb(bg)           # #RRGGBB or a colour name
        except ValueError:
            color = (255, 255, 255)
    return Image.new("RGB", sub.size, color)
