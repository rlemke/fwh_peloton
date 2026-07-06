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
    if p.suffix.lower() in RAW_EXTS:
        return _load_raw(p)                      # camera RAW → demosaiced RGB
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)  # honour camera rotation
    return img.convert("RGB")


# Camera RAW formats decoded via rawpy/LibRaw (orientation applied in postprocess).
RAW_EXTS = {".nef", ".nrw",            # Nikon
            ".cr2", ".cr3", ".crw",     # Canon
            ".arw", ".srf", ".sr2",     # Sony
            ".dng",                     # Adobe / DJI / others
            ".raf",                     # Fujifilm
            ".orf",                     # Olympus
            ".rw2",                     # Panasonic
            ".pef",                     # Pentax
            ".srw"}                     # Samsung


def _load_raw(path: Path) -> Any:
    """Decode a camera RAW file to an 8-bit RGB ``PIL.Image`` (camera white
    balance, auto-brightness, orientation applied by LibRaw)."""
    try:
        import rawpy  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"{path.suffix} is a camera RAW format — needs rawpy. "
            "Install it: pip install '.[raw]'  (or: pip install rawpy)"
        ) from exc
    import numpy as np  # noqa: PLC0415

    Image = _pil()
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    log.debug("decoded RAW %s (%dx%d)", path.name, rgb.shape[1], rgb.shape[0])
    return Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")


def load_image16(path: str | Path, *, highlight_mode: str = "clip") -> Any:
    """Load an image as a 16-bit RGB numpy array (H, W, 3) ``uint16``.

    Camera RAW is decoded straight to 16-bit (``output_bps=16``) so the full
    ~14-bit sensor tonal range survives into the enhancement math — this is what
    makes a heavy auto-brighten/dehaze stretch banding-free. Non-RAW inputs have
    no >8-bit data, so they are lifted 8→16 bit (×257) for a uniform pipeline.
    Orientation is already applied (LibRaw for RAW, EXIF for the rest).

    ``highlight_mode`` (RAW only): ``clip`` (default, LibRaw mode 0), ``blend``
    (gentle rolloff, mode 2) or ``reconstruct`` (mode 5) — recover blown highlights
    from any channel that stayed below clipping. No effect on non-RAW inputs.
    """
    import numpy as np  # noqa: PLC0415

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    if p.suffix.lower() in {".tif", ".tiff"}:            # keep a 16-bit TIFF's real depth
        import tifffile  # noqa: PLC0415
        a = tifffile.imread(str(p))
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        a = a[:, :, :3]
        if a.dtype == np.uint16:
            return np.ascontiguousarray(a)
        if a.dtype == np.uint8:
            return np.ascontiguousarray(a.astype(np.uint16) * 257)
        return np.clip(a, 0, 65535).astype(np.uint16)    # float/other → best-effort
    if p.suffix.lower() in RAW_EXTS:
        try:
            import rawpy  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"{p.suffix} is a camera RAW format — needs rawpy. "
                "Install it: pip install '.[raw]'"
            ) from exc
        hl = {"clip": 0, "blend": 2, "reconstruct": 5}.get(highlight_mode, 0)
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16, highlight_mode=hl,
                                  no_auto_bright=(hl != 0))
        return np.ascontiguousarray(rgb)                 # uint16 HxWx3, oriented
    return (np.asarray(load_image(p), dtype=np.uint16) * 257)  # 8-bit → 16-bit range


def save_tiff16(arr: Any, path: str | Path, dpi: int | None = None) -> Path:
    """Write a 16-bit RGB ``uint16`` (H, W, 3) array as a lossless (deflate) TIFF.
    No DCT/8-bit quantization — the archival, maximum-fidelity output."""
    import tifffile  # noqa: PLC0415

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {"photometric": "rgb", "compression": "deflate"}
    if dpi:
        kw["resolution"] = (dpi, dpi)
    tifffile.imwrite(str(p), arr, **kw)
    log.debug("wrote %s (%dx%d, 16-bit)", p, arr.shape[1], arr.shape[0])
    return p


def tiff_to_image8(path: str | Path) -> Any:
    """Read a TIFF (16-bit or 8-bit) as an 8-bit RGB ``PIL.Image`` — the derive-JPEG
    downconvert. 16-bit is scaled 65535→255 (÷257, rounded); grayscale is expanded
    to RGB and any alpha/extra channels are dropped."""
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    Image = _pil()
    a = tifffile.imread(str(path))
    if a.dtype == np.uint16:
        a = np.clip(np.rint(a / 257.0), 0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a).convert("RGB")
    return Image.fromarray(a[:, :, :3]).convert("RGB")


def save_image(img: Any, path: str | Path, quality: int = 92,
               dpi: int | None = None) -> Path:
    """Save an image, creating parent dirs. JPEG by extension, else PNG-ish.

    dpi — embed a print resolution (both axes). At a given dpi the physical print
    size is pixels/dpi, so a 1200x1800 image tagged 300 dpi prints as 4x6 inches;
    without the tag a print service assumes ~72 dpi and blows it up.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    params: dict[str, Any] = {}
    is_jpeg = p.suffix.lower() in {".jpg", ".jpeg"}
    if is_jpeg:
        params = {"quality": quality, "optimize": True}
        if img.mode != "RGB":
            img = img.convert("RGB")
        # `optimize=True` can overflow libjpeg's per-scanline buffer on very
        # high-entropy images ("broken data stream when writing image file" /
        # "Suspension not allowed here"). Raise MAXBLOCK to cover the whole image.
        from PIL import ImageFile  # noqa: PLC0415
        ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, img.width * img.height)
    if dpi:
        params["dpi"] = (dpi, dpi)
    try:
        img.save(p, **params)
    except OSError:
        if not is_jpeg:
            raise
        params.pop("optimize", None)          # last resort: unoptimized encode
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
