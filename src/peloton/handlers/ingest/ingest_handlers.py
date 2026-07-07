"""Handlers for peloton.Ingest — list images, convert RAW → full-res 16-bit TIFF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from peloton.handlers.shared import peloton_utils as U

NAMESPACE = "peloton.Ingest"

_IMG_EXTS = U.images.RAW_EXTS | {".jpg", ".jpeg", ".png", ".webp", ".bmp",
                                 ".tif", ".tiff", ".heic", ".heif"}


def handle_list_images(params: dict[str, Any]) -> dict[str, Any]:
    d = Path(params["in_dir"]).expanduser()
    paths = sorted(str(p) for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    return {"paths": paths, "count": len(paths)}


def handle_convert_raw(params: dict[str, Any]) -> dict[str, Any]:
    src = Path(params["image_path"]).expanduser()
    out_dir = Path(params["out_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = U.images.load_image16(src, highlight_mode=params.get("highlight_mode", "clip"))
    out = out_dir / f"{src.stem}.tif"
    U.images.save_tiff16(arr, out)
    return {"output": str(out)}


def handle_convert_tree(params: dict[str, Any]) -> dict[str, Any]:
    """Multi-threaded whole-directory/tree convert (RAW/TIFF/JPEG → TIFF/JPEG)."""
    cp = U.convert_photos
    in_dir = Path(params["in_dir"]).expanduser()
    out_dir = Path(params["out_dir"]).expanduser()
    resize = cp.parse_resize(params.get("resize") or None)
    common = dict(highlight_mode=params.get("highlight_mode", "clip"),
                  out_format=params.get("out_format", "tif"), resize=resize,
                  quality=int(params.get("quality", 95)), resume=params.get("resume", True),
                  workers=params.get("workers", "auto"))
    if params.get("recursive", True):
        exts = cp.parse_from(params.get("from_sel") or "raw", default=set(U.images.RAW_EXTS))
        s = cp.convert_tree(in_dir, out_dir, exts=exts, **common)
    else:
        in_exts = cp.parse_from(params.get("from_sel"), default=None) \
            if params.get("from_sel") else None
        s = cp.convert_dir(in_dir, out_dir, in_exts=in_exts, **common)
    return {"converted": s["converted"], "skipped": s["skipped"], "failed": s["failed"]}


def handle_copy_tree(params: dict[str, Any]) -> dict[str, Any]:
    """Multi-threaded recursive directory copy (structure mirrored, restart-safe)."""
    dst = Path(params["dst"]).expanduser()
    s = U.copytree.copy_tree(params["src"], dst, workers=params.get("workers", "auto"),
                             manifest_path=dst / "_copy_manifest.json")
    return {"copied": s["copied"], "skipped": s["skipped"], "failed": s["failed"]}


_DISPATCH: dict[str, Any] = {
    f"{NAMESPACE}.ListImages": handle_list_images,
    f"{NAMESPACE}.ConvertRaw": handle_convert_raw,
    f"{NAMESPACE}.ConvertTree": handle_convert_tree,
    f"{NAMESPACE}.CopyTree": handle_copy_tree,
}


def handle(payload: dict) -> dict:
    """Single RegistryRunner entrypoint — dispatch on the facet name."""
    return _DISPATCH[payload["_facet_name"]](payload)


def register_handlers(runner) -> None:
    for facet_name in _DISPATCH:
        runner.register_handler(facet_name=facet_name, module_uri=__name__, entrypoint="handle")
