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


_DISPATCH: dict[str, Any] = {
    f"{NAMESPACE}.ListImages": handle_list_images,
    f"{NAMESPACE}.ConvertRaw": handle_convert_raw,
}


def handle(payload: dict) -> dict:
    """Single RegistryRunner entrypoint — dispatch on the facet name."""
    return _DISPATCH[payload["_facet_name"]](payload)


def register_handlers(runner) -> None:
    for facet_name in _DISPATCH:
        runner.register_handler(facet_name=facet_name, module_uri=__name__, entrypoint="handle")
