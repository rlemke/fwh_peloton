"""End-to-end: a group cycling photo → one enhanced portrait per rider.

    process_photo(path) → detect riders → crop each (person ∪ bike, padded)
                        → upscale → face-restore → write <stem>_riderNN.jpg

Pure orchestration over the ``_peloton_tools`` primitives; returns a
JSON-serializable summary. Output goes to an explicit ``out_dir`` for now — the
sidecar/storage cache backend (agent-spec/cache-layout) is wired in when this
becomes a handler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from _peloton_tools import crop as _crop
from _peloton_tools import detect as _detect
from _peloton_tools import enhance as _enhance
from _peloton_tools import images as _images
from _peloton_tools import segment as _segment

log = logging.getLogger("peloton.pipeline")


def process_photo(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    conf: float = 0.25,
    pad_frac: float = 0.15,
    require_bike: bool = False,
    scale: int = 4,
    restore_faces: bool = True,
    fidelity: float = 0.7,
    segment: bool = False,
    cutout_bg: str = "white",
    sam_model: str = "mobile_sam.pt",
    use_mock: bool = False,
    detect_model: str = "yolov8n.pt",
    upscale_backend: str = "auto",
    face_backend: str = "auto",
) -> dict[str, Any]:
    """Process one photo. Returns a summary dict; writes one image per rider.

    segment — if True, SAM-mask each rider and cut them out of the background
    (``cutout_bg``: white/black/blur/transparent) instead of a rectangular crop.
    """
    src = Path(image_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = _images.load_image(src)
    w, h = _images.size(img)
    log.info("loaded %s (%dx%d)", src.name, w, h)

    riders = _detect.detect_riders(
        img, conf=conf, require_bike=require_bike,
        backend="yolo", model=detect_model, use_mock=use_mock,
    )

    results: list[dict[str, Any]] = []
    for r in riders:
        box = r.focus_box(pad_frac, w, h)
        if segment:
            # prompt SAM with the tight person∪bike box for an accurate mask,
            # then cut the rider out within the padded focus box.
            prompt = _crop.union(r.person_box, r.bike_box)
            mask = _segment.segment_box(img, prompt, model=sam_model, use_mock=use_mock)
            crop_img = _crop.cutout(img, mask, box, bg=cutout_bg)
        else:
            crop_img = img.crop(tuple(int(v) for v in box))

        # A transparent cutout carries an alpha channel; enhance the RGB and
        # re-attach a resized alpha at the end (upscale/face models want RGB).
        alpha = crop_img.getchannel("A") if crop_img.mode == "RGBA" else None
        rgb = crop_img.convert("RGB") if alpha is not None else crop_img

        up_img, up_backend = _enhance.upscale(rgb, scale=scale, backend=upscale_backend)
        face_backend_used = "skipped"
        if restore_faces:
            up_img, face_backend_used = _enhance.restore_faces(
                up_img, fidelity=fidelity, backend=face_backend)

        ext = "jpg"
        if alpha is not None:
            from PIL import Image  # noqa: PLC0415
            up_img = up_img.convert("RGBA")
            up_img.putalpha(alpha.resize(up_img.size, Image.LANCZOS))
            ext = "png"

        out_path = out / f"{src.stem}_rider{r.index:02d}.{ext}"
        _images.save_image(up_img, out_path)
        results.append({
            **r.to_dict(),
            "focus_box": [int(v) for v in box],
            "segmented": segment,
            **({"cutout_bg": cutout_bg} if segment else {}),
            "output": str(out_path),
            "output_size": list(_images.size(up_img)),
            "upscale_backend": up_backend,
            "face_backend": face_backend_used,
        })
        log.info("rider %02d → %s", r.index, out_path.name)

    summary = {
        "source": str(src),
        "source_size": [w, h],
        "n_riders": len(results),
        "out_dir": str(out),
        "riders": results,
    }
    log.info("done: %d rider portrait(s) from %s", len(results), src.name)
    return summary
