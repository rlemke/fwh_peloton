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
    aspect: float | None = None,
    out_size: tuple[int, int] | None = None,
    frame: str = "single",
    pad_color: str = "white",
    use_mock: bool = False,
    detect_model: str = "yolo11x.pt",
    upscale_backend: str = "auto",
    face_backend: str = "auto",
) -> dict[str, Any]:
    """Process one photo. Returns a summary dict; writes one or more images/rider.

    segment — SAM-mask each rider and cut them out of the background instead of a
    rectangular crop (``cutout_bg``).

    frame — ``single`` (tight rider crop), ``framed`` (expand the crop OUTWARD to
    ``aspect``/``out_size`` — real surrounding pixels, may include other riders,
    no distortion), or ``both``. ``aspect`` = w/h ratio; ``out_size`` = exact
    (w, h) pixels (implies the ratio); ``pad_color`` fills any residual when the
    photo edge is reached (name/#hex/``blur``).
    """
    target_ar = aspect or (out_size[0] / out_size[1] if out_size else None)
    kinds = (["single", "framed"] if frame == "both"
             else ["framed"] if frame == "framed" else ["single"])
    if "framed" in kinds and not target_ar:
        raise ValueError("framed output needs aspect= or out_size=")

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

    def _enhance_crop(crop_img: Any) -> tuple[Any, str, str, bool]:
        """crop → (alpha split) → upscale → face-restore → (alpha reattach)."""
        alpha = crop_img.getchannel("A") if crop_img.mode == "RGBA" else None
        rgb = crop_img.convert("RGB") if alpha is not None else crop_img
        up, ub = _enhance.upscale(rgb, scale=scale, backend=upscale_backend)
        fb = "skipped"
        if restore_faces:
            up, fb = _enhance.restore_faces(up, fidelity=fidelity, backend=face_backend)
        if alpha is not None:
            from PIL import Image  # noqa: PLC0415
            up = up.convert("RGBA")
            up.putalpha(alpha.resize(up.size, Image.LANCZOS))
        return up, ub, fb, alpha is not None

    results: list[dict[str, Any]] = []
    for r in riders:
        base = r.focus_box(pad_frac, w, h)
        outs: list[dict[str, Any]] = []
        for kind in kinds:
            if kind == "single":
                if segment:
                    prompt = _crop.union(r.person_box, r.bike_box)
                    mask = _segment.segment_box(img, prompt, model=sam_model, use_mock=use_mock)
                    crop_img = _crop.cutout(img, mask, base, bg=cutout_bg)
                else:
                    crop_img = img.crop(tuple(int(v) for v in base))
                up, ub, fb, is_rgba = _enhance_crop(crop_img)
            else:  # framed — expand OUTWARD to the target aspect, then size/pad
                abox, needs_pad = _crop.aspect_box(base, target_ar, w, h)
                up, ub, fb, is_rgba = _enhance_crop(img.crop(tuple(int(v) for v in abox)))
                if out_size:
                    up = _images.fit_to_size(up, out_size, color=pad_color); is_rgba = False
                elif needs_pad or abs(up.width / up.height - target_ar) > 0.01:
                    W2, H2 = up.size
                    tgt = ((round(H2 * target_ar), H2) if W2 / H2 < target_ar
                           else (W2, round(W2 / target_ar)))
                    up = _images.fit_to_size(up, tgt, color=pad_color); is_rgba = False

            suffix = f"_{kind}" if len(kinds) > 1 else ""
            out_path = out / f"{src.stem}_rider{r.index:02d}{suffix}.{'png' if is_rgba else 'jpg'}"
            _images.save_image(up, out_path)
            outs.append({"kind": kind, "output": str(out_path),
                         "output_size": list(_images.size(up)),
                         "upscale_backend": ub, "face_backend": fb})
            log.info("rider %02d [%s] → %s", r.index, kind, out_path.name)

        first = outs[0]
        results.append({
            **r.to_dict(),
            "focus_box": [int(v) for v in base],
            "segmented": segment,
            **({"cutout_bg": cutout_bg} if segment else {}),
            "outputs": outs,
            "output": first["output"],          # back-compat: primary output
            "output_size": first["output_size"],
            "upscale_backend": first["upscale_backend"],
            "face_backend": first["face_backend"],
        })

    summary = {
        "source": str(src),
        "source_size": [w, h],
        "n_riders": len(results),
        "out_dir": str(out),
        "riders": results,
    }
    log.info("done: %d rider portrait(s) from %s", len(results), src.name)
    return summary
