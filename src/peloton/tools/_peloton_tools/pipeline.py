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
    sharpen_framed: float = 130.0,
    dpi: int = 300,
    match_input: bool = False,
    print_sizes: list[tuple[str, float, float]] | None = None,
    auto_brighten: bool = False,
    brighten_target: float = 120.0,
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

    sharpen_framed — unsharp-mask strength (percent) applied to a framed output
    *after* it is fit to the target size. The fit is a downscale of the 4x
    upscaled region, which softens the Real-ESRGAN sharpening; this recovers it.
    0 disables. Only framed outputs are sharpened (the tight ``single`` crop is
    never downscaled, so it needs none).

    dpi — print resolution embedded in framed outputs so they print at their
    intended physical size (e.g. 1200x1800 @ 300 dpi = 4x6"). The tight ``single``
    crop is a variable-size digital view, so it is left untagged.

    match_input — scale every output so its long edge equals the input photo's
    long edge (a 24 MP source → a ~24 MP output). Each framed target and its dpi
    scale by the same factor, so the print size is unchanged (still 4x6"/8x10")
    just at higher resolution; the single crop is up-scaled to the same long edge.
    NOTE: for a small/distant rider this interpolates beyond the detail actually
    captured — big pixels, not new detail.

    print_sizes — a list of ``(label, width_in, height_in)`` print sizes, each
    emitting its own framed output (``<stem>_riderNN_<label>.jpg``) at
    ``inches*dpi`` pixels and that aspect (e.g. ``[("4x6",4,6),("8x10",8,10)]``).
    Overrides ``aspect``/``out_size``. Different sizes have different aspects, so
    each is a genuinely different crop, not a rescale of the others.

    auto_brighten — lift an under-exposed rider via gamma before enhancement.
    Metered on the *rider region* (not the frame — a backlit rider is dark even
    when the frame averages fine) toward ``brighten_target`` mean luminance
    (0..255). No-op when the rider is already bright enough; each output records
    the ``gamma`` applied.
    """
    want_single = frame in ("single", "both")
    want_framed = frame in ("framed", "both")

    src = Path(image_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = _images.load_image(src)
    w, h = _images.size(img)
    log.info("loaded %s (%dx%d)", src.name, w, h)
    long_edge = max(w, h)

    # Build one framed target per requested size: {label, target_ar, out_size, dpi}.
    framed_specs: list[dict[str, Any]] = []
    if want_framed:
        if print_sizes:
            for label, iw, ih in print_sizes:
                framed_specs.append({"label": label, "target_ar": iw / ih,
                                     "out_size": (max(1, round(iw * dpi)),
                                                  max(1, round(ih * dpi))), "dpi": dpi})
        else:
            ar = aspect or (out_size[0] / out_size[1] if out_size else None)
            if not ar:
                raise ValueError("framed output needs aspect=, out_size=, or print_sizes=")
            framed_specs.append({"label": "framed", "target_ar": ar,
                                 "out_size": out_size, "dpi": dpi})
        if match_input:  # scale each target's pixels + dpi so its long edge == input
            for sp in framed_specs:
                osz = sp["out_size"]
                if osz:
                    k = long_edge / max(osz)
                    sp["out_size"] = (max(1, round(osz[0] * k)), max(1, round(osz[1] * k)))
                    sp["dpi"] = max(1, round(sp["dpi"] * k))
                else:  # aspect-only → size the long dimension to the input long edge
                    ar = sp["target_ar"]
                    sp["out_size"] = ((round(long_edge * ar), long_edge) if ar < 1
                                      else (long_edge, round(long_edge / ar)))
                log.info("match_input: framed[%s] → %dx%d @ %d dpi", sp["label"],
                         sp["out_size"][0], sp["out_size"][1], sp["dpi"])
    n_kinds = (1 if want_single else 0) + len(framed_specs)

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

    def _sharpen(im: Any) -> Any:
        """Unsharp-mask to recover crispness lost in the fit-to-size downscale."""
        if sharpen_framed <= 0:
            return im
        from PIL import ImageFilter  # noqa: PLC0415
        mask = ImageFilter.UnsharpMask(radius=2.2, percent=int(sharpen_framed), threshold=2)
        if im.mode == "RGBA":  # sharpen colour, keep alpha
            a = im.getchannel("A")
            im = im.convert("RGB").filter(mask).convert("RGBA")
            im.putalpha(a)
            return im
        return im.filter(mask)

    def _to_long_edge(im: Any) -> Any:
        """Scale so the long edge == the input long edge (match_input)."""
        le = max(im.size)
        if le == long_edge:
            return im
        from PIL import Image  # noqa: PLC0415
        s = long_edge / le
        return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                         Image.LANCZOS)

    def _emit(up: Any, label: str, is_rgba: bool, out_dpi: int | None,
              ub: str, fb: str, gamma: float, r: Any, outs: list[dict[str, Any]]) -> None:
        suffix = f"_{label}" if n_kinds > 1 else ""
        out_path = out / f"{src.stem}_rider{r.index:02d}{suffix}.{'png' if is_rgba else 'jpg'}"
        _images.save_image(up, out_path, dpi=out_dpi)
        outs.append({"kind": "single" if label == "single" else "framed", "label": label,
                     "output": str(out_path), "output_size": list(_images.size(up)),
                     "upscale_backend": ub, "face_backend": fb, "brighten_gamma": gamma})
        log.info("rider %02d [%s] → %s%s", r.index, label, out_path.name,
                 f" (brighten {gamma:.2f})" if gamma > 1.0 else "")

    results: list[dict[str, Any]] = []
    for r in riders:
        base = r.focus_box(pad_frac, w, h)
        outs: list[dict[str, Any]] = []
        # Meter exposure on the rider region (not the frame) so a backlit rider
        # is corrected even when the frame averages fine; one gamma per rider.
        meter = img.crop(tuple(int(v) for v in r.focus_box(0.0, w, h))) if auto_brighten else None

        if want_single:
            if segment:
                prompt = _crop.union(r.person_box, r.bike_box)
                mask = _segment.segment_box(img, prompt, model=sam_model, use_mock=use_mock)
                crop_img = _crop.cutout(img, mask, base, bg=cutout_bg)
            else:
                crop_img = img.crop(tuple(int(v) for v in base))
            crop_img, gamma = (_enhance.auto_brighten(crop_img, meter=meter, target=brighten_target)
                               if auto_brighten else (crop_img, 1.0))
            up, ub, fb, is_rgba = _enhance_crop(crop_img)
            if match_input:
                upscaled = max(up.size) < long_edge
                up = _to_long_edge(up)                 # long edge == input (up or down)
                if upscaled:
                    up = _sharpen(up)                  # interpolated up → recover crispness
            _emit(up, "single", is_rgba, None, ub, fb, gamma, r, outs)

        for sp in framed_specs:  # each print size: expand OUTWARD to its aspect, then size/pad
            ar, osz = sp["target_ar"], sp["out_size"]
            abox, needs_pad = _crop.aspect_box(base, ar, w, h)
            region_crop = img.crop(tuple(int(v) for v in abox))
            region_crop, gamma = (_enhance.auto_brighten(region_crop, meter=meter, target=brighten_target)
                                  if auto_brighten else (region_crop, 1.0))
            up, ub, fb, is_rgba = _enhance_crop(region_crop)
            fitted = False
            if osz:
                up = _images.fit_to_size(up, osz, color=pad_color)
                is_rgba, fitted = False, True
            elif needs_pad or abs(up.width / up.height - ar) > 0.01:
                W2, H2 = up.size
                tgt = ((round(H2 * ar), H2) if W2 / H2 < ar else (W2, round(W2 / ar)))
                up = _images.fit_to_size(up, tgt, color=pad_color)
                is_rgba, fitted = False, True
            if fitted:
                up = _sharpen(up)
            _emit(up, sp["label"], is_rgba, sp["dpi"], ub, fb, gamma, r, outs)

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
