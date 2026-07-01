#!/usr/bin/env python3
"""process-photo — group cycling photo → one enhanced portrait per rider.

Usage:
    process_photo.py --image group.jpg --out-dir out/
    process_photo.py --image group.jpg --out-dir out/ --require-bike --scale 4
    process_photo.py --image group.jpg --out-dir out/ --use-mock   # offline, no models

Detects each rider (YOLO person+bicycle), crops the rider (person ∪ bike, padded),
upscales (Real-ESRGAN if available, else Lanczos) and face-restores (GFPGAN if
available, else passthrough). Prints a JSON summary on stdout; logs on stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import crop, pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a group cycling photo into per-rider enhanced portraits.")
    ap.add_argument("--image", required=True, help="input photo path")
    ap.add_argument("--out-dir", required=True, help="directory for per-rider outputs")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--pad", type=float, default=0.15, help="crop padding fraction (0.15 = 15%%)")
    ap.add_argument("--require-bike", action="store_true", help="drop persons with no paired bicycle")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor")
    ap.add_argument("--aspect", help="framed output aspect W:H (e.g. 4:5) — expands the crop to fit")
    ap.add_argument("--size", help="framed output exact pixels WxH (e.g. 1080x1350; implies aspect)")
    ap.add_argument("--print-sizes",
                    help="one framed output per print size, e.g. '4x6,8x10' (inches; overrides --size/--aspect)")
    ap.add_argument("--frame", choices=["single", "framed", "both"], default=None,
                    help="which outputs: single (tight), framed (fixed size), or both. "
                         "Default single, or framed when --aspect/--size is given.")
    ap.add_argument("--pad-color", default="white", help="fill at the photo edge: name|#hex|blur")
    ap.add_argument("--sharpen-framed", type=float, default=130.0,
                    help="unsharp %% on framed outputs after fit-to-size (0 disables)")
    ap.add_argument("--dpi", type=int, default=300,
                    help="print DPI embedded in framed outputs (1200x1800 @ 300 = 4x6\")")
    ap.add_argument("--match-input", action="store_true",
                    help="scale outputs so long edge = input long edge (~input MP); dpi scales too")
    ap.add_argument("--auto-brighten", action="store_true",
                    help="lighten under-exposed/backlit riders (metered on the rider, gamma, no-op if bright)")
    ap.add_argument("--brighten-target", type=float, default=120.0,
                    help="target rider mean luminance 0..255 for --auto-brighten (default 120)")
    ap.add_argument("--segment", action="store_true", help="SAM cutout each rider (mask, not bbox)")
    ap.add_argument("--cutout-bg", default="white", help="segment background: white|black|blur|transparent")
    ap.add_argument("--sam-model", default="mobile_sam.pt", help="SAM weights")
    ap.add_argument("--no-face-restore", action="store_true", help="skip face restoration")
    ap.add_argument("--fidelity", type=float, default=0.7, help="face-restore identity fidelity (0..1)")
    ap.add_argument("--model", default="yolo11x.pt", help="YOLO weights")
    ap.add_argument("--upscale-backend", default="auto", help="auto|realesrgan-ncnn|realesrgan|lanczos")
    ap.add_argument("--face-backend", default="auto", help="auto|gfpgan|codeformer|none")
    ap.add_argument("--use-mock", action="store_true", help="offline deterministic detector (no models)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    aspect = crop.parse_aspect(a.aspect) if a.aspect else None
    out_size = crop.parse_size(a.size) if a.size else None
    print_sizes = crop.parse_print_sizes(a.print_sizes) if a.print_sizes else None
    frame = a.frame or ("both" if print_sizes else
                        "framed" if (aspect or out_size) else "single")
    try:
        summary = pipeline.process_photo(
            a.image, a.out_dir, conf=a.conf, pad_frac=a.pad,
            require_bike=a.require_bike, scale=a.scale,
            aspect=aspect, out_size=out_size, frame=frame, pad_color=a.pad_color,
            print_sizes=print_sizes,
            auto_brighten=a.auto_brighten, brighten_target=a.brighten_target,
            sharpen_framed=a.sharpen_framed, dpi=a.dpi, match_input=a.match_input,
            segment=a.segment, cutout_bg=a.cutout_bg, sam_model=a.sam_model,
            restore_faces=not a.no_face_restore, fidelity=a.fidelity,
            use_mock=a.use_mock, detect_model=a.model,
            upscale_backend=a.upscale_backend, face_backend=a.face_backend,
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("peloton").error("%s", exc)
        return 1
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
