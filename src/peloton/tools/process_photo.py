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
from _peloton_tools import pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a group cycling photo into per-rider enhanced portraits.")
    ap.add_argument("--image", required=True, help="input photo path")
    ap.add_argument("--out-dir", required=True, help="directory for per-rider outputs")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--pad", type=float, default=0.15, help="crop padding fraction (0.15 = 15%%)")
    ap.add_argument("--require-bike", action="store_true", help="drop persons with no paired bicycle")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor")
    ap.add_argument("--no-face-restore", action="store_true", help="skip face restoration")
    ap.add_argument("--fidelity", type=float, default=0.7, help="face-restore identity fidelity (0..1)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--upscale-backend", default="auto", help="auto|realesrgan-ncnn|realesrgan|lanczos")
    ap.add_argument("--face-backend", default="auto", help="auto|gfpgan|none")
    ap.add_argument("--use-mock", action="store_true", help="offline deterministic detector (no models)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        summary = pipeline.process_photo(
            a.image, a.out_dir, conf=a.conf, pad_frac=a.pad,
            require_bike=a.require_bike, scale=a.scale,
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
