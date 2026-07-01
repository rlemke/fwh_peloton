#!/usr/bin/env python3
"""enhance-image — upscale + face-restore a single image.

Usage:
    enhance_image.py --image rider.jpg --out rider_hi.jpg
    enhance_image.py --image rider.jpg --out rider_hi.jpg --scale 4 --no-face-restore

Standalone enhance step (the same code the pipeline runs per rider). Degrades to
a Lanczos upscale + passthrough when the ML backends aren't installed.
stdout: JSON {input, output, scale, upscale_backend, face_backend}. stderr: logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import enhance, images  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Upscale + face-restore one image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--no-face-restore", action="store_true")
    ap.add_argument("--fidelity", type=float, default=0.7)
    ap.add_argument("--upscale-backend", default="auto")
    ap.add_argument("--face-backend", default="auto")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        img = images.load_image(a.image)
        img, up_backend = enhance.upscale(img, scale=a.scale, backend=a.upscale_backend)
        face_backend = "skipped"
        if not a.no_face_restore:
            img, face_backend = enhance.restore_faces(
                img, fidelity=a.fidelity, backend=a.face_backend)
        images.save_image(img, a.out)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("peloton").error("%s", exc)
        return 1

    json.dump({"input": str(a.image), "output": str(a.out), "scale": a.scale,
               "output_size": list(images.size(img)),
               "upscale_backend": up_backend, "face_backend": face_backend},
              sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
