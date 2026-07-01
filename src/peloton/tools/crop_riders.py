#!/usr/bin/env python3
"""crop-riders — detect riders and write one cropped image per rider (no enhance).

Usage:
    crop_riders.py --image group.jpg --out-dir crops/
    crop_riders.py --image group.jpg --out-dir crops/ --pad 0.2 --use-mock

The detect → crop half of the pipeline, for when you just want the cutouts.
stdout: JSON of the crops written. stderr: logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import detect, images  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Crop each detected rider to its own image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.15, help="crop padding fraction")
    ap.add_argument("--require-bike", action="store_true")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--use-mock", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        img = images.load_image(a.image)
        w, h = images.size(img)
        riders = detect.detect_riders(img, conf=a.conf, require_bike=a.require_bike,
                                      model=a.model, use_mock=a.use_mock)
        out_dir = Path(a.out_dir)
        stem = Path(a.image).stem
        crops = []
        for r in riders:
            box = r.focus_box(a.pad, w, h)
            crop = img.crop(tuple(int(v) for v in box))
            p = out_dir / f"{stem}_rider{r.index:02d}.jpg"
            images.save_image(crop, p)
            crops.append({**r.to_dict(), "focus_box": [int(v) for v in box],
                          "output": str(p), "output_size": list(images.size(crop))})
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("peloton").error("%s", exc)
        return 1

    json.dump({"source": str(a.image), "n_riders": len(crops), "crops": crops},
              sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
