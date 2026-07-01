#!/usr/bin/env python3
"""detect-riders — detect each cyclist in a photo, emit their boxes as JSON.

Usage:
    detect_riders.py --image group.jpg
    detect_riders.py --image group.jpg --require-bike --conf 0.3
    detect_riders.py --image group.jpg --use-mock        # offline, no models

stdout: JSON {"source", "size", "n_riders", "riders": [{index, score, person_box,
bike_box, has_bike}, ...]}. stderr: logs.
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
    ap = argparse.ArgumentParser(description="Detect riders (person + bicycle) in a photo.")
    ap.add_argument("--image", required=True, help="input photo path")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--require-bike", action="store_true", help="only riders paired with a bicycle")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--use-mock", action="store_true", help="offline deterministic detector")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        img = images.load_image(a.image)
        riders = detect.detect_riders(
            img, conf=a.conf, require_bike=a.require_bike,
            model=a.model, use_mock=a.use_mock)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("peloton").error("%s", exc)
        return 1

    out = {
        "source": str(a.image),
        "size": list(images.size(img)),
        "n_riders": len(riders),
        "riders": [r.to_dict() for r in riders],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
