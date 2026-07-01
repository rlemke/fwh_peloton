#!/usr/bin/env python3
"""batch-photos — run the peloton pipeline over a whole directory of photos.

Usage:
    batch_photos.py --in-dir ~/test_photos --out-dir ~/test_photos_output --require-bike
    batch_photos.py --in-dir photos/ --out-dir out/ --segment --face-backend codeformer

Processes every image in --in-dir, reusing the loaded models across photos (much
faster than a subprocess per photo). Writes per-rider outputs to --out-dir plus a
running <out>/manifest.json, and continues past per-photo errors. Logs progress on
stderr; prints a JSON summary on stdout at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import pipeline  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
log = logging.getLogger("peloton.batch")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the peloton pipeline over a directory of photos.")
    ap.add_argument("--in-dir", required=True, help="directory of input photos")
    ap.add_argument("--out-dir", required=True, help="directory for per-rider outputs + manifest.json")
    ap.add_argument("--limit", type=int, default=0, help="process at most N photos (0 = all)")
    # pipeline knobs (mirror process_photo)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--require-bike", action="store_true")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--segment", action="store_true")
    ap.add_argument("--cutout-bg", default="white")
    ap.add_argument("--sam-model", default="mobile_sam.pt")
    ap.add_argument("--no-face-restore", action="store_true")
    ap.add_argument("--fidelity", type=float, default=0.7)
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--upscale-backend", default="auto")
    ap.add_argument("--face-backend", default="auto")
    ap.add_argument("--use-mock", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    in_dir, out_dir = Path(a.in_dir).expanduser(), Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _EXTS)
    if a.limit:
        photos = photos[:a.limit]
    if not photos:
        log.error("no images found in %s", in_dir)
        return 1
    log.info("batch: %d photo(s) from %s → %s", len(photos), in_dir, out_dir)

    manifest: list[dict] = []
    n_ok = n_fail = n_riders = 0
    batch_t0 = time.time()
    for i, f in enumerate(photos, 1):
        t0 = time.time()
        try:
            s = pipeline.process_photo(
                f, out_dir, conf=a.conf, pad_frac=a.pad,
                require_bike=a.require_bike, scale=a.scale,
                segment=a.segment, cutout_bg=a.cutout_bg, sam_model=a.sam_model,
                restore_faces=not a.no_face_restore, fidelity=a.fidelity,
                use_mock=a.use_mock, detect_model=a.model,
                upscale_backend=a.upscale_backend, face_backend=a.face_backend,
            )
            dt = time.time() - t0
            n_ok += 1
            n_riders += s["n_riders"]
            manifest.append({"source": str(f), "n_riders": s["n_riders"],
                             "outputs": [r["output"] for r in s["riders"]],
                             "seconds": round(dt, 1)})
            log.info("[%d/%d] %s → %d rider(s) in %.1fs", i, len(photos), f.name,
                     s["n_riders"], dt)
        except Exception as exc:  # noqa: BLE001 - keep going across bad photos
            n_fail += 1
            manifest.append({"source": str(f), "error": str(exc)})
            log.error("[%d/%d] %s FAILED: %s", i, len(photos), f.name, exc)
        # write the manifest every photo so progress is inspectable mid-run
        (out_dir / "manifest.json").write_text(json.dumps(
            {"processed": i, "total": len(photos), "ok": n_ok, "failed": n_fail,
             "rider_portraits": n_riders, "photos": manifest}, indent=2))

    summary = {"total": len(photos), "ok": n_ok, "failed": n_fail,
               "rider_portraits": n_riders, "out_dir": str(out_dir),
               "elapsed_seconds": round(time.time() - batch_t0, 1)}
    log.info("DONE: %d photos (%d ok, %d failed) → %d rider portrait(s) in %.0fs",
             len(photos), n_ok, n_fail, n_riders, summary["elapsed_seconds"])
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
