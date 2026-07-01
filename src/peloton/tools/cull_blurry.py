#!/usr/bin/env python3
"""cull-blurry — move out-of-focus photos aside into an _unfocused/ directory.

Usage:
    cull_blurry.py --in-dir photos/                         # dry-run report: score distribution
    cull_blurry.py --in-dir photos/ --min-sharpness 40      # move soft photos → photos/_unfocused/
    cull_blurry.py --in-dir photos/ --min-sharpness 40 --metric rider --copy --dry-run

Scores each photo's focus (resolution-normalized variance of the Laplacian —
higher = sharper). ``--metric whole`` (default, fast, no models) scores the whole
frame; ``--metric rider`` detects the rider and scores just that region (better
for panning shots with an intentionally blurred background). With no
``--min-sharpness`` it just reports the distribution so you can pick a threshold;
with one, photos below it are moved (or ``--copy``'d) to ``--unfocused-dir``.
stdout: JSON. stderr: logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import detect, images, quality  # noqa: E402

_EXTS = ({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
         | images.RAW_EXTS)
log = logging.getLogger("peloton.cull")


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _score(img, metric: str, model: str, use_mock: bool) -> tuple[float, bool]:
    if metric == "rider":
        riders = detect.detect_riders(img, model=model, use_mock=use_mock, conf=0.25)
        if riders:
            w, h = images.size(img)
            region = img.crop(tuple(int(v) for v in riders[0].focus_box(0.0, w, h)))
            return quality.focus_score(region), True
    return quality.focus_score(img), False


def main() -> int:
    ap = argparse.ArgumentParser(description="Move out-of-focus photos to an _unfocused/ dir.")
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--unfocused-dir", default=None, help="default: <in-dir>/_unfocused")
    ap.add_argument("--min-sharpness", type=float, default=None,
                    help="focus score below this = unfocused; omit for a report-only dry-run")
    ap.add_argument("--metric", choices=["whole", "rider"], default="whole")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--dry-run", action="store_true", help="score + report, don't move")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-mock", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    in_dir = Path(a.in_dir).expanduser()
    photos = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _EXTS)
    if a.limit:
        photos = photos[:a.limit]
    if not photos:
        log.error("no images in %s", in_dir)
        return 1

    scored: list[dict] = []
    for i, p in enumerate(photos, 1):
        try:
            s, on_rider = _score(images.load_image(p), a.metric, a.model, a.use_mock)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", p.name, exc)
            continue
        scored.append({"path": str(p), "score": round(s, 1), "on_rider": on_rider})
        log.info("[%d/%d] %s: focus=%.1f", i, len(photos), p.name, s)

    vals = [d["score"] for d in scored]
    dist = {"min": round(min(vals), 1), "p10": round(_percentile(vals, 0.10), 1),
            "median": round(_percentile(vals, 0.5), 1),
            "p90": round(_percentile(vals, 0.9), 1), "max": round(max(vals), 1)}
    suggested = round(_percentile(vals, 0.5) * 0.4, 1)   # ~40% of median as a starting point
    log.info("focus distribution: %s | suggested --min-sharpness ~%.0f", dist, suggested)

    culled: list[dict] = []
    if a.min_sharpness is not None:
        unf = Path(a.unfocused_dir).expanduser() if a.unfocused_dir else in_dir / "_unfocused"
        unf.mkdir(parents=True, exist_ok=True)
        for d in scored:
            if d["score"] < a.min_sharpness:
                src = Path(d["path"])
                if not a.dry_run:
                    (shutil.copy2 if a.copy else shutil.move)(str(src), str(unf / src.name))
                culled.append(d)
        log.info("%s %d/%d photo(s) below %.1f → %s",
                 "would move" if a.dry_run else ("copied" if a.copy else "moved"),
                 len(culled), len(scored), a.min_sharpness, unf)

    summary = {"in_dir": str(in_dir), "scanned": len(scored), "metric": a.metric,
               "distribution": dist, "suggested_min_sharpness": suggested,
               "min_sharpness": a.min_sharpness, "dry_run": a.dry_run,
               "unfocused": len(culled),
               "culled": [c["path"] for c in culled]}
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
