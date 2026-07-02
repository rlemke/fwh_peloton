#!/usr/bin/env python3
"""tiffs-to-jpegs — derive shareable JPEGs from a directory of 16-bit TIFF masters.

Usage:
    tiffs_to_jpegs.py --in-dir tiff_out/ --out-dir jpeg_out/
    tiffs_to_jpegs.py --in-dir tiff_out/ --out-dir jpeg_out/ --quality 95 --overwrite

A separate, non-destructive step for the ``--out-format tiff`` pipeline output: the
lossless 16-bit ``.tif`` files stay as the archival masters, and this writes an
8-bit ``.jpg`` copy of each into a *different* directory (same filename stem). The
16-bit source is downconverted 65535→255; no re-detection, no re-enhancement — a
pure format conversion, so it's fast and idempotent.

By default outputs that already exist are skipped (re-runnable); ``--overwrite``
forces them. stdout: JSON summary. stderr: logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import images  # noqa: E402

_TIFF_EXTS = {".tif", ".tiff"}
log = logging.getLogger("peloton.tiff2jpg")


def convert_dir(in_dir: Path, out_dir: Path, *, quality: int = 100,
                overwrite: bool = False, limit: int = 0) -> dict:
    """Convert every TIFF in ``in_dir`` to a JPEG in ``out_dir`` (same stem).
    Returns a summary dict; continues past per-file errors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tiffs = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _TIFF_EXTS)
    if limit:
        tiffs = tiffs[:limit]
    n_ok = n_skip = n_fail = 0
    outputs: list[str] = []
    for i, f in enumerate(tiffs, 1):
        dst = out_dir / f"{f.stem}.jpg"
        if dst.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            img = images.tiff_to_image8(f)
            images.save_image(img, dst, quality=quality)
            n_ok += 1
            outputs.append(str(dst))
            log.info("[%d/%d] %s → %s (q%d)", i, len(tiffs), f.name, dst.name, quality)
        except Exception as exc:  # noqa: BLE001 - keep going across bad files
            n_fail += 1
            log.error("[%d/%d] %s FAILED: %s", i, len(tiffs), f.name, exc)
    return {"total": len(tiffs), "converted": n_ok, "skipped": n_skip,
            "failed": n_fail, "in_dir": str(in_dir), "out_dir": str(out_dir),
            "outputs": outputs}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive 8-bit JPEGs from a directory of 16-bit TIFF masters.")
    ap.add_argument("--in-dir", required=True, help="directory of .tif/.tiff masters")
    ap.add_argument("--out-dir", required=True, help="directory for the derived .jpg copies")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality (default 100)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-convert outputs that already exist (default: skip them)")
    ap.add_argument("--limit", type=int, default=0, help="convert at most N files (0 = all)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    in_dir, out_dir = Path(a.in_dir).expanduser(), Path(a.out_dir).expanduser()
    if not in_dir.is_dir():
        log.error("not a directory: %s", in_dir)
        return 1

    t0 = time.time()
    summary = convert_dir(in_dir, out_dir, quality=a.quality,
                          overwrite=a.overwrite, limit=a.limit)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    if summary["total"] == 0:
        log.error("no .tif/.tiff files found in %s", in_dir)
    log.info("DONE: %d converted, %d skipped, %d failed → %s in %.0fs",
             summary["converted"], summary["skipped"], summary["failed"],
             out_dir, summary["elapsed_seconds"])
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
