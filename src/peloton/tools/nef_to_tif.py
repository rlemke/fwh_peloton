#!/usr/bin/env python3
"""nef-to-tif — plain RAW → 16-bit TIFF convert at the ORIGINAL full resolution.

Usage:
    nef_to_tif.py --image D85_1960.NEF --out-dir out/
    nef_to_tif.py --in-dir raws/ --out-dir tifs/ --resume

A verbatim decode: camera white balance, full sensor resolution, no crop / detect /
enhancement — just the RAW developed to a lossless 16-bit TIFF the same pixel size as
the sensor (e.g. a 45 MP D850 NEF → an 8288x5520 16-bit TIFF). Also accepts .cr2/.cr3/
.arw/.dng/… (any supported RAW) and non-RAW images (lifted to 16-bit). stdout: JSON;
stderr: logs.
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

log = logging.getLogger("peloton.nef2tif")


def convert_one(src: Path, out_dir: Path, *, highlight_mode: str = "clip") -> Path:
    """Decode one RAW/image at native resolution → 16-bit TIFF. Returns the path."""
    arr = images.load_image16(src, highlight_mode=highlight_mode)   # uint16 HxWx3, full res
    out_path = out_dir / f"{src.stem}.tif"
    images.save_tiff16(arr, out_path)
    log.info("%s (%dx%d) → %s", src.name, arr.shape[1], arr.shape[0], out_path.name)
    return out_path


def convert_dir(in_dir: Path, out_dir: Path, *, highlight_mode: str = "clip",
                resume: bool = False, limit: int = 0) -> dict:
    """Convert every RAW/image in ``in_dir`` → 16-bit TIFF in ``out_dir`` (one level).
    Continues past per-file errors; ``resume`` skips outputs that already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = images.RAW_EXTS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
    srcs = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    n_ok = n_skip = n_fail = 0
    outputs: list[str] = []
    for i, f in enumerate(srcs, 1):
        dst = out_dir / f"{f.stem}.tif"
        if resume and dst.exists():
            n_skip += 1
            continue
        try:
            convert_one(f, out_dir, highlight_mode=highlight_mode)
            n_ok += 1
            outputs.append(str(dst))
        except Exception as exc:  # noqa: BLE001 — keep going across bad files
            n_fail += 1
            log.error("[%d/%d] %s FAILED: %s", i, len(srcs), f.name, exc)
    return {"total": len(srcs), "converted": n_ok, "skipped": n_skip, "failed": n_fail,
            "in_dir": str(in_dir), "out_dir": str(out_dir), "outputs": outputs}


def convert_tree(in_dir: Path, out_dir: Path, *, highlight_mode: str = "clip",
                 resume: bool = False, limit: int = 0, exts: set[str] | None = None) -> dict:
    """Recursively convert every RAW under ``in_dir`` → 16-bit TIFF under ``out_dir``,
    **mirroring the relative directory structure** (``in_dir/a/b/x.NEF`` →
    ``out_dir/a/b/x.tif``). RAW files only by default (sidecars/JPEGs ignored). Writes
    a running ``<out_dir>/_nef2tif_manifest.json`` for progress; ``resume`` skips
    outputs that already exist, so an interrupted run continues. Continues past
    per-file errors. ``exts`` overrides which suffixes are converted (testing)."""
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    exts = exts or images.RAW_EXTS
    srcs = sorted(p for p in in_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    total = len(srcs)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "_nef2tif_manifest.json"
    n_ok = n_skip = n_fail = 0
    times: list[float] = []
    log.info("recursive: %d RAW file(s) under %s → %s", total, in_dir, out_dir)
    for i, f in enumerate(srcs, 1):
        dst_dir = out_dir / f.relative_to(in_dir).parent
        dst = dst_dir / f"{f.stem}.tif"
        if resume and dst.exists():
            n_skip += 1
        else:
            t0 = time.time()
            try:
                dst_dir.mkdir(parents=True, exist_ok=True)
                arr = images.load_image16(f, highlight_mode=highlight_mode)
                images.save_tiff16(arr, dst)
                n_ok += 1
                times.append(time.time() - t0)
                log.info("[%d/%d] %s (%dx%d) → %s", i, total, f.name,
                         arr.shape[1], arr.shape[0], dst)
            except Exception as exc:  # noqa: BLE001 — keep going across bad files
                n_fail += 1
                log.error("[%d/%d] %s FAILED: %s", i, total, f.name, exc)
        if i % 5 == 0 or i == total:
            avg = sum(times[-50:]) / len(times[-50:]) if times else 0.0
            manifest.write_text(json.dumps(
                {"processed": i, "total": total, "ok": n_ok, "skipped": n_skip,
                 "failed": n_fail, "avg_s": round(avg, 1),
                 "in_dir": str(in_dir), "out_dir": str(out_dir)}, indent=2))
    return {"total": total, "converted": n_ok, "skipped": n_skip, "failed": n_fail,
            "in_dir": str(in_dir), "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert RAW → 16-bit TIFF at original resolution.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="a single RAW/image file")
    g.add_argument("--in-dir", help="a directory of RAW/image files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recursive", action="store_true",
                    help="with --in-dir: recurse, mirroring the input directory tree into --out-dir")
    ap.add_argument("--highlight-mode", choices=["clip", "blend", "reconstruct"],
                    default="clip", help="RAW highlight handling (default: verbatim clip)")
    ap.add_argument("--resume", action="store_true", help="skip outputs that already exist")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(a.out_dir).expanduser()
    t0 = time.time()
    if a.image:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = convert_one(Path(a.image).expanduser(), out_dir, highlight_mode=a.highlight_mode)
        summary = {"total": 1, "converted": 1, "skipped": 0, "failed": 0,
                   "output": str(p), "out_dir": str(out_dir)}
    else:
        in_dir = Path(a.in_dir).expanduser()
        if not in_dir.is_dir():
            log.error("not a directory: %s", in_dir)
            return 1
        conv = convert_tree if a.recursive else convert_dir
        summary = conv(in_dir, out_dir, highlight_mode=a.highlight_mode,
                       resume=a.resume, limit=a.limit)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    log.info("DONE: %d converted, %d skipped, %d failed → %s in %.0fs",
             summary["converted"], summary.get("skipped", 0), summary.get("failed", 0),
             out_dir, summary["elapsed_seconds"])
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
