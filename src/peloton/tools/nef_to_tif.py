#!/usr/bin/env python3
"""nef-to-tif — plain RAW → 16-bit TIFF convert at the ORIGINAL full resolution.

Usage:
    nef_to_tif.py --image D85_1960.NEF --out-dir out/
    nef_to_tif.py --in-dir raws/ --out-dir tifs/ --resume
    nef_to_tif.py --in-dir shoots/ --out-dir tifs/ --recursive --resume   # mirror the tree

A verbatim decode: camera white balance, full sensor resolution, no crop / detect /
enhancement — just the RAW developed to a lossless 16-bit TIFF the same pixel size as
the sensor (a 45 MP D850 NEF → an 8288x5520 16-bit TIFF). Also accepts .cr2/.cr3/.arw/
.dng/… (any supported RAW) and non-RAW images (lifted to 16-bit). stdout: JSON; stderr:
logs.

**Parallelism** (``--workers``): directory/recursive runs convert many files at once.
``auto`` (default) sizes the pool to the *free* CPUs (cores − load average), ramps up
while there is headroom and backs off when the machine saturates — so it uses as many
cores as it can without hurting overall performance (and yields to other work, e.g. a
co-located runner fleet). Pass an integer to pin the worker count, or ``1`` for serial.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import images  # noqa: E402

log = logging.getLogger("peloton.nef2tif")


def convert_one(src: Path, out_dir: Path, *, highlight_mode: str = "clip") -> Path:
    """Decode one RAW/image at native resolution → 16-bit TIFF. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = images.load_image16(src, highlight_mode=highlight_mode)   # uint16 HxWx3, full res
    out_path = out_dir / f"{src.stem}.tif"
    images.save_tiff16(arr, out_path)
    log.info("%s (%dx%d) → %s", src.name, arr.shape[1], arr.shape[0], out_path.name)
    return out_path


def _resolve_workers(workers: object) -> tuple[int, int, bool]:
    """(initial_target, hard_cap, adaptive) from the ``workers`` arg.
    ``auto``/None/0 → adaptive, starting at the currently-free core count."""
    ncpu = os.cpu_count() or 4
    hard_cap = max(1, ncpu)
    if workers in (None, 0, "0") or (isinstance(workers, str) and workers.lower() == "auto"):
        try:
            free = ncpu - os.getloadavg()[0]
        except (OSError, AttributeError):     # getloadavg unavailable
            free = ncpu
        return max(1, min(hard_cap, int(round(free)))), hard_cap, True
    return max(1, min(hard_cap, int(workers))), max(1, min(hard_cap, int(workers))), False


def _convert_many(tasks: list[tuple[Path, Path]], *, highlight_mode: str = "clip",
                  resume: bool = False, workers: object = "auto",
                  manifest_path: Path | None = None) -> dict:
    """Convert a list of ``(src, dst)`` pairs, adaptively parallel. Writes a running
    manifest to ``manifest_path`` if given. Continues past per-file errors."""
    total = len(tasks)
    start, hard_cap, adaptive = _resolve_workers(workers)
    ncpu = os.cpu_count() or 4

    st = {"ok": 0, "skip": 0, "fail": 0, "done": 0, "peak": start}
    times: deque[float] = deque(maxlen=100)
    target = [start]
    inflight = [0]
    cond = threading.Condition()

    def _manifest(i: int) -> None:
        if manifest_path is None:
            return
        avg = (sum(times) / len(times)) if times else 0.0
        manifest_path.write_text(json.dumps(
            {"processed": i, "total": total, "ok": st["ok"], "skipped": st["skip"],
             "failed": st["fail"], "avg_s": round(avg, 1), "workers": target[0]}, indent=2))

    def _do_one(src: Path, dst: Path) -> None:
        t0 = time.time()
        skipped = resume and dst.exists()
        err = None
        if not skipped:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                arr = images.load_image16(src, highlight_mode=highlight_mode)
                images.save_tiff16(arr, dst)
            except Exception as exc:  # noqa: BLE001 — keep going across bad files
                err = str(exc)
        dt = time.time() - t0
        with cond:
            st["done"] += 1
            i = st["done"]
            if skipped:
                st["skip"] += 1
            elif err:
                st["fail"] += 1
                log.error("[%d/%d] %s FAILED: %s", i, total, src.name, err)
            else:
                st["ok"] += 1
                times.append(dt)
                log.info("[%d/%d] %s → %s", i, total, src.name, dst.name)
            inflight[0] -= 1
            if i % 10 == 0 or i == total:
                _manifest(i)
            cond.notify_all()

    stop = threading.Event()

    def _controller() -> None:
        # Keep total system load near full but not oversubscribed: ramp workers up
        # while there's CPU headroom, back off when load exceeds the core count.
        while not stop.wait(8.0):
            try:
                la = os.getloadavg()[0]
            except (OSError, AttributeError):
                la = 0.0
            with cond:
                if la > ncpu * 1.05 and target[0] > 1:
                    target[0] -= 1
                elif la < ncpu * 0.90 and target[0] < hard_cap:
                    target[0] += 1
                st["peak"] = max(st["peak"], target[0])
                cond.notify_all()

    if adaptive:
        threading.Thread(target=_controller, daemon=True).start()
    log.info("converting %d file(s): %d worker(s)%s (cap %d cores)", total, start,
             " adaptive→free-CPU" if adaptive else "", hard_cap)

    with ThreadPoolExecutor(max_workers=hard_cap) as ex:
        for src, dst in tasks:
            with cond:
                while inflight[0] >= target[0]:      # throttle to the current target
                    cond.wait()
                inflight[0] += 1
            ex.submit(_do_one, src, dst)
        # context exit waits for every in-flight conversion
    stop.set()
    _manifest(total)
    return {"total": total, "converted": st["ok"], "skipped": st["skip"],
            "failed": st["fail"], "workers_peak": st["peak"]}


def convert_dir(in_dir: Path, out_dir: Path, *, highlight_mode: str = "clip",
                resume: bool = False, limit: int = 0, workers: object = "auto") -> dict:
    """Convert every RAW/image in ``in_dir`` → 16-bit TIFF in ``out_dir`` (one level)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = images.RAW_EXTS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
    srcs = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    tasks = [(f, out_dir / f"{f.stem}.tif") for f in srcs]
    s = _convert_many(tasks, highlight_mode=highlight_mode, resume=resume, workers=workers)
    s.update(in_dir=str(in_dir), out_dir=str(out_dir))
    return s


def convert_tree(in_dir: Path, out_dir: Path, *, highlight_mode: str = "clip",
                 resume: bool = False, limit: int = 0, exts: set[str] | None = None,
                 workers: object = "auto") -> dict:
    """Recursively convert every RAW under ``in_dir`` → 16-bit TIFF under ``out_dir``,
    **mirroring the relative directory structure** (``in_dir/a/b/x.NEF`` →
    ``out_dir/a/b/x.tif``). RAW files only by default (sidecars/JPEGs ignored). Writes a
    running ``<out_dir>/_nef2tif_manifest.json``; ``resume`` skips existing outputs, so
    an interrupted run continues. Adaptively parallel (see ``--workers``). ``exts``
    overrides which suffixes are converted (testing)."""
    exts = exts or images.RAW_EXTS
    srcs = sorted(p for p in in_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(f, out_dir / f.relative_to(in_dir).parent / f"{f.stem}.tif") for f in srcs]
    log.info("recursive: %d RAW file(s) under %s → %s", len(srcs), in_dir, out_dir)
    s = _convert_many(tasks, highlight_mode=highlight_mode, resume=resume, workers=workers,
                      manifest_path=out_dir / "_nef2tif_manifest.json")
    s.update(in_dir=str(in_dir), out_dir=str(out_dir))
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert RAW → 16-bit TIFF at original resolution.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="a single RAW/image file")
    g.add_argument("--in-dir", help="a directory of RAW/image files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recursive", action="store_true",
                    help="with --in-dir: recurse, mirroring the input directory tree into --out-dir")
    ap.add_argument("--workers", default="auto",
                    help="parallel workers: 'auto' (adaptive to free CPU, default) or an integer (1 = serial)")
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
                       resume=a.resume, limit=a.limit, workers=a.workers)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    log.info("DONE: %d converted, %d skipped, %d failed → %s in %.0fs",
             summary["converted"], summary.get("skipped", 0), summary.get("failed", 0),
             out_dir, summary["elapsed_seconds"])
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
