#!/usr/bin/env python3
"""convert-photos — batch image format/resolution converter.

Convert between formats — **RAW (NEF/CR2/ARW/DNG/…), TIFF, JPEG, PNG, HEIC in → TIFF
or JPEG out** — at the original resolution or resized. Everything is decoded to a
16-bit working image, so TIFF output is lossless 16-bit and JPEG output is 8-bit.

Usage:
    # one file
    convert_photos.py --image D85_1960.NEF --out-dir out/                    # → 16-bit TIFF, full res
    convert_photos.py --image shot.NEF --out-dir out/ --format jpeg --quality 95
    convert_photos.py --image master.tif --out-dir out/ --format jpeg --resize 2048   # TIF → JPEG, long edge 2048

    # a directory (one level)
    convert_photos.py --in-dir raws/ --out-dir tifs/                          # RAW → TIFF
    convert_photos.py --in-dir tifs/ --out-dir jpgs/ --from tif --format jpeg --resize 50%

    # a whole tree, mirroring the input directory structure (resumable, adaptive-parallel)
    convert_photos.py --in-dir shoots/ --out-dir out/ --recursive --format jpeg --resize 3000 --resume

Options:
  --format tif|jpeg   output format (default tif = lossless 16-bit; jpeg = 8-bit)
  --quality N         JPEG quality 1-100 (default 95)
  --resize SPEC       resolution: ``N`` (long edge = N px) · ``WxH`` (fit within box) ·
                      ``50%`` / ``0.5`` (scale). Omit to keep the original resolution.
  --from SPEC         which inputs to convert: ``raw`` (default for --recursive), an
                      extension list like ``tif`` or ``jpg,jpeg``, or ``any``.
  --workers auto|N    parallel workers (auto = adaptive to free CPU, default; see below).

Parallelism (``--workers``): directory/recursive runs convert many files at once.
``auto`` sizes the pool to the *free* CPUs (cores − load average), ramps up while there
is headroom and backs off when the machine saturates — using as many cores as it can
without hurting overall performance (and yielding to co-located work like a runner
fleet). Pass an integer to pin the count, or ``1`` for serial.

stdout: JSON summary. stderr: logs.
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

log = logging.getLogger("peloton.convert")

_NONRAW_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


# ---- format + resolution helpers -------------------------------------------------

def _norm_fmt(fmt: str) -> str:
    return "tif" if fmt.lower() in ("tif", "tiff") else "jpeg"


def _ext(fmt: str) -> str:
    return "tif" if _norm_fmt(fmt) == "tif" else "jpg"


def parse_resize(spec: str | None):
    """``'N'`` → long edge N · ``'WxH'`` → fit within box · ``'50%'``/``'0.5'`` → scale.
    Returns a ``(kind, value)`` tuple or ``None``."""
    if not spec:
        return None
    s = spec.strip().lower()
    if s.endswith("%"):
        return ("scale", float(s[:-1]) / 100.0)
    if "x" in s:
        w, h = s.split("x")
        return ("box", (int(w), int(h)))
    if "." in s:
        return ("scale", float(s))
    return ("longedge", int(s))


def _resize_arr(arr, mode):
    """Resize a uint16 HxWx3 array per ``mode`` (from ``parse_resize``)."""
    import cv2  # noqa: PLC0415

    h, w = int(arr.shape[0]), int(arr.shape[1])
    kind, val = mode
    if kind == "scale":
        s = float(val)
    elif kind == "longedge":
        s = float(val) / max(w, h)
    else:  # box: fit within, preserve aspect
        s = min(val[0] / w, val[1] / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    if (nw, nh) == (w, h):
        return arr
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(arr, (nw, nh), interpolation=interp)


def _save(arr, dst: Path, fmt: str, quality: int) -> None:
    """Write a uint16 HxWx3 array as lossless 16-bit TIFF or 8-bit JPEG."""
    if _norm_fmt(fmt) == "tif":
        images.save_tiff16(arr, dst)
    else:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        a8 = np.clip(np.rint(arr.astype("float32") / 257.0), 0, 255).astype("uint8")
        images.save_image(Image.fromarray(a8, "RGB"), dst, quality=quality)


def parse_from(spec: str | None, *, default: set[str]) -> set[str]:
    """``None`` → default · ``'raw'`` → RAW_EXTS · ``'any'`` → RAW+images ·
    else a comma list of extensions (``'tif'``, ``'jpg,jpeg'``)."""
    if not spec:
        return default
    s = spec.strip().lower()
    if s == "raw":
        return set(images.RAW_EXTS)
    if s == "any":
        return set(images.RAW_EXTS) | _NONRAW_IMG
    return {"." + tok.strip().lstrip(".") for tok in s.split(",") if tok.strip()}


# ---- single-file convert ---------------------------------------------------------

def convert_one(src: Path, out_dir: Path, *, highlight_mode: str = "clip",
                out_format: str = "tif", resize=None, quality: int = 95) -> Path:
    """Convert one file → TIFF/JPEG (optionally resized). Returns the output path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = images.load_image16(src, highlight_mode=highlight_mode)
    if resize:
        arr = _resize_arr(arr, resize)
    dst = out_dir / f"{src.stem}.{_ext(out_format)}"
    _save(arr, dst, out_format, quality)
    log.info("%s → %s (%dx%d)", src.name, dst.name, arr.shape[1], arr.shape[0])
    return dst


# ---- adaptive-parallel engine ----------------------------------------------------

def _resolve_workers(workers: object) -> tuple[int, int, bool]:
    """(initial_target, hard_cap, adaptive). ``auto``/None/0 → adaptive, starting at
    the currently-free core count (cores − load average)."""
    ncpu = os.cpu_count() or 4
    hard_cap = max(1, ncpu)
    if workers in (None, 0, "0") or (isinstance(workers, str) and workers.lower() == "auto"):
        try:
            free = ncpu - os.getloadavg()[0]
        except (OSError, AttributeError):
            free = ncpu
        return max(1, min(hard_cap, int(round(free)))), hard_cap, True
    n = max(1, min(hard_cap, int(workers)))
    return n, n, False


def _convert_many(tasks: list[tuple[Path, Path]], *, highlight_mode: str = "clip",
                  out_format: str = "tif", resize=None, quality: int = 95,
                  resume: bool = False, workers: object = "auto",
                  manifest_path: Path | None = None) -> dict:
    """Convert ``(src, dst)`` pairs, adaptively parallel; running manifest optional."""
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
                arr = images.load_image16(src, highlight_mode=highlight_mode)
                if resize:
                    arr = _resize_arr(arr, resize)
                dst.parent.mkdir(parents=True, exist_ok=True)
                _save(arr, dst, out_format, quality)
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
    log.info("converting %d file(s) → %s: %d worker(s)%s (cap %d cores)", total,
             _ext(out_format).upper(), start, " adaptive→free-CPU" if adaptive else "", hard_cap)

    with ThreadPoolExecutor(max_workers=hard_cap) as ex:
        for src, dst in tasks:
            with cond:
                while inflight[0] >= target[0]:
                    cond.wait()
                inflight[0] += 1
            ex.submit(_do_one, src, dst)
    stop.set()
    _manifest(total)
    return {"total": total, "converted": st["ok"], "skipped": st["skip"],
            "failed": st["fail"], "workers_peak": st["peak"]}


# ---- directory / tree ------------------------------------------------------------

def convert_dir(in_dir: Path, out_dir: Path, *, highlight_mode: str = "clip",
                out_format: str = "tif", resize=None, quality: int = 95,
                resume: bool = False, limit: int = 0, workers: object = "auto",
                in_exts: set[str] | None = None) -> dict:
    """Convert every matching file in ``in_dir`` (one level) → TIFF/JPEG in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = in_exts if in_exts is not None else (set(images.RAW_EXTS) | _NONRAW_IMG)
    ext = _ext(out_format)
    srcs = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    tasks = [(f, out_dir / f"{f.stem}.{ext}") for f in srcs]
    s = _convert_many(tasks, highlight_mode=highlight_mode, out_format=out_format,
                      resize=resize, quality=quality, resume=resume, workers=workers)
    s.update(in_dir=str(in_dir), out_dir=str(out_dir))
    return s


def convert_tree(in_dir: Path, out_dir: Path, *, highlight_mode: str = "clip",
                 out_format: str = "tif", resize=None, quality: int = 95,
                 resume: bool = False, limit: int = 0, exts: set[str] | None = None,
                 workers: object = "auto") -> dict:
    """Recursively convert every matching file under ``in_dir`` → TIFF/JPEG under
    ``out_dir``, **mirroring the relative directory structure**. Default inputs are RAW
    only (override with ``exts``). Writes ``<out_dir>/_convert_manifest.json``; ``resume``
    skips existing outputs. Adaptively parallel (see ``--workers``)."""
    exts = exts if exts is not None else set(images.RAW_EXTS)
    ext = _ext(out_format)
    srcs = sorted(p for p in in_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(f, out_dir / f.relative_to(in_dir).parent / f"{f.stem}.{ext}") for f in srcs]
    log.info("recursive: %d file(s) under %s → %s", len(srcs), in_dir, out_dir)
    s = _convert_many(tasks, highlight_mode=highlight_mode, out_format=out_format,
                      resize=resize, quality=quality, resume=resume, workers=workers,
                      manifest_path=out_dir / "_convert_manifest.json")
    s.update(in_dir=str(in_dir), out_dir=str(out_dir))
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert images between RAW/TIFF/JPEG at any resolution.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="a single input file")
    g.add_argument("--in-dir", help="a directory of input files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recursive", action="store_true",
                    help="with --in-dir: recurse, mirroring the input tree into --out-dir")
    ap.add_argument("--format", choices=["tif", "tiff", "jpeg", "jpg"], default="tif",
                    help="output format (tif = lossless 16-bit; jpeg = 8-bit)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default 95)")
    ap.add_argument("--resize", default=None,
                    help="N (long edge) | WxH (fit box) | 50%%/0.5 (scale); omit = original")
    ap.add_argument("--from", dest="from_", default=None,
                    help="inputs to convert: raw (default for --recursive) | any | ext list (tif, jpg,jpeg)")
    ap.add_argument("--workers", default="auto",
                    help="parallel workers: 'auto' (adaptive, default) or an integer (1 = serial)")
    ap.add_argument("--highlight-mode", choices=["clip", "blend", "reconstruct"],
                    default="clip", help="RAW highlight handling (default: verbatim clip)")
    ap.add_argument("--resume", action="store_true", help="skip outputs that already exist")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(a.out_dir).expanduser()
    resize = parse_resize(a.resize)
    t0 = time.time()
    if a.image:
        p = convert_one(Path(a.image).expanduser(), out_dir, highlight_mode=a.highlight_mode,
                        out_format=a.format, resize=resize, quality=a.quality)
        summary = {"total": 1, "converted": 1, "skipped": 0, "failed": 0,
                   "output": str(p), "out_dir": str(out_dir)}
    else:
        in_dir = Path(a.in_dir).expanduser()
        if not in_dir.is_dir():
            log.error("not a directory: %s", in_dir)
            return 1
        common = dict(highlight_mode=a.highlight_mode, out_format=a.format, resize=resize,
                      quality=a.quality, resume=a.resume, limit=a.limit, workers=a.workers)
        if a.recursive:
            exts = parse_from(a.from_, default=set(images.RAW_EXTS))
            summary = convert_tree(in_dir, out_dir, exts=exts, **common)
        else:
            in_exts = parse_from(a.from_, default=set(images.RAW_EXTS) | _NONRAW_IMG)
            summary = convert_dir(in_dir, out_dir, in_exts=in_exts, **common)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    log.info("DONE: %d converted, %d skipped, %d failed → %s in %.0fs",
             summary["converted"], summary.get("skipped", 0), summary.get("failed", 0),
             out_dir, summary["elapsed_seconds"])
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
