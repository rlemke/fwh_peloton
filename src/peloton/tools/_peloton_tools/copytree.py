"""Parallel recursive directory copy — mirror a source tree into a destination.

Preserves the relative directory structure, copies file metadata (``copy2``), and is
restart-safe: a file whose destination already exists with the same size is skipped.
Adaptively parallel — sizes the worker pool to the *free* CPUs and, because bulk copy
is I/O-bound, holds a fixed pool (the OS/disk is the ceiling, not CPU); pass an explicit
worker count to override. Continues past per-file errors; writes an optional running
manifest for progress.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger("peloton.copytree")


def _resolve_workers(workers: object) -> int:
    ncpu = os.cpu_count() or 4
    if workers in (None, 0, "0") or (isinstance(workers, str) and str(workers).lower() == "auto"):
        try:
            free = ncpu - os.getloadavg()[0]
        except (OSError, AttributeError):
            free = ncpu
        return max(2, min(ncpu, int(round(free))))       # I/O-bound: a few workers hide latency
    return max(1, min(ncpu, int(workers)))


def copy_tree(src: str | Path, dst: str | Path, *, workers: object = "auto",
              manifest_path: str | Path | None = None) -> dict:
    """Recursively copy ``src`` → ``dst`` (mirroring structure), in parallel. Returns
    a summary dict. ``resume`` is implicit — same-size destinations are skipped."""
    src, dst = Path(src).expanduser(), Path(dst).expanduser()
    n = _resolve_workers(workers)
    files = [p for p in src.rglob("*") if p.is_file()]
    total = len(files)
    dst.mkdir(parents=True, exist_ok=True)
    for d in {f.parent for f in files}:                  # pre-create the dir tree once
        (dst / d.relative_to(src)).mkdir(parents=True, exist_ok=True)

    st = {"ok": 0, "skip": 0, "fail": 0, "done": 0, "bytes": 0}
    lock = threading.Lock()
    t0 = time.time()
    mp = Path(manifest_path) if manifest_path else None

    def _write(i: int) -> None:
        if mp is None:
            return
        el = time.time() - t0
        mp.write_text(json.dumps(
            {"processed": i, "total": total, "copied": st["ok"], "skipped": st["skip"],
             "failed": st["fail"], "gb": round(st["bytes"] / 1e9, 1),
             "files_per_s": round(st["ok"] / el, 2) if el else 0.0,
             "workers": n, "src": str(src), "dst": str(dst)}, indent=2))

    def _one(f: Path) -> None:
        rel = f.relative_to(src)
        d = dst / rel
        err = None
        skipped = False
        nbytes = 0
        try:
            size = f.stat().st_size
            if d.exists() and d.stat().st_size == size:
                skipped = True
            else:
                shutil.copy2(f, d)
                nbytes = size
        except Exception as exc:  # noqa: BLE001 — keep going across bad files
            err = str(exc)
        with lock:
            st["done"] += 1
            i = st["done"]
            if skipped:
                st["skip"] += 1
            elif err:
                st["fail"] += 1
                log.error("[%d/%d] %s FAILED: %s", i, total, rel, err)
            else:
                st["ok"] += 1
                st["bytes"] += nbytes
            if i % 50 == 0 or i == total:
                _write(i)

    log.info("copying %d file(s): %s → %s (%d workers)", total, src, dst, n)
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_one, files))
    _write(total)
    return {"total": total, "copied": st["ok"], "skipped": st["skip"], "failed": st["fail"],
            "gb": round(st["bytes"] / 1e9, 1), "src": str(src), "dst": str(dst),
            "workers": n}
