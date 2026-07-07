#!/usr/bin/env python3
"""copy-tree — parallel recursive directory copy (mirror SRC into DST).

Usage:
    copy_tree.py --src photos_jpg/ --dst /Volumes/backup/photos_jpg/
    copy_tree.py --src a/ --dst b/ --workers 8

Preserves the relative directory structure, copies file metadata, and is restart-safe
(skips files already present at the destination with the same size). Multi-threaded —
``--workers auto`` (default) sizes to free CPUs; bulk copy is I/O-bound, so the disk is
usually the ceiling. stdout: JSON summary; stderr: logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import copytree  # noqa: E402

log = logging.getLogger("peloton.copytree.cli")


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel recursive directory copy.")
    ap.add_argument("--src", required=True, help="source directory")
    ap.add_argument("--dst", required=True, help="destination directory (structure mirrored)")
    ap.add_argument("--workers", default="auto",
                    help="parallel workers: 'auto' (default) or an integer")
    ap.add_argument("--manifest", default=None,
                    help="path for a running progress manifest (JSON); default <dst>/_copy_manifest.json")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    src = Path(a.src).expanduser()
    if not src.is_dir():
        log.error("not a directory: %s", src)
        return 1
    manifest = a.manifest or (Path(a.dst).expanduser() / "_copy_manifest.json")
    t0 = time.time()
    s = copytree.copy_tree(src, a.dst, workers=a.workers, manifest_path=manifest)
    s["elapsed_seconds"] = round(time.time() - t0, 1)
    log.info("DONE: %d copied, %d skipped, %d failed (%.1f GB) → %s in %.0fs",
             s["copied"], s["skipped"], s["failed"], s["gb"], s["dst"], s["elapsed_seconds"])
    json.dump(s, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
