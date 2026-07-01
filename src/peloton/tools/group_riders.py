#!/usr/bin/env python3
"""group-riders — cluster per-rider crops by face into one folder per person.

Usage:
    group_riders.py --in-dir ~/test_photos_output --out-dir ~/test_photos_output/by_rider
    group_riders.py --in-dir crops/ --out-dir grouped/ --threshold 0.4 --copy

Point it at the per-rider crops produced by process/batch. It embeds each crop's
face (InsightFace/ArcFace), clusters same-rider crops together, and writes one
``rider_NNN/`` folder per person with the members ranked best-shot first (by the
model-free quality score). Crops with no detectable face go to ``_no_face/``.
stdout: JSON summary + <out>/groups.json. stderr: logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _peloton_tools import images, quality, recognize  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
log = logging.getLogger("peloton.group")


def _link(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Group per-rider crops by face recognition.")
    ap.add_argument("--in-dir", required=True, help="directory of per-rider crops (top level)")
    ap.add_argument("--out-dir", required=True, help="output: one rider_NNN/ folder per person")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min cosine similarity to join a group (higher = stricter)")
    ap.add_argument("--model", default="buffalo_l", help="InsightFace model")
    ap.add_argument("--min-size", type=int, default=1, help="drop groups smaller than this")
    ap.add_argument("--copy", action="store_true", help="copy files (default: symlink)")
    ap.add_argument("--use-mock", action="store_true", help="offline deterministic embeddings")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level.upper(), stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")

    in_dir, out_dir = Path(a.in_dir).expanduser(), Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _EXTS)
    if not crops:
        log.error("no crops in %s", in_dir)
        return 1

    items: list[dict] = []
    embs: list = []
    no_face: list[Path] = []
    for p in crops:
        try:
            img = images.load_image(p)
            faces = recognize.face_embeddings(img, model=a.model, use_mock=a.use_mock)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", p.name, exc)
            continue
        if not faces:
            no_face.append(p)
            continue
        emb, box = faces[0]
        q = quality.score(img, face_box=box)
        embs.append(emb)
        items.append({"path": p, "score": q["score"], "sharpness": q["sharpness"]})
    log.info("embedded %d crop(s); %d had no detectable face", len(items), len(no_face))

    labels = recognize.cluster(embs, threshold=a.threshold) if embs else []
    groups: dict[int, list[dict]] = {}
    for lab, it in zip(labels, items):
        groups.setdefault(lab, []).append(it)

    manifest: dict[str, dict] = {}
    rid = 0
    for _lab, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < a.min_size:
            continue
        members.sort(key=lambda m: -m["score"])  # best shot first
        rid += 1
        gdir = out_dir / f"rider_{rid:03d}"
        gdir.mkdir(exist_ok=True)
        for rank, m in enumerate(members):
            tag = "best_" if rank == 0 else ""
            _link(m["path"], gdir / f"{rank:02d}_{tag}{m['path'].name}", a.copy)
        manifest[f"rider_{rid:03d}"] = {
            "count": len(members),
            "best": str(members[0]["path"]),
            "members": [{"path": str(m["path"]), "score": m["score"]} for m in members],
        }
    if no_face:
        nf = out_dir / "_no_face"
        nf.mkdir(exist_ok=True)
        for p in no_face:
            _link(p, nf / p.name, a.copy)

    (out_dir / "groups.json").write_text(json.dumps(manifest, indent=2))
    summary = {"in_dir": str(in_dir), "out_dir": str(out_dir),
               "crops": len(crops), "riders": rid,
               "no_face": len(no_face),
               "biggest_group": max((g["count"] for g in manifest.values()), default=0)}
    log.info("DONE: %d crop(s) → %d rider group(s) (%d no-face)",
             len(crops), rid, len(no_face))
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
