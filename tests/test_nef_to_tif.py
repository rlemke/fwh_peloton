"""RAW→TIFF verbatim converter (offline: exercised on non-RAW inputs, no rawpy)."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

import nef_to_tif


@pytest.fixture()
def img_dir(tmp_path):
    from PIL import Image
    d = tmp_path / "in"
    d.mkdir()
    rng = np.random.default_rng(4)
    for name in ("a", "b"):
        Image.fromarray(rng.integers(0, 256, (150, 220, 3)).astype("uint8")).save(d / f"{name}.jpg")
    return d


def test_convert_one_is_16bit_same_dims(img_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    p = nef_to_tif.convert_one(img_dir / "a.jpg", out)
    assert p.suffix == ".tif" and p.is_file()
    a = tifffile.imread(p)
    assert a.dtype == np.uint16 and a.shape == (150, 220, 3)   # original resolution preserved


def test_convert_dir_all(img_dir, tmp_path):
    out = tmp_path / "out"
    s = nef_to_tif.convert_dir(img_dir, out)
    assert s["total"] == 2 and s["converted"] == 2 and s["failed"] == 0
    assert len(list(out.glob("*.tif"))) == 2


def test_convert_dir_resume_skips(img_dir, tmp_path):
    out = tmp_path / "out"
    nef_to_tif.convert_dir(img_dir, out)
    again = nef_to_tif.convert_dir(img_dir, out, resume=True)
    assert again["converted"] == 0 and again["skipped"] == 2


def test_convert_tree_mirrors_structure(tmp_path):
    from PIL import Image
    src = tmp_path / "in"
    (src / "eventA" / "D850").mkdir(parents=True)
    (src / "eventB").mkdir(parents=True)
    rng = np.random.default_rng(6)
    for rel in ("eventA/D850/a.jpg", "eventA/D850/b.jpg", "eventB/c.jpg"):
        Image.fromarray(rng.integers(0, 256, (80, 100, 3)).astype("uint8")).save(src / rel)
    out = tmp_path / "out"
    # exts override so the test can use JPEGs (no rawpy needed)
    s = nef_to_tif.convert_tree(src, out, exts={".jpg"})
    assert s["total"] == 3 and s["converted"] == 3 and s["failed"] == 0
    # relative directory structure is preserved, .NEF/.jpg → .tif
    assert (out / "eventA" / "D850" / "a.tif").is_file()
    assert (out / "eventA" / "D850" / "b.tif").is_file()
    assert (out / "eventB" / "c.tif").is_file()
    assert (out / "_nef2tif_manifest.json").is_file()
    a = tifffile.imread(out / "eventB" / "c.tif")
    assert a.dtype == np.uint16 and a.shape == (80, 100, 3)


def test_resolve_workers():
    start, cap, adaptive = nef_to_tif._resolve_workers("auto")
    assert 1 <= start <= cap and adaptive is True     # auto → free-CPU sized, adaptive
    s2, c2, ad2 = nef_to_tif._resolve_workers(3)
    assert s2 == 3 and c2 == 3 and ad2 is False        # pinned worker count
    s1, _, _ = nef_to_tif._resolve_workers(1)
    assert s1 == 1                                      # serial


def test_convert_tree_fixed_workers(tmp_path):
    from PIL import Image
    src = tmp_path / "in" / "d"
    src.mkdir(parents=True)
    rng = np.random.default_rng(8)
    for n in ("a", "b", "c", "e"):
        Image.fromarray(rng.integers(0, 256, (60, 90, 3)).astype("uint8")).save(src.parent / f"{n}.jpg")
    out = tmp_path / "out"
    s = nef_to_tif.convert_tree(tmp_path / "in", out, exts={".jpg"}, workers=2)
    assert s["total"] == 4 and s["converted"] == 4 and s["failed"] == 0
    assert len(list(out.glob("*.tif"))) == 4
