"""RAW→TIFF verbatim converter (offline: exercised on non-RAW inputs, no rawpy)."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

import convert_photos as nef_to_tif   # general converter (nef_to_tif is a back-compat shim)


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
    assert (out / "_convert_manifest.json").is_file()
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


def test_parse_resize_forms():
    assert nef_to_tif.parse_resize(None) is None
    assert nef_to_tif.parse_resize("2048") == ("longedge", 2048)
    assert nef_to_tif.parse_resize("800x600") == ("box", (800, 600))
    assert nef_to_tif.parse_resize("50%") == ("scale", 0.5)
    assert nef_to_tif.parse_resize("0.25") == ("scale", 0.25)


def test_convert_one_to_jpeg(tmp_path):
    from PIL import Image
    src = tmp_path / "a.jpg"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (120, 200, 3)).astype("uint8")).save(src)
    out = tmp_path / "out"
    p = nef_to_tif.convert_one(src, out, out_format="jpeg", quality=90)
    assert p.suffix == ".jpg" and Image.open(p).mode == "RGB" and Image.open(p).size == (200, 120)


def test_convert_one_resize_longedge(tmp_path):
    from PIL import Image
    src = tmp_path / "b.png"
    Image.fromarray(np.random.default_rng(2).integers(0, 256, (400, 1000, 3)).astype("uint8")).save(src)
    out = tmp_path / "out"
    p = nef_to_tif.convert_one(src, out, out_format="jpeg", resize=nef_to_tif.parse_resize("500"))
    assert max(Image.open(p).size) == 500          # long edge scaled to 500


def test_tif_to_jpeg_roundtrip(tmp_path):
    # write a 16-bit TIFF, then convert TIFF → JPEG (exercises the tiff-aware loader)
    import tifffile
    from PIL import Image
    src = tmp_path / "m.tif"
    tifffile.imwrite(src, (np.random.default_rng(3).integers(0, 65536, (100, 150, 3))).astype(np.uint16),
                     photometric="rgb")
    out = tmp_path / "out"
    p = nef_to_tif.convert_one(src, out, out_format="jpeg")
    assert p.suffix == ".jpg" and Image.open(p).size == (150, 100)
