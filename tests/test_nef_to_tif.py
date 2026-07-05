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
