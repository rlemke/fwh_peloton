"""Derive-JPEGs step: a directory of 16-bit TIFF masters → 8-bit JPEG copies."""

from __future__ import annotations

import numpy as np
import pytest

import tiffs_to_jpegs
from _peloton_tools import images


@pytest.fixture()
def tiff_dir(tmp_path):
    """Two 16-bit TIFF 'masters' written the same way the pipeline writes them."""
    d = tmp_path / "tiff"
    d.mkdir()
    rng = np.random.default_rng(3)
    for name in ("D1_rider00_single", "D1_rider00_context"):
        arr = (rng.integers(0, 65536, (120, 90, 3))).astype(np.uint16)
        images.save_tiff16(arr, d / f"{name}.tif")
    # a non-TIFF that must be ignored
    (d / "notes.txt").write_text("ignore me")
    return d


def test_derives_one_jpeg_per_tiff(tiff_dir, tmp_path):
    out = tmp_path / "jpeg"
    summary = tiffs_to_jpegs.convert_dir(tiff_dir, out, quality=100)
    assert summary["total"] == 2
    assert summary["converted"] == 2
    assert summary["failed"] == 0
    jpgs = sorted(p.name for p in out.glob("*.jpg"))
    assert jpgs == ["D1_rider00_context.jpg", "D1_rider00_single.jpg"]


def test_output_is_8bit_rgb_same_dims(tiff_dir, tmp_path):
    from PIL import Image
    out = tmp_path / "jpeg"
    tiffs_to_jpegs.convert_dir(tiff_dir, out)
    im = Image.open(out / "D1_rider00_single.jpg")
    assert im.mode == "RGB"          # 8-bit
    assert im.size == (90, 120)      # dims preserved (W, H)


def test_skip_existing_then_overwrite(tiff_dir, tmp_path):
    out = tmp_path / "jpeg"
    tiffs_to_jpegs.convert_dir(tiff_dir, out)
    # second run skips (idempotent), converts nothing
    again = tiffs_to_jpegs.convert_dir(tiff_dir, out)
    assert again["converted"] == 0 and again["skipped"] == 2
    # overwrite re-converts
    forced = tiffs_to_jpegs.convert_dir(tiff_dir, out, overwrite=True)
    assert forced["converted"] == 2 and forced["skipped"] == 0
