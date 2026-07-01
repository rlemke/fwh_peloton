"""Pure box-geometry tests — the crop-correctness core (no models, no images)."""

from __future__ import annotations

from _peloton_tools import crop


def test_clamp_to_bounds_and_ints():
    assert crop.clamp_box((-10, -5, 50.6, 200), 100, 100) == (0, 0, 51, 100)


def test_clamp_swaps_inverted():
    assert crop.clamp_box((60, 70, 10, 20), 100, 100) == (10, 20, 60, 70)


def test_union_of_person_and_bike():
    person = (30, 10, 70, 80)
    bike = (25, 55, 75, 95)
    assert crop.union(person, bike) == (25, 10, 75, 95)


def test_union_ignores_none():
    assert crop.union((10, 10, 20, 20), None) == (10, 10, 20, 20)


def test_iou_ranges():
    a = (0, 0, 10, 10)
    assert crop.iou(a, a) == 1.0
    assert crop.iou(a, (100, 100, 110, 110)) == 0.0
    assert 0.0 < crop.iou(a, (5, 5, 15, 15)) < 1.0


def test_pad_box_expands_and_clamps():
    # 40x40 box, 25% pad → +10 each side, but clamped to the 100x100 image.
    out = crop.pad_box((0, 0, 40, 40), 0.25, 100, 100)
    assert out == (0, 0, 50, 50)  # left/top clamp at 0, right/bottom +10


def test_crop_box_dims_match(monkeypatch):
    from PIL import Image
    img = Image.new("RGB", (100, 80), "white")
    out = crop.crop_box(img, (10, 10, 60, 50))
    assert out.size == (50, 40)


def test_cutout_backgrounds():
    import numpy as np
    from PIL import Image
    img = Image.new("RGB", (100, 100), (10, 20, 30))
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    # solid hex colour outside the mask
    red = crop.cutout(img, mask, (0, 0, 100, 100), bg="#ff0000", feather=0)
    assert red.mode == "RGB" and red.size == (100, 100)
    assert np.asarray(red)[0, 0].tolist() == [255, 0, 0]     # corner = red bg
    assert np.asarray(red)[50, 50].tolist() == [10, 20, 30]  # centre = subject

    # bokeh + transparent variants keep the right mode/size
    assert crop.cutout(img, mask, (0, 0, 100, 100), bg="bokeh").mode == "RGB"
    t = crop.cutout(img, mask, (0, 0, 100, 100), bg="transparent")
    assert t.mode == "RGBA"
