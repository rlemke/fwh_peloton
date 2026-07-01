"""End-to-end pipeline test in offline mock mode — no models, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from _peloton_tools import detect, pipeline


@pytest.fixture()
def photo(tmp_path):
    from PIL import Image
    p = tmp_path / "group.jpg"
    Image.new("RGB", (900, 600), (135, 180, 220)).save(p)
    return p


def test_mock_detects_three_riders(photo):
    from _peloton_tools import images
    img = images.load_image(photo)
    riders = detect.detect_riders(img, use_mock=True)
    assert len(riders) == 3
    assert all(r.bike_box is not None for r in riders)
    # sorted largest-first, re-indexed
    assert [r.index for r in riders] == [0, 1, 2]


def test_process_photo_writes_one_output_per_rider(photo, tmp_path):
    out = tmp_path / "out"
    # Force the non-ML backends so the offline suite stays fast + deterministic
    # regardless of whether the .[enhance] models are installed on the host.
    summary = pipeline.process_photo(
        photo, out, use_mock=True, scale=2, restore_faces=True,
        upscale_backend="lanczos", face_backend="none")
    assert summary["n_riders"] == 3
    assert len(summary["riders"]) == 3
    for r in summary["riders"]:
        op = Path(r["output"])
        assert op.is_file()
        # upscaled: output larger than the source crop (x2)
        assert r["output_size"][0] > (r["focus_box"][2] - r["focus_box"][0])
        assert r["upscale_backend"] == "lanczos"
        assert r["face_backend"] == "none"
    # 3 files on disk
    assert len(list(out.glob("group_rider*.jpg"))) == 3


def test_focus_box_stays_in_bounds(photo):
    from _peloton_tools import images
    img = images.load_image(photo)
    w, h = images.size(img)
    for r in detect.detect_riders(img, use_mock=True):
        x1, y1, x2, y2 = r.focus_box(0.2, w, h)
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
