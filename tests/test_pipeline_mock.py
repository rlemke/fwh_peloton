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


def test_process_photo_segment_mock_writes_cutouts(photo, tmp_path):
    out = tmp_path / "seg"
    summary = pipeline.process_photo(
        photo, out, use_mock=True, scale=2, restore_faces=False,
        segment=True, cutout_bg="white", upscale_backend="lanczos")
    assert summary["n_riders"] == 3
    for r in summary["riders"]:
        assert r["segmented"] is True
        assert r["cutout_bg"] == "white"
        assert Path(r["output"]).is_file()
    assert len(list(out.glob("*.jpg"))) == 3


def test_process_photo_segment_transparent_is_png(photo, tmp_path):
    out = tmp_path / "seg_t"
    summary = pipeline.process_photo(
        photo, out, use_mock=True, scale=2, restore_faces=False,
        segment=True, cutout_bg="transparent", upscale_backend="lanczos")
    from PIL import Image
    for r in summary["riders"]:
        p = Path(r["output"])
        assert p.suffix == ".png"
        assert Image.open(p).mode == "RGBA"


def test_frame_both_writes_single_and_framed(photo, tmp_path):
    out = tmp_path / "fr"
    summary = pipeline.process_photo(
        photo, out, use_mock=True, scale=2, restore_faces=False,
        frame="both", aspect=0.8, upscale_backend="lanczos")
    for r in summary["riders"]:
        kinds = {o["kind"] for o in r["outputs"]}
        assert kinds == {"single", "framed"}
        for o in r["outputs"]:
            assert Path(o["output"]).is_file()
    assert len(list(out.glob("*_single.jpg"))) == 3
    assert len(list(out.glob("*_framed.jpg"))) == 3


def test_framed_size_is_exact(photo, tmp_path):
    out = tmp_path / "sz"
    summary = pipeline.process_photo(
        photo, out, use_mock=True, scale=2, restore_faces=False,
        frame="framed", out_size=(400, 500), upscale_backend="lanczos")
    for r in summary["riders"]:
        assert r["outputs"][0]["output_size"] == [400, 500]   # exact target size


def test_framed_sharpen_increases_crispness(tmp_path):
    # A framed output is downscaled to fit the target, softening the upscale;
    # sharpen_framed>0 must recover crispness. Needs a textured photo (unsharp is
    # a no-op on the flat mock fixture).
    import numpy as np
    from PIL import Image

    from _peloton_tools import quality
    rng = np.random.default_rng(7)
    p = tmp_path / "textured.jpg"
    Image.fromarray(rng.integers(0, 256, (600, 900, 3)).astype("uint8")).save(p)

    def run(strength, sub):
        s = pipeline.process_photo(
            p, tmp_path / sub, use_mock=True, scale=2, restore_faces=False,
            frame="framed", out_size=(400, 500), upscale_backend="lanczos",
            sharpen_framed=strength)
        return Image.open(s["riders"][0]["outputs"][0]["output"])

    soft, sharp = run(0.0, "soft"), run(150.0, "sharp")
    assert quality.focus_score(sharp) > quality.focus_score(soft)


def test_framed_embeds_dpi_single_does_not(photo, tmp_path):
    from PIL import Image
    out = tmp_path / "dpi"
    summary = pipeline.process_photo(
        photo, out, use_mock=True, scale=2, restore_faces=False,
        frame="both", out_size=(400, 600), upscale_backend="lanczos", dpi=300)
    for r in summary["riders"]:
        by_kind = {o["kind"]: o["output"] for o in r["outputs"]}
        assert Image.open(by_kind["framed"]).info.get("dpi") == (300, 300)
        assert Image.open(by_kind["single"]).info.get("dpi") is None


def test_framed_needs_aspect(photo, tmp_path):
    # a framed run with no aspect/out_size is a usage error
    with pytest.raises(ValueError):
        pipeline.process_photo(photo, tmp_path / "e", use_mock=True, frame="framed")


def test_focus_box_stays_in_bounds(photo):
    from _peloton_tools import images
    img = images.load_image(photo)
    w, h = images.size(img)
    for r in detect.detect_riders(img, use_mock=True):
        x1, y1, x2, y2 = r.focus_box(0.2, w, h)
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
