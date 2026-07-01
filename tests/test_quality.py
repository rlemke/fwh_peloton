"""Model-free quality scoring tests (no models)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from _peloton_tools import quality


def test_sharpness_sharp_greater_than_blurred():
    rng = np.random.default_rng(0)
    sharp = Image.fromarray(rng.integers(0, 256, (64, 64, 3)).astype("uint8"))
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))
    assert quality.sharpness(sharp) > quality.sharpness(blurred)


def test_focus_score_blurred_lower_and_resolution_normalized():
    rng = np.random.default_rng(1)
    sharp = Image.fromarray(rng.integers(0, 256, (900, 700, 3)).astype("uint8"))
    blurred = sharp.filter(ImageFilter.GaussianBlur(5))
    assert quality.focus_score(sharp) > quality.focus_score(blurred)
    # normalization: same content at 2x size scores comparably (within tolerance)
    big = sharp.resize((1400, 1800))
    assert abs(quality.focus_score(sharp) - quality.focus_score(big)) < quality.focus_score(sharp) * 0.5


def test_mean_luma_signed_and_ordered():
    assert quality.mean_luma(Image.new("RGB", (16, 16), (30, 30, 30))) < \
        quality.mean_luma(Image.new("RGB", (16, 16), (200, 200, 200)))


def test_exposure_midgrey_high_black_low():
    assert quality.exposure(Image.new("RGB", (32, 32), (128, 128, 128))) > 0.9
    assert quality.exposure(Image.new("RGB", (32, 32), (0, 0, 0))) < 0.3


def test_score_shape_and_face_region():
    img = Image.new("RGB", (64, 64), (100, 120, 140))
    s = quality.score(img, face_box=(10, 10, 40, 40))
    assert set(s) == {"score", "sharpness", "exposure"}
    assert 0.0 <= s["score"] <= 1.0
