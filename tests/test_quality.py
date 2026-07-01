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


def test_exposure_midgrey_high_black_low():
    assert quality.exposure(Image.new("RGB", (32, 32), (128, 128, 128))) > 0.9
    assert quality.exposure(Image.new("RGB", (32, 32), (0, 0, 0))) < 0.3


def test_score_shape_and_face_region():
    img = Image.new("RGB", (64, 64), (100, 120, 140))
    s = quality.score(img, face_box=(10, 10, 40, 40))
    assert set(s) == {"score", "sharpness", "exposure"}
    assert 0.0 <= s["score"] <= 1.0
