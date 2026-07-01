"""auto_brighten: lighten dark images via gamma, preserve highlights, no-op when bright."""

from __future__ import annotations

from PIL import Image

from _peloton_tools import enhance, quality


def test_dark_image_is_lightened_toward_target():
    dark = Image.new("RGB", (32, 32), (40, 40, 40))
    out, gamma = enhance.auto_brighten(dark, target=120)
    assert gamma > 1.0
    assert quality.mean_luma(out) > quality.mean_luma(dark)
    assert abs(quality.mean_luma(out) - 120) < 15   # lands near the target


def test_already_bright_is_noop():
    bright = Image.new("RGB", (32, 32), (140, 140, 140))
    out, gamma = enhance.auto_brighten(bright, target=120)
    assert gamma == 1.0
    assert out is bright


def test_highlights_preserved():
    # a dark frame with a pure-white patch: white must stay white after brightening
    im = Image.new("RGB", (32, 32), (40, 40, 40))
    im.paste((255, 255, 255), (0, 0, 8, 8))
    out, gamma = enhance.auto_brighten(im, target=120)
    assert gamma > 1.0
    assert out.getpixel((2, 2)) == (255, 255, 255)


def test_meter_region_drives_correction():
    # frame is bright overall, but the metered region is dark → still corrected
    im = Image.new("RGB", (64, 64), (200, 200, 200))
    dark_region = Image.new("RGB", (16, 16), (40, 40, 40))
    _, gamma = enhance.auto_brighten(im, meter=dark_region, target=120)
    assert gamma > 1.0
