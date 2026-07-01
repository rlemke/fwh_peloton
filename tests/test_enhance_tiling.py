"""Regression test for the tiled upscaler's coverage — a tiling gap once left a
black rectangle in the bottom-right corner of large crops."""

from __future__ import annotations

import pytest


def test_tiled_upscale_covers_every_pixel_no_black():
    torch = pytest.importorskip("torch")
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    from _peloton_tools import enhance

    class _FakeSR:
        scale = 2

        def __call__(self, t):  # [1,3,h,w] → [1,3,2h,2w]
            return F.interpolate(t, scale_factor=2, mode="nearest")

    # Non-tile-aligned dimensions stress the right/bottom edge handling.
    img = Image.new("RGB", (203, 149), (200, 100, 50))
    out = enhance._run_tiled(_FakeSR(), img, tile=64, overlap=8)

    assert out.size == (406, 298)
    a = np.asarray(out)
    # No uncovered (all-black) pixel anywhere, incl. the bottom-right corner.
    assert int((a.sum(axis=2) == 0).sum()) == 0
    assert a[-1, -1].tolist() != [0, 0, 0]
    # Constant-colour input is preserved everywhere (nearest upscale).
    assert abs(int(a[:, :, 0].mean()) - 200) <= 2
