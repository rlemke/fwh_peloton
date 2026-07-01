"""Deterministic offline mock detector — for tests and ``--use-mock``.

Returns a fixed set of "riders" laid out across the image so the whole
load → crop → enhance → output pipeline can be exercised with no models and no
network. The boxes scale with the image so cropped outputs are always in-bounds.
"""

from __future__ import annotations

from typing import Any

from _peloton_tools.detect import Rider


def mock_riders(img: Any, n: int = 3) -> list[Rider]:
    """``n`` evenly-spaced riders across the middle band of the image, each with
    a person box and a slightly-lower bicycle box."""
    w, h = int(img.width), int(img.height)
    riders: list[Rider] = []
    cell = w / n
    pw = cell * 0.6          # person width per cell
    ptop, pbot = h * 0.20, h * 0.75
    for i in range(n):
        cx = cell * (i + 0.5)
        px1, px2 = cx - pw / 2, cx + pw / 2
        person = (px1, ptop, px2, pbot)
        # bike overlaps the lower half of the person (legs on the bike)
        bike = (px1 - pw * 0.1, h * 0.55, px2 + pw * 0.1, h * 0.92)
        riders.append(
            Rider(person_box=person, score=0.90 - 0.05 * i, bike_box=bike,
                  index=i, meta={"mock": True})
        )
    return riders
