"""Model-free image-quality scoring — rank a rider's crops, cull blurry shots.

No ML: sharpness (variance of the Laplacian) + exposure. Used to pick the
best-shot per rider after grouping. Absolute values don't matter — the score is
for *ranking* crops of the same rider, so a plain, deterministic metric is ideal.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("peloton.quality")


def sharpness(img: Any) -> float:
    """Variance of the Laplacian (higher = sharper). Uses OpenCV if present
    (ships with ultralytics), else a numpy discrete Laplacian."""
    import numpy as np  # noqa: PLC0415

    a = np.asarray(img.convert("L"), dtype=np.float64)
    try:
        import cv2  # noqa: PLC0415
        return float(cv2.Laplacian(a, cv2.CV_64F).var())
    except ImportError:
        lap = (-4.0 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
               + np.roll(a, 1, 1) + np.roll(a, -1, 1))
        return float(lap[1:-1, 1:-1].var())


def focus_score(img: Any, *, height: int = 512) -> float:
    """Resolution-normalized sharpness: resize to a fixed ``height`` first, then
    Laplacian variance — so photos of different megapixels are comparable and a
    single ``--min-sharpness`` threshold means the same thing across a folder."""
    if img.height > height:
        img = img.resize((max(1, img.width * height // img.height), height))
    return sharpness(img)


def exposure(img: Any) -> float:
    """0..1 (1 = well-exposed). Penalizes distance from mid-grey and clipping."""
    import numpy as np  # noqa: PLC0415

    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    clip = float((a < 0.02).mean() + (a > 0.98).mean())
    return max(0.0, 1.0 - 2.0 * abs(float(a.mean()) - 0.5) - clip)


def score(img: Any, *, face_box: Any = None) -> dict[str, float]:
    """Rank-oriented quality score. If ``face_box`` is given, sharpness is
    measured on the face region — that's what matters for a rider portrait."""
    region = img
    if face_box is not None:
        x1, y1, x2, y2 = (int(v) for v in face_box)
        if x2 > x1 and y2 > y1:
            region = img.crop((x1, y1, x2, y2))
    sh = sharpness(region)
    ex = exposure(img)
    sh_n = sh / (sh + 300.0)          # soft 0..1 for combining
    return {"score": round(0.7 * sh_n + 0.3 * ex, 4),
            "sharpness": round(sh, 1), "exposure": round(ex, 3)}
