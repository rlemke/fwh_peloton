"""Rider segmentation — tight per-rider cutout masks via box-prompted SAM.

Given a rider's bounding box (from ``detect``), SAM returns a pixel mask of just
that rider+bike, so we can cut them out of the background instead of keeping a
rectangular crop. Uses Ultralytics' SAM (``mobile_sam.pt`` by default — small +
fast); the box prompt keeps it to the one rider.
"""

from __future__ import annotations

import logging
from typing import Any

from _peloton_tools import crop as _crop

NAMESPACE = "peloton"
log = logging.getLogger("peloton.segment")

_SAM_CACHE: dict[str, Any] = {}


def _load_sam(model: str) -> Any:
    if model in _SAM_CACHE:
        return _SAM_CACHE[model]
    try:
        from ultralytics import SAM  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Segmentation needs Ultralytics SAM. Install it: pip install '.[detect]'\n"
            "Or run with use_mock=True / --use-mock."
        ) from exc
    log.info("loading SAM weights: %s", model)
    m = SAM(model)
    _SAM_CACHE[model] = m
    return m


def _box_mask(box: _crop.Box, width: int, height: int) -> Any:
    import numpy as np  # noqa: PLC0415
    m = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = _crop.clamp_box(box, width, height)
    m[y1:y2, x1:x2] = True
    return m


def segment_box(img: Any, box: _crop.Box, *, model: str = "mobile_sam.pt",
                use_mock: bool = False) -> Any:
    """Boolean ``HxW`` mask of the object in ``box`` (SAM box prompt).

    Falls back to a filled-box mask if SAM finds nothing (or in mock mode), so
    the cutout path always yields something.
    """
    import numpy as np  # noqa: PLC0415

    w, h = int(img.width), int(img.height)
    if use_mock:
        return _box_mask(box, w, h)

    sam = _load_sam(model)
    res = sam.predict(np.asarray(img.convert("RGB")), bboxes=[list(box)], verbose=False)
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        log.warning("SAM found no mask for box %s — using filled box", box)
        return _box_mask(box, w, h)
    mask = res[0].masks.data[0].detach().cpu().numpy() > 0.5
    log.info("segmented rider: %.1f%% of box filled",
             100.0 * mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])].mean())
    return mask
