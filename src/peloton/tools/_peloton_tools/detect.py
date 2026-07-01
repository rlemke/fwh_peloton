"""Rider detection — find each cyclist in a group photo.

A *rider* is a ``person`` detection optionally paired with the ``bicycle`` they
are on. We run an object detector (Ultralytics YOLO by default), keep the
``person`` + ``bicycle`` classes, and pair each person with the bicycle whose
box best overlaps theirs (a cyclist's legs overlap their bike).

Backends:
    use_mock=True  → deterministic boxes from ``peloton_mocks`` (offline / tests)
    backend="yolo" → Ultralytics YOLO (lazy import; ``pip install '.[detect]'``)

The heavy import is lazy so this module (and the mock path) load without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from _peloton_tools import crop as _crop

NAMESPACE = "peloton"
log = logging.getLogger("peloton.detect")

# COCO class ids used by the default YOLO weights.
_COCO_PERSON = 0
_COCO_BICYCLE = 1

_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class Rider:
    """One detected rider: a person box, an optional paired bicycle box, and the
    detector's confidence."""

    person_box: _crop.Box
    score: float
    bike_box: _crop.Box | None = None
    index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def focus_box(self, pad_frac: float, width: int, height: int) -> _crop.Box:
        """The padded box to crop for this rider — person ∪ bicycle, expanded."""
        base = _crop.union(self.person_box, self.bike_box)
        return _crop.pad_box(base, pad_frac, width, height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "score": round(float(self.score), 4),
            "person_box": [round(float(v), 1) for v in self.person_box],
            "bike_box": ([round(float(v), 1) for v in self.bike_box]
                         if self.bike_box else None),
            "has_bike": self.bike_box is not None,
            **({"meta": self.meta} if self.meta else {}),
        }


def _load_yolo(model: str) -> Any:
    if model in _MODEL_CACHE:
        return _MODEL_CACHE[model]
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Rider detection needs Ultralytics YOLO. Install it:\n"
            "    pip install '.[detect]'   (or: pip install ultralytics)\n"
            "Or run with use_mock=True / --use-mock for the offline path."
        ) from exc
    log.info("loading YOLO weights: %s", model)
    m = YOLO(model)
    _MODEL_CACHE[model] = m
    return m


def _pair_bikes(persons: list[tuple[_crop.Box, float]],
                bikes: list[tuple[_crop.Box, float]],
                min_iou: float) -> list[Rider]:
    """Greedily pair each person with the best-overlapping unused bicycle."""
    used: set[int] = set()
    riders: list[Rider] = []
    # Biggest (nearest/most-prominent) people first.
    order = sorted(range(len(persons)), key=lambda i: _crop.area(persons[i][0]),
                   reverse=True)
    for pi in order:
        pbox, pscore = persons[pi]
        best_j, best_iou = -1, min_iou
        for j, (bbox, _bscore) in enumerate(bikes):
            if j in used:
                continue
            ov = _crop.iou(pbox, bbox)
            if ov > best_iou:
                best_iou, best_j = ov, j
        bike_box = None
        if best_j >= 0:
            used.add(best_j)
            bike_box = bikes[best_j][0]
        riders.append(Rider(person_box=pbox, score=pscore, bike_box=bike_box))
    return riders


def detect_riders(
    img: Any,
    *,
    conf: float = 0.25,
    require_bike: bool = False,
    min_pair_iou: float = 0.02,
    backend: str = "yolo",
    model: str = "yolov8n.pt",
    use_mock: bool = False,
) -> list[Rider]:
    """Detect riders in a ``PIL.Image``. Returns riders sorted largest-first
    (nearest riders first), re-indexed 0..N-1.

    require_bike — drop persons with no paired bicycle (filters spectators).
    """
    if use_mock:
        from _peloton_tools import peloton_mocks  # noqa: PLC0415
        riders = peloton_mocks.mock_riders(img)
    elif backend == "yolo":
        riders = _detect_yolo(img, conf=conf, model=model, min_pair_iou=min_pair_iou)
    else:
        raise ValueError(f"unknown detect backend: {backend!r}")

    if require_bike:
        riders = [r for r in riders if r.bike_box is not None]
    riders.sort(key=lambda r: _crop.area(r.person_box), reverse=True)
    for i, r in enumerate(riders):
        r.index = i
    log.info("detected %d rider(s)%s", len(riders),
             " (with bike)" if require_bike else "")
    return riders


def _detect_yolo(img: Any, *, conf: float, model: str, min_pair_iou: float) -> list[Rider]:
    m = _load_yolo(model)
    results = m.predict(img, conf=conf, classes=[_COCO_PERSON, _COCO_BICYCLE],
                        verbose=False)
    persons: list[tuple[_crop.Box, float]] = []
    bikes: list[tuple[_crop.Box, float]] = []
    for res in results:
        for b in res.boxes:
            cls = int(b.cls[0])
            box = tuple(float(v) for v in b.xyxy[0].tolist())
            score = float(b.conf[0])
            if cls == _COCO_PERSON:
                persons.append((box, score))  # type: ignore[arg-type]
            elif cls == _COCO_BICYCLE:
                bikes.append((box, score))  # type: ignore[arg-type]
    log.info("yolo: %d person, %d bicycle box(es)", len(persons), len(bikes))
    return _pair_bikes(persons, bikes, min_pair_iou)
