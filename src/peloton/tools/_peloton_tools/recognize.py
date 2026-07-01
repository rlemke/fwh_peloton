"""Face recognition + same-rider clustering — group photos/crops by person.

An event has the same riders across many photos. We embed each rider's face
(InsightFace / ArcFace, 512-d L2-normalized) and cluster by cosine similarity, so
all shots of one person land in one group. The heavy import is lazy; a
deterministic mock (embedding from the image bytes) makes the clustering testable
offline.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

log = logging.getLogger("peloton.recognize")

_APP: Any = None


def _load_app(model: str = "buffalo_l") -> Any:
    global _APP
    if _APP is not None:
        return _APP
    try:
        from insightface.app import FaceAnalysis  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Face grouping needs InsightFace. Install it: pip install '.[recognize]'\n"
            "Or run with use_mock=True / --use-mock for the offline path."
        ) from exc
    log.info("loading InsightFace model: %s", model)
    app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    _APP = app
    return app


def _mock_embedding(img: Any) -> Any:
    """Deterministic pseudo-embedding from the image content — identical images
    map to identical vectors (so tests can build clusters)."""
    import numpy as np  # noqa: PLC0415

    small = img.convert("L").resize((16, 16))
    seed = int(hashlib.sha256(small.tobytes()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(512).astype("float32")
    return v / (np.linalg.norm(v) + 1e-9)


def face_embeddings(img: Any, *, model: str = "buffalo_l",
                    use_mock: bool = False) -> list[tuple[Any, Any]]:
    """List of ``(embedding[512], bbox)`` for the faces in ``img`` — largest first."""
    import numpy as np  # noqa: PLC0415

    if use_mock:
        return [(_mock_embedding(img), (0.0, 0.0, float(img.width), float(img.height)))]

    app = _load_app(model)
    faces = app.get(np.asarray(img.convert("RGB"))[:, :, ::-1])
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return [(f.normed_embedding, tuple(float(v) for v in f.bbox)) for f in faces]


def cluster(embeddings: list[Any], *, threshold: float = 0.35) -> list[int]:
    """Greedy online clustering by cosine similarity (embeddings are L2-normed, so
    cosine = dot). Returns a cluster id per embedding. ``threshold`` = min cosine
    similarity to join a cluster (higher = stricter / more clusters)."""
    import numpy as np  # noqa: PLC0415

    centroids: list[Any] = []
    members: list[list[int]] = []
    labels: list[int] = []
    for i, e in enumerate(embeddings):
        best, best_sim = -1, threshold
        for ci, c in enumerate(centroids):
            sim = float(np.dot(e, c))
            if sim >= best_sim:
                best_sim, best = sim, ci
        if best < 0:
            best = len(centroids)
            centroids.append(np.asarray(e, dtype="float32").copy())
            members.append([])
        members[best].append(i)
        m = np.mean([embeddings[j] for j in members[best]], axis=0)
        centroids[best] = m / (np.linalg.norm(m) + 1e-9)   # keep centroid normed
        labels.append(best)
    log.info("clustered %d face(s) into %d rider group(s)",
             len(embeddings), len(centroids))
    return labels
