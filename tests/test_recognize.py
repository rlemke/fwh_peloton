"""Face-clustering tests using the deterministic mock embeddings (no models)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from _peloton_tools import recognize


def _img(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (48, 48, 3)).astype("uint8"))


def test_mock_embedding_deterministic():
    a = _img(1)
    e1 = recognize.face_embeddings(a, use_mock=True)[0][0]
    e2 = recognize.face_embeddings(a.copy(), use_mock=True)[0][0]
    assert float(np.dot(e1, e2)) > 0.999  # identical image → identical embedding


def test_cluster_groups_same_rider_apart_from_others():
    a, b = _img(1), _img(2)
    imgs = [a, a.copy(), b, a.copy(), b.copy()]  # 3×a, 2×b interleaved
    embs = [recognize.face_embeddings(im, use_mock=True)[0][0] for im in imgs]
    labels = recognize.cluster(embs, threshold=0.9)
    assert labels[0] == labels[1] == labels[3]     # the three 'a' crops
    assert labels[2] == labels[4]                   # the two 'b' crops
    assert labels[0] != labels[2]                   # different riders
    assert len(set(labels)) == 2
