"""FFL-workflow layer — domain discovery, handler registration, handler execution.

Skipped where facetwork isn't installed (the tools/library run without it)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("facetwork")
import peloton  # noqa: E402


def test_domain_registers_expected_facets():
    assert peloton.domain is not None and peloton.domain.name == "peloton"

    class FakeRunner:
        def __init__(self):
            self.names: set[str] = set()

        def register_handler(self, facet_name, module_uri, entrypoint="handle", **kw):
            self.names.add(facet_name)

    r = FakeRunner()
    peloton.domain.register_handlers(r)
    assert r.names == {
        "peloton.Ingest.ListImages",
        "peloton.Ingest.ConvertRaw",
        "peloton.Portraits.ProcessPhoto",
    }


def test_process_photo_handler_mock(tmp_path):
    from PIL import Image

    from peloton.handlers.portraits.portraits_handlers import handle
    p = tmp_path / "g.jpg"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (300, 400, 3)).astype("uint8")).save(p)
    out = handle({"_facet_name": "peloton.Portraits.ProcessPhoto", "image_path": str(p),
                  "out_dir": str(tmp_path / "o"), "use_mock": True, "out_format": "jpg"})
    assert out["n_riders"] == 3 and len(out["outputs"]) == 3
