"""Handler for peloton.Portraits — one photo → per-rider enhanced portraits."""

from __future__ import annotations

from typing import Any

from peloton.handlers.shared import peloton_utils as U

NAMESPACE = "peloton.Portraits"


def handle_process_photo(params: dict[str, Any]) -> dict[str, Any]:
    s = U.pipeline.process_photo(
        params["image_path"], params["out_dir"],
        require_bike=params.get("require_bike", True),
        context=params.get("context", True),
        auto_brighten=params.get("auto_brighten", True),
        restore_faces=params.get("face_restore", False),
        out_format=params.get("out_format", "tiff"),
        use_mock=params.get("use_mock", False),
    )
    return {"n_riders": s["n_riders"], "outputs": [r["output"] for r in s["riders"]]}


_DISPATCH: dict[str, Any] = {
    f"{NAMESPACE}.ProcessPhoto": handle_process_photo,
}


def handle(payload: dict) -> dict:
    return _DISPATCH[payload["_facet_name"]](payload)


def register_handlers(runner) -> None:
    for facet_name in _DISPATCH:
        runner.register_handler(facet_name=facet_name, module_uri=__name__, entrypoint="handle")
