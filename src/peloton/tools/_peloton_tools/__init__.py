"""Shared library behind the peloton tools + (future) handlers.

Package-unique name (``_peloton_tools``) per the tools-pattern contract so it
never collides in ``sys.modules`` with other co-installed domain packages.

Modules:
    images   — load/save/normalize images (Pillow)
    crop     — pure box geometry + cropping (no models; fully testable)
    detect   — rider detection (YOLO person+bicycle) with an offline mock
    enhance  — upscale + face-restore with graceful non-ML fallbacks
    pipeline — orchestrates load → detect → crop → enhance → per-rider outputs
    peloton_mocks — deterministic offline detector for tests / --use-mock
    sidecar/storage — cache primitives (agent-spec/cache-layout)
"""

NAMESPACE = "peloton"
