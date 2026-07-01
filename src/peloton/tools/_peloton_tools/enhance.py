"""Upscale + face restoration, with graceful degradation.

Both stages are *best-effort*: if the ML backend isn't installed the pipeline
still produces a usable output (a Lanczos upscale / the unmodified crop) instead
of failing. Each function returns ``(image, backend_used)`` so callers can record
what actually ran.

Upscale backends (tried in order for backend="auto"):
    realesrgan-ncnn   — the standalone `realesrgan-ncnn-vulkan` binary (no torch)
    realesrgan        — the Python package (needs torch + weights)
    lanczos           — PIL high-quality resample (always available)

Face-restore backends:
    gfpgan / codeformer — if importable
    none                — passthrough (always available)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("peloton.enhance")

_NCNN_BIN = "realesrgan-ncnn-vulkan"
_WEIGHTS_DIR = os.path.expanduser(
    os.environ.get("FW_PELOTON_WEIGHTS_DIR", "~/.cache/peloton/weights"))
_SPANDREL_CACHE: dict[str, Any] = {}


def _torch_device() -> str:
    """MPS (Apple GPU) → CUDA → CPU, overridable with FW_PELOTON_DEVICE."""
    dev = os.environ.get("FW_PELOTON_DEVICE")
    if dev:
        return dev
    try:
        import torch  # noqa: PLC0415
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _weights_path(filename: str, env: str) -> str:
    return os.environ.get(env) or os.path.join(_WEIGHTS_DIR, filename)


def upscale(img: Any, scale: int = 4, backend: str = "auto") -> tuple[Any, str]:
    """Return an upscaled copy of ``img`` and the backend that produced it."""
    order = ([backend] if backend != "auto"
             else ["realesrgan-ncnn", "realesrgan", "lanczos"])
    for b in order:
        try:
            if b == "realesrgan-ncnn":
                out = _upscale_ncnn(img, scale)
            elif b == "realesrgan":
                out = _upscale_realesrgan_py(img, scale)
            elif b == "lanczos":
                out = _upscale_lanczos(img, scale)
            else:
                raise ValueError(f"unknown upscale backend: {b!r}")
            if out is not None:
                log.info("upscaled x%d via %s (%dx%d → %dx%d)", scale, b,
                         img.width, img.height, out.width, out.height)
                return out, b
        except Exception as exc:  # noqa: BLE001 - fall through to the next backend
            log.warning("upscale backend %s unavailable/failed: %s", b, exc)
    # order was explicit + failed; last resort so we never return None
    return _upscale_lanczos(img, scale), "lanczos"


def _upscale_lanczos(img: Any, scale: int) -> Any:
    from PIL import Image  # noqa: PLC0415
    return img.resize((img.width * scale, img.height * scale), Image.LANCZOS)


def _upscale_ncnn(img: Any, scale: int) -> Any | None:
    exe = shutil.which(_NCNN_BIN)
    if not exe:
        return None
    from PIL import Image  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "in.png", Path(td) / "out.png"
        img.save(ip)
        # -n picks the x4 model; the tool always produces x4, we resize if needed.
        subprocess.run([exe, "-i", str(ip), "-o", str(op), "-s", "4"],
                       check=True, capture_output=True, timeout=600)
        out = Image.open(op).convert("RGB")
        if scale != 4:
            out = out.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        return out


def _load_spandrel(weights: str) -> Any:
    if weights in _SPANDREL_CACHE:
        return _SPANDREL_CACHE[weights]
    from spandrel import ImageModelDescriptor, ModelLoader  # noqa: PLC0415
    model = ModelLoader().load_from_file(weights)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(f"{weights} is not a single-image super-resolution model")
    model.to(_torch_device()).eval()
    log.info("loaded Real-ESRGAN weights %s (x%d) on %s",
             Path(weights).name, model.scale, _torch_device())
    _SPANDREL_CACHE[weights] = model
    return model


def _run_tiled(model: Any, img: Any, tile: int = 384, overlap: int = 16) -> Any:
    """Tiled x``model.scale`` inference with overlap-blend — bounds memory so
    large rider crops don't OOM on MPS/CPU."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    device = _torch_device()
    s = int(model.scale)
    W, H = img.width, img.height
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    out = np.zeros((H * s, W * s, 3), dtype=np.float32)
    acc = np.zeros((H * s, W * s, 1), dtype=np.float32)
    step = max(1, tile - overlap)
    # Tile origins that guarantee FULL coverage: start every `step`, but always
    # include a final tile flush against the right/bottom edge (W-tile / H-tile)
    # so the last strip is never skipped (that left a black bottom-right corner).
    def _starts(extent: int) -> list[int]:
        xs = list(range(0, max(1, extent - tile + 1), step))
        last = max(0, extent - tile)
        if not xs or xs[-1] != last:
            xs.append(last)
        return xs

    with torch.no_grad():
        for y in _starts(H):
            for x in _starts(W):
                y2, x2 = min(y + tile, H), min(x + tile, W)
                patch = arr[y:y2, x:x2, :].transpose(2, 0, 1)
                t = torch.from_numpy(patch)[None].to(device)
                r = model(t).clamp(0, 1)[0].detach().cpu().float().numpy().transpose(1, 2, 0)
                oy, ox = y * s, x * s
                out[oy:oy + r.shape[0], ox:ox + r.shape[1], :] += r
                acc[oy:oy + r.shape[0], ox:ox + r.shape[1], :] += 1.0
    if float(acc.min()) == 0.0:  # coverage guard — never emit an unfilled (black) gap
        log.warning("tiling left an uncovered region; filling from a plain resize")
        base = np.asarray(img.resize((W * s, H * s), Image.LANCZOS), dtype=np.float32)
        gap = (acc[:, :, 0] == 0)
        out[gap] = base[gap]
        acc[gap] = 1.0
    out = (out / np.maximum(acc, 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _upscale_realesrgan_py(img: Any, scale: int) -> Any | None:
    """Real-ESRGAN via spandrel (plain torch, no basicsr). Returns None if the
    library/weights aren't present so ``auto`` falls back to Lanczos."""
    try:
        import spandrel  # noqa: F401,PLC0415
    except ImportError:
        return None
    weights = _weights_path("RealESRGAN_x4plus.pth", "FW_PELOTON_REALESRGAN_WEIGHTS")
    if not os.path.isfile(weights):
        log.warning("Real-ESRGAN weights missing (%s) — falling back", weights)
        return None
    model = _load_spandrel(weights)
    out = _run_tiled(model, img)
    native = int(model.scale)
    if scale != native:
        from PIL import Image  # noqa: PLC0415
        out = out.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    return out


def restore_faces(img: Any, fidelity: float = 0.7, backend: str = "auto") -> tuple[Any, str]:
    """Best-effort face restoration. ``fidelity`` (0..1) trades identity-fidelity
    (high) vs restoration-strength (low). Returns ``(image, backend_used)``.
    """
    order = ([backend] if backend != "auto" else ["gfpgan", "none"])
    for b in order:
        try:
            if b == "gfpgan":
                out = _restore_gfpgan(img, fidelity)
            elif b == "codeformer":
                out = _restore_codeformer(img, fidelity)
            elif b in ("none", "passthrough"):
                out = img
            else:
                raise ValueError(f"unknown face-restore backend: {b!r}")
            if out is not None:
                if b != "none":
                    log.info("face-restore via %s (fidelity=%.2f)", b, fidelity)
                return out, b
        except Exception as exc:  # noqa: BLE001
            log.warning("face-restore backend %s unavailable/failed: %s", b, exc)
    return img, "none"


_GFPGAN_CACHE: dict[str, Any] = {}


def _get_gfpgan(weights: str) -> Any:
    if weights in _GFPGAN_CACHE:
        return _GFPGAN_CACHE[weights]
    from gfpgan import GFPGANer  # noqa: PLC0415
    r = GFPGANer(model_path=weights, upscale=1, arch="clean",
                 channel_multiplier=2, bg_upsampler=None)
    log.info("loaded GFPGAN %s", Path(weights).name)
    _GFPGAN_CACHE[weights] = r
    return r


def _restore_gfpgan(img: Any, fidelity: float) -> Any | None:
    """GFPGAN face restoration. Returns None (→ passthrough) if the library or
    weights are absent, so ``auto`` degrades cleanly."""
    import sys  # noqa: PLC0415
    try:
        # basicsr (a GFPGAN dep) imports torchvision.transforms.functional_tensor,
        # removed in torchvision>=0.17 — alias it to the current module so the
        # import succeeds on modern torch.
        import torchvision.transforms.functional as _tvf  # noqa: PLC0415
        sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvf)
        import gfpgan  # noqa: F401,PLC0415
    except ImportError:
        return None
    weights = _weights_path("GFPGANv1.4.pth", "FW_PELOTON_GFPGAN_WEIGHTS")
    if not os.path.isfile(weights):
        log.warning("GFPGAN weights missing (%s) — passthrough", weights)
        return None

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    restorer = _get_gfpgan(weights)
    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    # GFPGAN's `weight` blends restored↔original — map our identity `fidelity`.
    _cropped, _restored_faces, restored = restorer.enhance(
        bgr, has_aligned=False, only_center_face=False, paste_back=True,
        weight=float(fidelity))
    if restored is None:
        return None
    return Image.fromarray(restored[:, :, ::-1])


_CODEFORMER_CACHE: dict[str, Any] = {}


def _face_device() -> str:
    """Face detect/restore on CPU by default — retinaface/CodeFormer hit MPS op
    gaps. Override with FW_PELOTON_FACE_DEVICE."""
    return os.environ.get("FW_PELOTON_FACE_DEVICE", "cpu")


def _get_codeformer(weights: str, device: str) -> Any:
    if weights in _CODEFORMER_CACHE:
        return _CODEFORMER_CACHE[weights]
    import spandrel_extra_arches  # noqa: PLC0415
    from spandrel import ModelLoader  # noqa: PLC0415
    spandrel_extra_arches.install()
    net = ModelLoader().load_from_file(weights).model.to(device).eval()
    log.info("loaded CodeFormer %s on %s", Path(weights).name, device)
    _CODEFORMER_CACHE[weights] = net
    return net


def _restore_codeformer(img: Any, fidelity: float) -> Any | None:
    """CodeFormer face restoration — facexlib align/paste + the CodeFormer net
    (spandrel_extra_arches). ``fidelity`` maps to CodeFormer's ``weight`` (higher
    = more faithful to the input, lower = stronger restoration). Returns None
    (→ passthrough) if libs/weights are absent, or the input if no face is found.
    """
    import sys  # noqa: PLC0415
    try:
        import torchvision.transforms.functional as _tvf  # noqa: PLC0415
        sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvf)
        import spandrel_extra_arches  # noqa: F401,PLC0415
        from basicsr.utils import img2tensor, tensor2img  # noqa: PLC0415
        from facexlib.utils.face_restoration_helper import (  # noqa: PLC0415
            FaceRestoreHelper)
        from torchvision.transforms.functional import normalize  # noqa: PLC0415
    except ImportError:
        return None
    weights = _weights_path("codeformer.pth", "FW_PELOTON_CODEFORMER_WEIGHTS")
    if not os.path.isfile(weights):
        log.warning("CodeFormer weights missing (%s) — passthrough", weights)
        return None

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    device = _face_device()
    net = _get_codeformer(weights, device)
    helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1),
                               det_model="retinaface_resnet50", save_ext="png",
                               use_parse=True, device=device)
    helper.clean_all()
    helper.read_image(np.asarray(img.convert("RGB"))[:, :, ::-1].copy())
    if helper.get_face_landmarks_5(only_center_face=False, resize=640,
                                   eye_dist_threshold=5) == 0:
        return img  # no face — leave the crop unchanged
    helper.align_warp_face()
    for cf in helper.cropped_faces:
        ft = img2tensor(cf / 255.0, bgr2rgb=True, float32=True)
        normalize(ft, (0.5,) * 3, (0.5,) * 3, inplace=True)
        with torch.no_grad():
            out = net(ft.unsqueeze(0).to(device), weight=float(fidelity))
            out = out[0] if isinstance(out, (tuple, list)) else out
        helper.add_restored_face(
            tensor2img(out.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype("uint8"))
    helper.get_inverse_affine(None)
    return Image.fromarray(helper.paste_faces_to_input_image()[:, :, ::-1])
