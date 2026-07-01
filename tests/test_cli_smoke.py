"""Smoke tests for the CLIs: --help exits 0, and the pipeline runs end-to-end
in --use-mock mode (offline)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "src" / "peloton" / "tools"
CLIS = ["detect_riders.py", "crop_riders.py", "enhance_image.py", "process_photo.py", "batch_photos.py", "group_riders.py", "cull_blurry.py"]


def _run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


def test_each_cli_help_exits_zero():
    for cli in CLIS:
        r = _run(str(TOOLS / cli), "--help")
        assert r.returncode == 0, f"{cli} --help failed: {r.stderr}"
        assert "usage" in r.stdout.lower()


def test_detect_riders_cli_mock(tmp_path):
    from PIL import Image
    img = tmp_path / "g.jpg"
    Image.new("RGB", (800, 500), "gray").save(img)
    r = _run(str(TOOLS / "detect_riders.py"), "--image", str(img), "--use-mock")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["n_riders"] == 3
    assert out["riders"][0]["has_bike"] is True


def test_process_photo_cli_mock(tmp_path):
    from PIL import Image
    img = tmp_path / "g.jpg"
    Image.new("RGB", (800, 500), "gray").save(img)
    out_dir = tmp_path / "out"
    r = _run(str(TOOLS / "process_photo.py"), "--image", str(img),
             "--out-dir", str(out_dir), "--use-mock", "--scale", "2")
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)
    assert summary["n_riders"] == 3
    assert len(list(out_dir.glob("*.jpg"))) == 3
