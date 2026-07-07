"""Parallel recursive copy — structure mirroring, resume, worker resolution."""

from __future__ import annotations

from _peloton_tools import copytree


def _tree(root):
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "x.txt").write_text("hello")
    (root / "a" / "y.txt").write_text("yo")
    (root / "z.txt").write_text("zz")


def test_copy_tree_mirrors_structure(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _tree(src)
    dst = tmp_path / "dst"
    s = copytree.copy_tree(src, dst, workers=2)
    assert s["copied"] == 3 and s["failed"] == 0
    assert (dst / "a" / "b" / "x.txt").read_text() == "hello"
    assert (dst / "a" / "y.txt").is_file() and (dst / "z.txt").is_file()


def test_copy_tree_resume_skips(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _tree(src)
    dst = tmp_path / "dst"
    copytree.copy_tree(src, dst)
    again = copytree.copy_tree(src, dst)          # same-size files already present
    assert again["copied"] == 0 and again["skipped"] == 3


def test_resolve_workers():
    assert copytree._resolve_workers(4) == 4
    n = copytree._resolve_workers("auto")
    assert n >= 2                                  # I/O-bound floor
