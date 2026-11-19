import pytest

from maia_face import weights
from maia_face.weights import WeightSpec


def _spec(tmp_path, *contents):
    urls = []
    for i, data in enumerate(contents):
        p = tmp_path / f"src{i}.bin"
        p.write_bytes(data)
        urls.append(p.as_uri())
    return WeightSpec("model.onnx", tuple(urls), min_bytes=100, license="MIT")


def test_skips_lfs_pointer_and_uses_next_mirror(tmp_path, monkeypatch):
    lfs = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n" + b" " * 100
    monkeypatch.setitem(weights.MODELS, "m", _spec(tmp_path, lfs, b"\x08\x01" + b"x" * 500))
    path = weights.fetch("m", tmp_path / "cache")
    assert path.read_bytes().startswith(b"\x08\x01")


def test_cached_file_is_reused(tmp_path, monkeypatch):
    spec = _spec(tmp_path, b"\x08\x01" + b"x" * 500)
    monkeypatch.setitem(weights.MODELS, "m", spec)
    first = weights.fetch("m", tmp_path / "cache")
    (tmp_path / "src0.bin").unlink()  # source gone: must not be needed again
    assert weights.fetch("m", tmp_path / "cache") == first


def test_clear_error_when_every_mirror_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(weights.MODELS, "m", _spec(tmp_path, b"<!DOCTYPE html>" + b" " * 200, b"tiny"))
    with pytest.raises(RuntimeError, match="Download it manually"):
        weights.fetch("m", tmp_path / "cache")
    assert not list((tmp_path / "cache").glob(".download-*"))  # no partial files left behind
