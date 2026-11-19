"""Download-once model weights, cached under ``$MAIA_FACE_HOME`` (default ``~/.cache/maia-face``).

Files are fetched from the upstream project (OpenCV Zoo), written atomically,
and sanity-checked: a Git LFS pointer or an HTML error page is never mistaken
for a model.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("maia_face.weights")


@dataclass(frozen=True)
class WeightSpec:
    filename: str
    urls: tuple[str, ...]
    min_bytes: int
    license: str


_ZOO = "https://huggingface.co/opencv/{repo}/resolve/main/{file}"
_LFS = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{repo}/{file}"


def _zoo(repo: str, file: str, min_bytes: int, license: str) -> WeightSpec:
    return WeightSpec(file, (_ZOO.format(repo=repo, file=file), _LFS.format(repo=repo, file=file)), min_bytes, license)


MODELS: dict[str, WeightSpec] = {
    "yunet": _zoo("face_detection_yunet", "face_detection_yunet_2023mar.onnx", 100_000, "MIT"),
    "sface": _zoo("face_recognition_sface", "face_recognition_sface_2021dec.onnx", 10_000_000, "Apache-2.0"),
}


def home() -> Path:
    return Path(os.environ.get("MAIA_FACE_HOME", Path.home() / ".cache" / "maia-face"))


def _looks_valid(path: Path, spec: WeightSpec) -> bool:
    if not path.is_file() or path.stat().st_size < spec.min_bytes:
        return False
    with open(path, "rb") as f:
        head = f.read(64)
    return not (
        head.startswith(b"version https://git-lfs") or head.lstrip().lower().startswith((b"<!doctype", b"<html"))
    )


def fetch(name: str, directory: Path | None = None) -> Path:
    """Return the local path of model ``name``, downloading it on first use."""
    spec = MODELS[name]
    directory = Path(directory or home())
    path = directory / spec.filename
    if _looks_valid(path, spec):
        return path
    directory.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in spec.urls:
        log.info("downloading %s (%s license) from %s", spec.filename, spec.license, url)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".download-")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
                shutil.copyfileobj(resp, out)
            if _looks_valid(Path(tmp), spec):
                os.replace(tmp, path)
                return path
            errors.append(f"{url}: not a valid model file")
        except OSError as exc:
            errors.append(f"{url}: {exc}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    raise RuntimeError(
        f"could not download {spec.filename}:\n  "
        + "\n  ".join(errors)
        + f"\nDownload it manually into {directory} (or set MAIA_FACE_HOME)."
    )
