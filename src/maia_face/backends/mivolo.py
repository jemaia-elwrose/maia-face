"""Age and gender with MiVOLO (face + body transformer).

MiVOLO's own package cannot be installed with current tooling (its setup.py
imports ``pkg_resources`` and pins ultralytics 8.1.0), so maia-face loads its
source from a pinned checkout instead: run ``maia-face setup-mivolo`` once.
The model code needs timm 0.8.13.dev0 exactly (later timm versions changed the
VOLO internals it builds on); ``pip install "maia-face[mivolo]"`` pins it.

Licenses: MiVOLO code Apache-2.0; check your checkpoint (published v2 weights
are CC BY-NC 4.0, non-commercial). A missing face or body crop is passed as an
empty input, as in MiVOLO's own pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import weights

MIVOLO_REPO = "https://github.com/WildChlamydia/MiVOLO.git"
MIVOLO_COMMIT = "37475e3"  # tested with timm 0.8.13.dev0, torch 2.10-2.15


@dataclass
class Attributes:
    age: float
    gender: str  # "male" | "female"
    gender_conf: float


def source_dir() -> Path:
    return Path(os.environ.get("MAIA_MIVOLO_SRC", weights.home() / "MiVOLO"))


def setup_source(directory: Path | None = None) -> Path:
    """Fetch MiVOLO's source at the pinned commit (needs git)."""
    directory = Path(directory or source_dir())
    if (directory / "mivolo" / "model" / "mi_volo.py").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", str(directory), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("remote", "add", "origin", MIVOLO_REPO)
    try:
        run("fetch", "-q", "--depth", "1", "origin", MIVOLO_COMMIT)
        run("checkout", "-q", "FETCH_HEAD")
    except subprocess.CalledProcessError:  # short SHAs can't be fetched directly from some servers
        run("fetch", "-q", "origin")
        run("checkout", "-q", MIVOLO_COMMIT)
    return directory


def _import_mivolo(src: Path | None):
    src = Path(src or source_dir())
    if not (src / "mivolo").is_dir():
        raise RuntimeError(f"MiVOLO source not found in {src}. Run `maia-face setup-mivolo` once.")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mivolo.data.misc import prepare_classification_images
    from mivolo.model.mi_volo import MiVOLO

    return MiVOLO, prepare_classification_images


class MiVOLOEstimator:
    def __init__(self, checkpoint: str, device: str = "cuda", half: bool = False, source: str | None = None) -> None:
        mivolo_cls, self._prepare_images = _import_mivolo(source)
        self.model = mivolo_cls(checkpoint, device=device, half=half)

    def _prepare(self, crops: list[np.ndarray | None]):
        m = self.model
        return self._prepare_images(crops, m.input_size, m.data_config["mean"], m.data_config["std"], device=m.device)

    def estimate(self, pairs: list[tuple[np.ndarray | None, np.ndarray | None]]) -> list[Attributes]:
        import torch

        if not pairs:
            return []
        uses_bodies = getattr(self.model.meta, "with_persons_model", True)
        for face, body in pairs:
            if face is None and (body is None or not uses_bodies):
                raise ValueError("this request has no usable crop (face-only checkpoints need a face crop)")
        faces = [f for f, _ in pairs]
        with torch.inference_mode():
            x = self._prepare(faces)
            if uses_bodies:
                x = torch.cat((x, self._prepare([b for _, b in pairs])), dim=1)
            out = self.model.inference(x).float()
        probs = out[:, :2].softmax(-1)
        meta = self.model.meta
        ages = out[:, 2] * (meta.max_age - meta.min_age) + meta.avg_age
        results = []
        for i in range(len(pairs)):
            idx = int(probs[i].argmax())
            results.append(Attributes(float(ages[i]), "male" if idx == 0 else "female", float(probs[i, idx])))
        return results
