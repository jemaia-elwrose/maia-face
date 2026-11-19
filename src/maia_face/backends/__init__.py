"""Face model backends.

=========== ======================== =================================== ======================
name        detector / recognizer    weights license                     install
=========== ======================== =================================== ======================
opencv      YuNet / SFace (128-d)    MIT / Apache-2.0 (commercial OK)    included
insightface SCRFD / ArcFace (512-d)  non-commercial research only        ``maia-face[insightface]``
=========== ======================== =================================== ======================

Attributes (age, gender): ``mivolo`` (weights CC BY-NC 4.0 / see upstream), ``maia-face[mivolo]``.
"""

from __future__ import annotations

from .base import Detector, Embedder


def build(name: str = "opencv", **options) -> tuple[Detector, Embedder]:
    name = name.lower()
    if name == "opencv":
        from .opencv_zoo import SFaceEmbedder, YuNetDetector

        return YuNetDetector(**options.get("detector", {})), SFaceEmbedder(**options.get("embedder", {}))
    if name == "insightface":
        from .insightface import load

        return load(**options)
    raise ValueError(f"unknown backend '{name}' (opencv | insightface)")
