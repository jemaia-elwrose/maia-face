"""YuNet detector (MIT) + SFace recognizer (Apache-2.0) from the OpenCV Zoo.

Pure OpenCV DNN: no PyTorch, runs well on CPU, licensed for commercial use.
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2
import numpy as np

from .. import weights
from ..types import Face
from .base import l2_normalize


@contextmanager
def _quiet_opencv():
    """OpenCV 5's new DNN engine warns "Targets are not supported by the new graph
    engine" when these models load (harmless: they run on the default CPU target).
    Silence only that moment and restore the caller's log level afterwards."""
    log = getattr(getattr(cv2, "utils", None), "logging", None)
    if log is None:
        yield
        return
    previous = log.getLogLevel()
    log.setLogLevel(log.LOG_LEVEL_ERROR)
    try:
        yield
    finally:
        log.setLogLevel(previous)


class YuNetDetector:
    def __init__(
        self, model: str | None = None, score_threshold: float = 0.6, nms_threshold: float = 0.3, min_size: int = 0
    ) -> None:
        path = model or str(weights.fetch("yunet"))
        self.min_size = min_size
        with _quiet_opencv():
            self.net = cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold, nms_threshold, 5000)

    def detect(self, image: np.ndarray) -> list[Face]:
        h, w = image.shape[:2]
        self.net.setInputSize((w, h))
        _, rows = self.net.detect(image)
        faces = []
        for r in rows if rows is not None else []:
            x, y, bw, bh = (float(v) for v in r[:4])
            if min(bw, bh) < self.min_size:
                continue
            box = np.array([max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)], np.float32)
            faces.append(Face(box=box, score=float(r[14]), landmarks=np.asarray(r[4:14], np.float32).reshape(5, 2)))
        return faces


class SFaceEmbedder:
    name = "sface-2021dec"
    dim = 128
    threshold = 0.363  # cosine; OpenCV's recommended value for SFace

    def __init__(self, model: str | None = None) -> None:
        path = model or str(weights.fetch("sface"))
        with _quiet_opencv():
            self.net = cv2.FaceRecognizerSF.create(path, "")

    def embed_aligned(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        return l2_normalize(np.vstack([self.net.feature(c).reshape(1, -1) for c in crops]))
