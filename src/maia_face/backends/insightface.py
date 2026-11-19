"""SCRFD detector + ArcFace recognizer from InsightFace.

Stronger than the OpenCV backend on hard faces, but InsightFace's pretrained
weights are licensed for NON-COMMERCIAL RESEARCH ONLY. The library (MIT) is
used to run them; obtain a commercial license from InsightFace before using
them in a product.
"""

from __future__ import annotations

import logging

import numpy as np

from ..types import Face
from .base import l2_normalize

log = logging.getLogger("maia_face.insightface")


class SCRFDDetector:
    def __init__(self, model, min_size: int = 0) -> None:
        self.model, self.min_size = model, min_size  # score threshold is set in FaceAnalysis.prepare()

    def detect(self, image: np.ndarray) -> list[Face]:
        boxes, kpss = self.model.detect(image, max_num=0)
        faces = []
        for box, kps in zip(boxes, kpss if kpss is not None else [None] * len(boxes), strict=True):
            if kps is None or min(box[2] - box[0], box[3] - box[1]) < self.min_size:
                continue
            faces.append(
                Face(
                    box=np.asarray(box[:4], np.float32),
                    score=float(box[4]),
                    landmarks=np.asarray(kps, np.float32).reshape(5, 2),
                )
            )
        return faces


class ArcFaceEmbedder:
    dim = 512
    threshold = 0.35

    def __init__(self, model, name: str) -> None:
        self.model, self.name = model, name

    def embed_aligned(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        return l2_normalize(self.model.get_feat(crops))


def load(
    pack: str = "buffalo_l",
    root: str | None = None,
    device: str = "auto",
    det_size: int = 640,
    score_threshold: float = 0.5,
    min_size: int = 0,
    **_,
):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    available = ort.get_available_providers()
    use_gpu = device != "cpu" and "CUDAExecutionProvider" in available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    log.warning("InsightFace '%s' weights are licensed for non-commercial research only", pack)
    kwargs = {"root": root} if root else {}
    app = FaceAnalysis(name=pack, providers=providers, allowed_modules=["detection", "recognition"], **kwargs)
    app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=score_threshold, det_size=(det_size, det_size))
    detector = SCRFDDetector(app.det_model, min_size)
    return detector, ArcFaceEmbedder(app.models["recognition"], name=f"insightface-{pack}")
