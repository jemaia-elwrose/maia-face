"""maia-infer workers. After ``pip install maia-face`` they are available by name:

    maia-infer worker face                       # detect + embed (OpenCV backend)
    maia-infer worker face --opt backend=insightface
    maia-infer worker face-attr --opt checkpoint=/models/mivolo.pth.tar

``face`` reply: arrays = [faces (N, 15) float32, embeddings (N, D) float32],
meta = {"space", "dim", "threshold"}. Request meta ``{"embed": false}`` skips
recognition (detection only).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from maia_infer import Result, Worker
from maia_infer.models.detection import select_device

from .align import align_face
from .types import pack_faces

log = logging.getLogger("maia_face.workers")


class FaceWorker(Worker):
    service = "face"
    max_batch = 8
    max_wait_ms = 4

    def setup(self) -> None:
        from .backends import build

        t0 = time.perf_counter()
        name = self.options.get("backend", "opencv")
        score = float(self.options.get("score", 0.6 if name == "opencv" else 0.5))
        min_size = int(self.options.get("min_size", 0))
        if name == "opencv":
            opts = {"detector": {"score_threshold": score, "min_size": min_size}}
        else:
            opts = {
                "pack": self.options.get("pack", "buffalo_l"),
                "score_threshold": score,
                "min_size": min_size,
                "device": self.options.get("device", "auto"),
            }
        self.detector, self.embedder = build(name, **opts)
        log.info(
            "FaceWorker ready: %s, %d-d, threshold %.3f (%.1fs)",
            self.embedder.name,
            self.embedder.dim,
            self.embedder.threshold,
            time.perf_counter() - t0,
        )

    def predict_batch(self, requests):
        per_request, crops, owners = [], [], []
        for i, r in enumerate(requests):
            image = r.arrays[0]
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"expected an HxWx3 BGR image, got {image.shape}")
            faces = self.detector.detect(image)
            per_request.append(faces)
            if r.meta.get("embed", True):
                for f in faces:
                    crops.append(align_face(image, f.landmarks))
                    owners.append(i)
        # One recognizer call for every face in the batch.
        embeddings = self.embedder.embed_aligned(crops) if crops else np.zeros((0, self.embedder.dim), np.float32)
        by_request = [[] for _ in requests]
        for owner, emb in zip(owners, embeddings, strict=True):
            by_request[owner].append(emb)
        meta = {"space": self.embedder.name, "dim": self.embedder.dim, "threshold": self.embedder.threshold}
        return [
            Result(
                arrays=[pack_faces(faces), np.asarray(embs, np.float32).reshape(-1, self.embedder.dim)],
                meta=meta,
            )
            for faces, embs in zip(per_request, by_request, strict=True)
        ]


class AttributesWorker(Worker):
    """Age and gender with MiVOLO. Request: arrays = [face_crop, body_crop] (either may
    be an empty 0x0x3 array when missing). Reply meta: {"age", "gender", "gender_conf"}."""

    service = "face-attr"
    max_batch = 16
    max_wait_ms = 4

    def setup(self) -> None:
        from .backends.mivolo import MiVOLOEstimator

        checkpoint = self.options.get("checkpoint")
        if not checkpoint:
            raise ValueError("face-attr needs --opt checkpoint=/path/to/mivolo_checkpoint.pth.tar")
        device = select_device(self.options.get("device", "auto"))
        self.estimator = MiVOLOEstimator(
            checkpoint, device=device, half=self.options.get("half") == "true", source=self.options.get("mivolo_src")
        )
        log.info("AttributesWorker ready on %s", device)

    def predict_batch(self, requests):
        pairs = []
        for r in requests:
            face, body = (r.arrays + [None, None])[:2]
            pairs.append(
                (face if face is not None and face.size else None, body if body is not None and body.size else None)
            )
        return [
            {"age": a.age, "gender": a.gender, "gender_conf": a.gender_conf} for a in self.estimator.estimate(pairs)
        ]
