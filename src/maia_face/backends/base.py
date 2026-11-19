from __future__ import annotations

from typing import Protocol

import numpy as np

from ..align import align_face
from ..types import Face


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Face]:
        """Faces in a BGR image, with 5 landmarks each."""


class Embedder(Protocol):
    #: Identifies the embedding space: embeddings from different backends must never be compared.
    name: str
    dim: int
    #: Cosine similarity at or above which two faces are considered the same person.
    threshold: float

    def embed_aligned(self, crops: list[np.ndarray]) -> np.ndarray:
        """(N, dim) L2-normalised embeddings for aligned 112x112 BGR crops."""


def embed_faces(embedder: Embedder, image: np.ndarray, faces: list[Face]) -> np.ndarray:
    if not faces:
        return np.zeros((0, embedder.dim), np.float32)
    crops = [align_face(image, f.landmarks) for f in faces]
    return embedder.embed_aligned(crops)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
