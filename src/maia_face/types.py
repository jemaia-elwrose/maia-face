"""Core data types and the wire format shared by workers and clients."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Columns of the faces array sent over maia-infer: box(4) + score(1) + 5 landmarks (x, y).
FACE_COLUMNS = 15


@dataclass
class Face:
    box: np.ndarray  # (4,) x1, y1, x2, y2 in pixels
    score: float
    landmarks: np.ndarray  # (5, 2): eye, eye, nose, mouth corner, mouth corner (image left first)
    embedding: np.ndarray | None = None  # (D,) L2-normalised
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


def pack_faces(faces: list[Face]) -> np.ndarray:
    out = np.zeros((len(faces), FACE_COLUMNS), np.float32)
    for i, f in enumerate(faces):
        out[i, :4] = f.box
        out[i, 4] = f.score
        out[i, 5:] = np.asarray(f.landmarks, np.float32).reshape(-1)
    return out


def unpack_faces(arr: np.ndarray, embeddings: np.ndarray | None = None) -> list[Face]:
    faces = []
    for i, row in enumerate(np.asarray(arr, np.float32).reshape(-1, FACE_COLUMNS)):
        emb = None if embeddings is None or len(embeddings) == 0 else np.asarray(embeddings[i], np.float32)
        faces.append(
            Face(box=row[:4].copy(), score=float(row[4]), landmarks=row[5:].reshape(5, 2).copy(), embedding=emb)
        )
    return faces
