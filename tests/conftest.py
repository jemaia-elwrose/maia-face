"""Stand-in models: a 'face' is a coloured square; its colour is its identity."""

import cv2
import numpy as np
import pytest

from maia_face.align import ARCFACE_TEMPLATE
from maia_face.backends.base import l2_normalize
from maia_face.types import Face

COLORS = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}


def scene(*faces, size=(240, 320)):
    """faces: (color_name, x, y, side). Returns a BGR image."""
    img = np.zeros((*size, 3), np.uint8)
    for color, x, y, side in faces:
        cv2.rectangle(img, (x, y), (x + side - 1, y + side - 1), COLORS[color], -1)
    return img


class FakeDetector:
    def detect(self, image):
        mask = (image.max(axis=2) > 0).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        faces = []
        for x, y, w, h, _ in stats[1:n]:
            lm = ARCFACE_TEMPLATE / 112.0 * np.array([w, h], np.float32) + np.array([x, y], np.float32)
            faces.append(Face(box=np.array([x, y, x + w, y + h], np.float32), score=0.99, landmarks=lm))
        return sorted(faces, key=lambda f: f.box[0])


class FakeEmbedder:
    name, dim, threshold = "fake-color", 4, 0.9

    def __init__(self):
        self.calls = []

    def embed_aligned(self, crops):
        self.calls.append(len(crops))
        means = np.array([c.reshape(-1, 3).mean(axis=0) for c in crops], np.float32)
        return l2_normalize(np.hstack([means, np.full((len(crops), 1), 1.0, np.float32)]))


@pytest.fixture
def fake_backend(monkeypatch):
    import maia_face.backends as backends

    det, emb = FakeDetector(), FakeEmbedder()
    monkeypatch.setattr(backends, "build", lambda name="opencv", **kw: (det, emb))
    return det, emb
