"""The InsightFace backend against stand-ins shaped like insightface 2.x objects."""

import sys
import types

import numpy as np

from maia_face.backends.base import embed_faces


def test_insightface_backend_glue(monkeypatch):
    calls = {}

    class Det:
        def detect(self, img, input_size=None, max_num=0, metric="default"):
            det = np.array([[10, 10, 60, 70, 0.9], [100, 20, 104, 24, 0.8]], np.float32)  # 2nd too small
            kps = np.tile(np.array([[20, 30], [50, 30], [35, 45], [22, 58], [48, 58]], np.float32), (2, 1, 1))
            return det, kps

    class Rec:
        def get_feat(self, imgs):
            calls["feat_shapes"] = [i.shape for i in imgs]
            return np.ones((len(imgs), 512), np.float32) * 3

    class FaceAnalysis:
        def __init__(self, name, root="~/.insightface", allowed_modules=None, **kw):
            calls["init"] = (name, allowed_modules, kw["providers"])
            self.det_model, self.models = Det(), {"detection": Det(), "recognition": Rec()}

        def prepare(self, ctx_id=0, det_thresh=0.5, det_size=None):
            calls["prepare"] = (ctx_id, det_thresh, det_size)

    ort = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "insightface", types.ModuleType("insightface"))
    monkeypatch.setitem(sys.modules, "insightface.app", types.SimpleNamespace(FaceAnalysis=FaceAnalysis))

    from maia_face.backends.insightface import load

    detector, embedder = load(pack="buffalo_l", score_threshold=0.45, min_size=10)
    assert calls["init"] == ("buffalo_l", ["detection", "recognition"], ["CPUExecutionProvider"])
    assert calls["prepare"] == (-1, 0.45, (640, 640))
    img = np.zeros((120, 160, 3), np.uint8)
    faces = detector.detect(img)
    assert len(faces) == 1 and faces[0].landmarks.shape == (5, 2)
    emb = embed_faces(embedder, img, faces)
    assert calls["feat_shapes"] == [(112, 112, 3)]
    assert emb.shape == (1, 512) and np.isclose(np.linalg.norm(emb[0]), 1.0)
    assert embedder.name == "insightface-buffalo_l"
