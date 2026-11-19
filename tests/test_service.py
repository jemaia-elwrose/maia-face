import json
import socket
import threading
import time

import numpy as np
import pytest
from conftest import scene
from maia_infer import Broker
from maia_infer.protocol import Request

from maia_face import Gallery
from maia_face.client import FaceClient
from maia_face.workers import AttributesWorker, FaceWorker


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_worker_batches_all_faces_into_one_recognizer_call(fake_backend):
    _, emb = fake_backend
    w = FaceWorker()
    w.setup()
    out = w.predict_batch(
        [
            Request(arrays=[scene(("red", 10, 10, 40), ("blue", 100, 10, 40))]),
            Request(arrays=[scene(("green", 50, 50, 60))]),
            Request(arrays=[scene()]),  # no faces
        ]
    )
    assert [o.arrays[0].shape for o in out] == [(2, 15), (1, 15), (0, 15)]
    assert [o.arrays[1].shape for o in out] == [(2, 4), (1, 4), (0, 4)]
    assert emb.calls == [3]  # one call for the whole batch
    assert out[0].meta == {"space": "fake-color", "dim": 4, "threshold": 0.9}


def test_worker_detection_only(fake_backend):
    w = FaceWorker()
    w.setup()
    (out,) = w.predict_batch([Request(arrays=[scene(("red", 10, 10, 40))], meta={"embed": False})])
    assert out.arrays[0].shape == (1, 15) and out.arrays[1].shape == (0, 4)


def test_end_to_end_identify_over_maia_infer(fake_backend, tmp_path):
    front, back = f"tcp://127.0.0.1:{_port()}", f"tcp://127.0.0.1:{_port()}"
    broker = Broker(front, back)
    worker = FaceWorker(broker=back, heartbeat_interval=0.2)
    threading.Thread(target=broker.run, daemon=True).start()
    threading.Thread(target=worker.run, daemon=True).start()
    try:
        end = time.time() + 5
        while time.time() < end and "face" not in broker.stats()["services"]:
            time.sleep(0.05)
        fc = FaceClient(front)
        with Gallery(tmp_path / "g.db") as g:
            (alice,) = fc.detect(scene(("red", 20, 20, 50)))
            g.enroll("alice", alice.embedding, fc.space)
            results = fc.identify(scene(("blue", 10, 10, 40), ("red", 150, 60, 70)), g)
        assert [m.name for _, m in results] == [None, "alice"]
        assert results[1][0].box.tolist() == [150, 60, 220, 130]
    finally:
        worker.stop()
        broker.stop()
        time.sleep(0.3)


class _FakeEstimator:
    def __init__(self):
        self.pairs = None

    def estimate(self, pairs):
        from maia_face.backends.mivolo import Attributes

        self.pairs = pairs
        return [Attributes(30.0 + i, "female", 0.9) for i in range(len(pairs))]


def test_attributes_worker_handles_missing_crops():
    w = AttributesWorker()
    w.estimator = _FakeEstimator()
    face, empty = np.ones((40, 40, 3), np.uint8), np.zeros((0, 0, 3), np.uint8)
    out = w.predict_batch([Request(arrays=[face, empty]), Request(arrays=[empty, face]), Request(arrays=[face])])
    assert [(p[0] is None, p[1] is None) for p in w.estimator.pairs] == [(False, True), (True, False), (False, True)]
    assert out[1] == {"age": 31.0, "gender": "female", "gender_conf": 0.9}


def test_mivolo_decoding_matches_reference_formula():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from maia_face.backends.mivolo import MiVOLOEstimator

    est = MiVOLOEstimator.__new__(MiVOLOEstimator)  # skip loading the real model
    meta = SimpleNamespace(min_age=1.0, max_age=95.0, avg_age=42.0)
    logits = torch.tensor([[2.0, -1.0, 0.1], [-3.0, 1.0, -0.2]])
    est.model = SimpleNamespace(
        meta=meta, inference=lambda x: logits, input_size=224, data_config={"mean": 0, "std": 1}, device="cpu"
    )
    est._prepare_images = lambda crops, *a, **k: torch.zeros(len(crops), 3, 8, 8)
    face = np.zeros((10, 10, 3), np.uint8)
    a, b = est.estimate([(face, face), (None, face)])
    assert a.gender == "male" and b.gender == "female"
    assert a.age == pytest.approx(0.1 * 94 + 42) and b.age == pytest.approx(-0.2 * 94 + 42)
    assert a.gender_conf == pytest.approx(torch.softmax(torch.tensor([2.0, -1.0]), 0)[0].item())


def test_cli_enroll_identify_verify_list_remove(fake_backend, tmp_path, capsys):
    import cv2

    from maia_face.cli import main

    paths = {}
    for name, faces in {
        "alice1": [("red", 30, 30, 60)],
        "alice2": [("red", 100, 80, 80)],
        "group": [("blue", 10, 10, 40), ("red", 150, 60, 70)],
        "two": [("red", 10, 10, 40), ("green", 150, 60, 70)],
    }.items():
        paths[name] = str(tmp_path / f"{name}.png")
        cv2.imwrite(paths[name], scene(*faces))
    db = ["--gallery", str(tmp_path / "g.db")]

    main([*db, "enroll", "alice", paths["alice1"], paths["alice2"], paths["two"]])
    assert "2 sample(s)" in capsys.readouterr().out  # 'two' skipped: several faces

    main([*db, "identify", paths["group"], "--save", str(tmp_path / "out.png")])
    (result,) = json.loads(capsys.readouterr().out)
    assert [f["name"] for f in result["faces"]] == [None, "alice"]
    assert (tmp_path / "out.png").exists()

    main([*db, "verify", paths["alice1"], paths["alice2"]])
    assert json.loads(capsys.readouterr().out)["same_person"] is True

    main([*db, "list"])
    assert json.loads(capsys.readouterr().out)[0]["samples"] == 2
    main([*db, "remove", "alice"])
    assert capsys.readouterr().out.strip() == "removed"


def test_mivolo_missing_source_has_clear_message(tmp_path):
    from maia_face.backends.mivolo import MiVOLOEstimator

    with pytest.raises(RuntimeError, match="maia-face setup-mivolo"):
        MiVOLOEstimator("x.pth", device="cpu", source=str(tmp_path))


def test_mivolo_face_only_checkpoint_needs_faces():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from maia_face.backends.mivolo import MiVOLOEstimator

    est = MiVOLOEstimator.__new__(MiVOLOEstimator)
    meta = SimpleNamespace(min_age=1.0, max_age=95.0, avg_age=42.0, with_persons_model=False)
    seen = {}

    def inference(x):
        seen["channels"] = x.shape[1]
        return torch.zeros(x.shape[0], 3)

    est.model = SimpleNamespace(
        meta=meta, inference=inference, input_size=8, data_config={"mean": 0, "std": 1}, device="cpu"
    )
    est._prepare_images = lambda crops, *a, **k: torch.zeros(len(crops), 3, 8, 8)
    face = np.zeros((10, 10, 3), np.uint8)
    assert len(est.estimate([(face, face)])) == 1 and seen["channels"] == 3  # body ignored
    with pytest.raises(ValueError, match="face-only"):
        est.estimate([(None, face)])
