import math

import cv2
import numpy as np
import pytest

from maia_face import AttributeSmoother, Gallery, IdentityVoter, associate, crop, pack_faces, unpack_faces
from maia_face.align import ARCFACE_TEMPLATE, align_face
from maia_face.types import Face


def test_alignment_maps_landmarks_onto_template():
    # Place the template rotated by 20 deg, scaled x2.5 and shifted, draw dots there, align back.
    a, s, t = math.radians(20), 2.5, np.array([140.0, 90.0])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    landmarks = (ARCFACE_TEMPLATE @ rot.T) * s + t
    img = np.zeros((480, 640, 3), np.uint8)
    for x, y in landmarks:
        cv2.circle(img, (round(x), round(y)), 6, (255, 255, 255), -1)
    out = align_face(img, landmarks)
    assert out.shape == (112, 112, 3)
    for x, y in ARCFACE_TEMPLATE:
        assert out[round(y), round(x)].max() > 200  # each dot lands on its template position


def test_pack_unpack_roundtrip():
    f = Face(
        box=np.array([1, 2, 30, 40], np.float32), score=0.9, landmarks=np.arange(10, dtype=np.float32).reshape(5, 2)
    )
    emb = np.ones((1, 4), np.float32)
    (g,) = unpack_faces(pack_faces([f]), emb)
    np.testing.assert_array_equal(g.box, f.box)
    np.testing.assert_array_equal(g.landmarks, f.landmarks)
    assert g.embedding.shape == (4,)


def _unit(*v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_gallery_identify_threshold_and_best_sample(tmp_path):
    with Gallery(tmp_path / "g.db") as g:
        g.enroll("alice", np.stack([_unit(1, 0, 0), _unit(1, 0.2, 0)]), "s")
        g.enroll("bob", _unit(0, 1, 0), "s")
        alice, bob, nobody = g.identify(np.stack([_unit(1, 0.21, 0), _unit(0.1, 1, 0), _unit(0, 0, 1)]), "s", 0.8)
        assert alice.name == "alice" and alice.score > 0.99  # best of alice's two samples
        assert bob.name == "bob"
        assert nobody.name is None and nobody.candidate is not None
        assert [i["name"] for i in g.identities()] == ["alice", "bob"]
        assert g.identities()[0]["samples"] == 2


def test_gallery_never_mixes_embedding_spaces(tmp_path):
    with Gallery(tmp_path / "g.db") as g:
        g.enroll("alice", _unit(1, 0, 0), "sface")
        assert g.identify(_unit(1, 0, 0), "arcface", 0.5)[0].name is None
        with pytest.raises(ValueError, match="dim"):
            g.identify(_unit(1, 0), "sface", 0.5)


def test_gallery_remove_and_reenroll(tmp_path):
    with Gallery(tmp_path / "g.db") as g:
        g.enroll("alice", _unit(1, 0, 0), "s")
        assert g.remove("alice") and not g.remove("alice")
        assert g.identify(_unit(1, 0, 0), "s", 0.5)[0].name is None
        g.enroll("alice", _unit(0, 1, 0), "s")
        assert g.identify(_unit(0, 1, 0), "s", 0.5)[0].name == "alice"


def test_gallery_sees_writes_from_another_connection(tmp_path):
    path = tmp_path / "g.db"
    with Gallery(path) as reader, Gallery(path) as writer:
        assert reader.identify(_unit(1, 0, 0), "s", 0.5)[0].name is None  # caches an empty matrix
        writer.enroll("carol", _unit(1, 0, 0), "s")
        assert reader.identify(_unit(1, 0, 0), "s", 0.5)[0].name == "carol"  # reloaded via data_version


def test_gallery_shared_across_processes(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "g.db"
    code = (
        "import numpy as np; from maia_face import Gallery; "
        f"Gallery(r'{path}').enroll('dave', np.array([[0, 0, 1]], np.float32), 's')"
    )
    with Gallery(path) as g:
        g.identify(_unit(0, 0, 1), "s", 0.5)
        subprocess.run([sys.executable, "-c", code], check=True, env={**__import__("os").environ})
        assert g.identify(_unit(0, 0, 1), "s", 0.5)[0].name == "dave"


def test_identity_voter_needs_consistent_evidence():
    v = IdentityVoter(min_votes=3, min_share=0.6)
    assert v.update(1, "alice", 0.6, frame=0).name is None  # one vote is not enough
    v.update(1, "alice", 0.7, frame=1)
    assert v.update(1, "alice", 0.65, frame=2).name == "alice"
    for f in range(3, 9):  # now mostly someone else
        d = v.update(1, "bob", 0.8, frame=f)
    assert d.name == "bob"


def test_identity_voter_unknowns_block_weak_matches():
    v = IdentityVoter(min_votes=2, min_share=0.6)
    for f in range(6):
        v.update(7, None, 0.0, frame=f)
    v.update(7, "alice", 0.4, frame=6)
    assert v.update(7, "alice", 0.4, frame=7).name is None


def test_identity_voter_prunes_stale_tracks():
    v = IdentityVoter(max_age=10)
    v.update(1, "a", 0.9, frame=0)
    v.update(2, "b", 0.9, frame=15)
    assert v.prune(frame=20) == [1]
    assert v.decision(1).votes == 0 and v.decision(2).votes == 1


def test_attribute_smoother():
    s = AttributeSmoother(window=5)
    for age, g, c in [(30, "male", 0.9), (80, "female", 0.55), (32, "male", 0.8), (31, "male", 0.7)]:
        out = s.update(3, age, g, c)
    assert out[0] == pytest.approx(31.5)  # median ignores the 80 outlier
    assert out[1] == "male"


def test_associate_faces_to_people():
    persons = np.array([[0, 0, 100, 300], [200, 0, 300, 300]], np.float32)
    faces = np.array([[230, 10, 270, 50], [30, 20, 70, 60], [120, 10, 150, 40], [20, 250, 60, 290]], np.float32)
    assert associate(faces, persons) == [1, 0, -1, -1]  # 3rd outside everyone; 4th at the feet


def test_associate_one_face_per_person():
    persons = np.array([[0, 0, 100, 300]], np.float32)
    faces = np.array([[30, 20, 70, 60], [35, 25, 75, 65]], np.float32)
    assert sorted(associate(faces, persons)) == [-1, 0]


def test_crop_clamps_to_image():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop(img, [-10, -10, 50, 50]).shape == (50, 50, 3)
    assert crop(img, [10, 10, 90, 90], pad=0.5).shape == (100, 100, 3)


def test_quiet_opencv_restores_log_level():
    import cv2

    from maia_face.backends.opencv_zoo import _quiet_opencv

    log = cv2.utils.logging
    log.setLogLevel(log.LOG_LEVEL_INFO)
    with _quiet_opencv():
        assert log.getLogLevel() == log.LOG_LEVEL_ERROR
    assert log.getLogLevel() == log.LOG_LEVEL_INFO
    log.setLogLevel(log.LOG_LEVEL_WARNING)
