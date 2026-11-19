"""Check the real models end to end with one photo of yourself.

    python examples/smoke_test.py me.jpg [--backend insightface]

Downloads the weights on first run, then checks that the face is found, that the
same face is recognised after a mirror flip, a shift and a resize (similarity
above the threshold), and that a heavily blurred blob is not.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from maia_face.cli import LocalEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--backend", default="opencv")
    args = ap.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"cannot read {args.image}")

    t0 = time.perf_counter()
    engine = LocalEngine(args.backend)
    print(f"models ready in {time.perf_counter() - t0:.1f}s: {engine.space}, threshold {engine.threshold}")

    t0 = time.perf_counter()
    faces = engine.faces(image)
    print(f"{len(faces)} face(s) in {(time.perf_counter() - t0) * 1000:.0f} ms")
    if not faces:
        sys.exit("FAIL: no face found")
    ref = max(faces, key=lambda f: f.width * f.height)

    def best_similarity(img):
        found = engine.faces(img)
        return max((float(ref.embedding @ f.embedding) for f in found), default=None)

    h, w = image.shape[:2]
    checks = {
        "mirrored": cv2.flip(image, 1),
        "shifted": cv2.warpAffine(image, np.float32([[1, 0, w * 0.1], [0, 1, h * 0.05]]), (w, h)),
        "resized 0.7x": cv2.resize(image, None, fx=0.7, fy=0.7),
    }
    ok = True
    for name, img in checks.items():
        sim = best_similarity(img)
        passed = sim is not None and sim >= engine.threshold
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'} {name:<13} similarity {sim if sim is None else round(sim, 3)}")
    noise = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, image.shape, np.uint8), (51, 51), 0)
    print(f"  {'PASS' if not engine.faces(noise) else 'WARN'} no face in blurred noise")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
