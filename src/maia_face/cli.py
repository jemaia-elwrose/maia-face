"""maia-face enroll | identify | verify | list | remove

Runs the models in-process by default (no broker needed); pass ``--endpoint``
to use a running ``face`` service instead.

    maia-face enroll "Alice" alice1.jpg alice2.jpg
    maia-face identify group.jpg --save group_annotated.jpg
    maia-face verify a.jpg b.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np

from .gallery import Gallery
from .types import Face


class LocalEngine:
    def __init__(self, backend: str) -> None:
        from .backends import build
        from .backends.base import embed_faces

        self.detector, self.embedder = build(backend)
        self._embed = embed_faces
        self.space, self.threshold = self.embedder.name, self.embedder.threshold

    def faces(self, image: np.ndarray) -> list[Face]:
        faces = self.detector.detect(image)
        for f, e in zip(faces, self._embed(self.embedder, image, faces), strict=True):
            f.embedding = e
        return faces


class RemoteEngine:
    def __init__(self, endpoint: str) -> None:
        from .client import FaceClient

        self.fc = FaceClient(endpoint)

    def faces(self, image: np.ndarray) -> list[Face]:
        faces = self.fc.detect(image)
        self.space, self.threshold = self.fc.space, self.fc.threshold
        return faces


def _read(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return image


def _largest(faces: list[Face]) -> Face:
    return max(faces, key=lambda f: f.width * f.height)


def _draw(image: np.ndarray, faces: list[Face], labels: list[str]) -> np.ndarray:
    out = image.copy()
    for f, label in zip(faces, labels, strict=True):
        x1, y1, x2, y2 = (int(v) for v in f.box)
        color = (80, 200, 120) if not label.startswith("unknown") else (60, 60, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="maia-face", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gallery", default=os.getenv("MAIA_FACE_GALLERY", "gallery.db"))
    ap.add_argument("--backend", default=os.getenv("MAIA_FACE_BACKEND", "opencv"), help="opencv | insightface")
    ap.add_argument("--endpoint", help="use a running 'face' service, e.g. tcp://localhost:5555")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="add face images for a person")
    e.add_argument("name")
    e.add_argument("images", nargs="+")
    e.add_argument("--largest", action="store_true", help="if an image has several faces, use the largest")

    i = sub.add_parser("identify", help="find enrolled people in images")
    i.add_argument("images", nargs="+")
    i.add_argument("--threshold", type=float)
    i.add_argument("--save", help="write an annotated copy (single image) to this path")

    v = sub.add_parser("verify", help="are the largest faces in two images the same person?")
    v.add_argument("image_a")
    v.add_argument("image_b")
    v.add_argument("--threshold", type=float)

    sub.add_parser("list", help="list enrolled identities")
    sm = sub.add_parser("setup-mivolo", help="fetch MiVOLO's source (pinned commit) for the face-attr worker")
    sm.add_argument("--dir", help="where to put it (default: $MAIA_MIVOLO_SRC or ~/.cache/maia-face/MiVOLO)")
    r = sub.add_parser("remove", help="delete an identity")
    r.add_argument("name")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "setup-mivolo":
        from .backends.mivolo import setup_source

        print(f"MiVOLO source ready in {setup_source(args.dir)}")
        return

    if args.cmd in ("list", "remove"):
        with Gallery(args.gallery) as g:
            if args.cmd == "list":
                print(json.dumps(g.identities(), indent=2))
            else:
                print("removed" if g.remove(args.name) else f"no identity named '{args.name}'")
        return

    engine = RemoteEngine(args.endpoint) if args.endpoint else LocalEngine(args.backend)

    if args.cmd == "verify":
        a, b = (engine.faces(_read(p)) for p in (args.image_a, args.image_b))
        if not a or not b:
            raise SystemExit("no face found in " + (args.image_a if not a else args.image_b))
        sim = float(_largest(a).embedding @ _largest(b).embedding)
        thr = args.threshold if args.threshold is not None else engine.threshold
        print(json.dumps({"similarity": round(sim, 4), "threshold": thr, "same_person": sim >= thr}))
        return

    with Gallery(args.gallery) as gallery:
        if args.cmd == "enroll":
            vectors = []
            for path in args.images:
                faces = engine.faces(_read(path))
                if not faces:
                    print(f"skip {path}: no face found", file=sys.stderr)
                    continue
                if len(faces) > 1 and not args.largest:
                    print(f"skip {path}: {len(faces)} faces (use --largest)", file=sys.stderr)
                    continue
                vectors.append(_largest(faces).embedding)
            if not vectors:
                raise SystemExit("nothing enrolled")
            gallery.enroll(args.name, np.stack(vectors), engine.space)
            print(f"enrolled '{args.name}' with {len(vectors)} sample(s) [{engine.space}]")
            return

        results = []
        for path in args.images:
            image = _read(path)
            faces = engine.faces(image)
            thr = args.threshold if args.threshold is not None else engine.threshold
            matches = gallery.identify(np.stack([f.embedding for f in faces]), engine.space, thr) if faces else []
            labels = [f"{m.name} {m.score:.2f}" if m.name else f"unknown {m.score:.2f}" for m in matches]
            results.append(
                {
                    "image": path,
                    "faces": [
                        {"box": [round(float(x), 1) for x in f.box], "name": m.name, "score": round(m.score, 4)}
                        for f, m in zip(faces, matches, strict=True)
                    ],
                }
            )
            if args.save and len(args.images) == 1:
                cv2.imwrite(args.save, _draw(image, faces, labels))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
