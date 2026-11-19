"""Age and gender for every person in an image: YOLO finds people, the face
service finds faces, faces are matched to people, MiVOLO estimates attributes.

    maia-infer broker &
    maia-infer worker yolo &
    maia-infer worker face &
    maia-infer worker face-attr --opt checkpoint=model_imdb_cross_person.pth.tar &
    python examples/age_gender.py friends.jpeg --save friends_age_gender.jpg

People whose face isn't visible still get an estimate from the body alone
(less accurate, marked "body only").
"""

import argparse

import cv2
import numpy as np
from maia_infer import Client

from maia_face import associate
from maia_face.client import FaceClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--endpoint", default="tcp://localhost:5555")
    ap.add_argument("--save", help="write an annotated copy here")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")
    client = Client(args.endpoint, timeout_ms=10000)  # first face-attr call warms up the GPU
    faces = FaceClient(client)

    # 1. People (YOLO): rows of x1, y1, x2, y2, conf, class_id
    persons = client.infer("yolo", frame, meta={"conf": 0.4, "classes": ["person"]}).arrays[0]
    # 2. Faces: list of Face objects (box, score, landmarks); no embeddings needed here
    face_list = faces.detect(frame, embed=False)
    face_boxes = np.array([f.box for f in face_list], np.float32).reshape(-1, 4)
    # 3. Which face belongs to which person: owner[i] = person index of face i, or -1
    owner = associate(face_boxes, persons)
    face_of_person = {p: i for i, p in enumerate(owner) if p >= 0}
    print(f"{len(persons)} people, {len(face_list)} faces, {len(face_of_person)} matched")

    # 4. Attributes per person: face + body when we have both, body alone otherwise
    for p, person in enumerate(persons):
        face_box = face_boxes[face_of_person[p]] if p in face_of_person else None
        attrs = faces.attributes(frame, face_box=face_box, person_box=person)
        source = "face+body" if face_box is not None else "body only"
        print(f"person {p}: age {attrs.age:.0f}, {attrs.gender} ({attrs.gender_conf:.2f}) [{source}]")

        x1, y1, x2, y2 = (int(v) for v in person[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2)
        if face_box is not None:
            fx1, fy1, fx2, fy2 = (int(v) for v in face_box)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
        label = f"{attrs.gender[0].upper()} {attrs.age:.0f}" + ("" if face_box is not None else " (body)")
        cv2.putText(frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 120), 2)

    # Faces with no person box (e.g. YOLO missed a partly hidden person): face alone
    for i, p in enumerate(owner):
        if p < 0:
            attrs = faces.attributes(frame, face_box=face_boxes[i])
            print(f"face {i} (no person box): age {attrs.age:.0f}, {attrs.gender} ({attrs.gender_conf:.2f})")

    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
