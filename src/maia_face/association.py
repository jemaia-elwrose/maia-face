"""Match face boxes to person boxes (for face + body models such as MiVOLO)."""

from __future__ import annotations

import numpy as np


def associate(
    face_boxes: np.ndarray, person_boxes: np.ndarray, min_inside: float = 0.8, max_head_fraction: float = 0.5
) -> list[int]:
    """For each face, the index of the person it belongs to, or -1.

    A face belongs to a person when most of it lies inside the person box and
    its centre is in the top part of that box (where a head is). Each person
    gets at most one face; ties go to the tightest fit.
    """
    faces = np.asarray(face_boxes, np.float32).reshape(-1, 4)
    persons = np.asarray(person_boxes, np.float32).reshape(-1, 4)
    result = [-1] * len(faces)
    if not len(faces) or not len(persons):
        return result
    candidates = []
    for i, f in enumerate(faces):
        f_area = max((f[2] - f[0]) * (f[3] - f[1]), 1e-6)
        cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
        for j, p in enumerate(persons):
            iw = max(0.0, min(f[2], p[2]) - max(f[0], p[0]))
            ih = max(0.0, min(f[3], p[3]) - max(f[1], p[1]))
            inside = iw * ih / f_area
            head_zone = p[1] + (p[3] - p[1]) * max_head_fraction
            if inside >= min_inside and p[0] <= cx <= p[2] and p[1] <= cy <= head_zone:
                p_area = (p[2] - p[0]) * (p[3] - p[1])
                candidates.append((-inside, f_area / max(p_area, 1e-6) * -1, i, j))
    taken_faces, taken_persons = set(), set()
    for *_, i, j in sorted(candidates):
        if i not in taken_faces and j not in taken_persons:
            result[i] = j
            taken_faces.add(i)
            taken_persons.add(j)
    return result


def crop(image: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return np.ascontiguousarray(image[y1:y2, x1:x2])
