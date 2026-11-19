"""Five-point face alignment to the standard 112x112 ArcFace template.

Every recognizer in maia-face (SFace, ArcFace) receives crops aligned the same
way, so detectors and recognizers can be mixed freely.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Landmark positions (left eye, right eye, nose, left mouth, right mouth) in a 112x112 crop.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float32,
)


def similarity_matrix(landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """2x3 similarity transform (rotation, uniform scale, translation) mapping landmarks to the template."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not estimate alignment from landmarks")
    return matrix


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    return cv2.warpAffine(image, similarity_matrix(landmarks, size), (size, size), borderValue=0.0)
