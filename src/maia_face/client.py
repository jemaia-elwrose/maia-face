"""Use the ``face`` / ``face-attr`` services from a pipeline."""

from __future__ import annotations

import numpy as np
from maia_infer import Client

from .association import crop
from .backends.mivolo import Attributes
from .gallery import Gallery, Match
from .types import Face, unpack_faces


class FaceClient:
    def __init__(
        self, client: Client | str = "tcp://localhost:5555", service: str = "face", attr_service: str = "face-attr"
    ) -> None:
        self.client = Client(client) if isinstance(client, str) else client
        self.service, self.attr_service = service, attr_service
        self.space: str | None = None
        self.threshold: float | None = None

    def detect(self, image: np.ndarray, embed: bool = True) -> list[Face]:
        res = self.client.infer(self.service, image, meta={"embed": embed})
        self.space, self.threshold = res.meta.get("space"), res.meta.get("threshold")
        return unpack_faces(res.arrays[0], res.arrays[1] if embed else None)

    def identify(self, image: np.ndarray, gallery: Gallery, threshold: float | None = None) -> list[tuple[Face, Match]]:
        faces = self.detect(image)
        if not faces:
            return []
        matches = gallery.identify(
            np.stack([f.embedding for f in faces]), self.space, threshold if threshold is not None else self.threshold
        )
        return list(zip(faces, matches, strict=True))

    def attributes(self, image: np.ndarray, face_box=None, person_box=None) -> Attributes:
        empty = np.zeros((0, 0, 3), np.uint8)
        face = crop(image, face_box, pad=0.1) if face_box is not None else empty
        body = crop(image, person_box) if person_box is not None else empty
        m = self.client.infer(self.attr_service, [face, body]).meta
        return Attributes(m["age"], m["gender"], m["gender_conf"])
