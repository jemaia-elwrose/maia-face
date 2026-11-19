"""Age and gender with MiVOLO (face + body transformer).

Uses the MiVOLO library API (``MiVOLO(checkpoint)``, face and body crops
stacked channel-wise). Code: Apache-2.0. Check the license of the checkpoint
you use: the published v2 weights are CC BY-NC 4.0 (non-commercial).
Either crop may be missing; the available one is then used for both inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Attributes:
    age: float
    gender: str  # "male" | "female"
    gender_conf: float


class MiVOLOEstimator:
    def __init__(self, checkpoint: str, device: str = "cuda", half: bool = False) -> None:
        from mivolo.data.misc import prepare_classification_images
        from mivolo.model.mi_volo import MiVOLO

        self._prepare_images = prepare_classification_images
        self.model = MiVOLO(checkpoint, device=device, half=half)

    def _prepare(self, crops: list[np.ndarray]):
        m = self.model
        return self._prepare_images(crops, m.input_size, m.data_config["mean"], m.data_config["std"], device=m.device)

    def estimate(self, pairs: list[tuple[np.ndarray | None, np.ndarray | None]]) -> list[Attributes]:
        import torch

        if not pairs:
            return []
        faces, bodies = [], []
        for face, body in pairs:
            if face is None and body is None:
                raise ValueError("each request needs a face crop, a body crop, or both")
            faces.append(face if face is not None else body)
            bodies.append(body if body is not None else face)
        with torch.inference_mode():
            out = self.model.inference(torch.cat((self._prepare(faces), self._prepare(bodies)), dim=1))
        probs = out[:, :2].softmax(-1)
        meta = self.model.meta
        ages = out[:, 2] * (meta.max_age - meta.min_age) + meta.avg_age
        results = []
        for i in range(len(pairs)):
            idx = int(probs[i].argmax())
            results.append(Attributes(float(ages[i]), "male" if idx == 0 else "female", float(probs[i, idx])))
        return results
