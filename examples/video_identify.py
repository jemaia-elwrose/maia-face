"""Identify enrolled people in a video or RTSP stream through the `face` service,
voting per track so a name is shown only once it is consistent across frames.

    maia-infer broker &  maia-infer worker face &
    maia-face enroll "Alice" alice*.jpg
    python examples/video_identify.py --source rtsp://camera/stream --show
"""

import argparse

import cv2
import numpy as np

from maia_face import Gallery, IdentityVoter
from maia_face.client import FaceClient


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class IoUTracker:
    """Tiny tracker for the demo; in production use the tracker of your pipeline (e.g. ByteTrack)."""

    def __init__(self, threshold=0.3):
        self.tracks, self.next_id, self.threshold = {}, 1, threshold

    def update(self, boxes):
        ids, used = [], set()
        for box in boxes:
            best = max(((iou(box, b), t) for t, b in self.tracks.items() if t not in used), default=(0, None))
            tid = best[1] if best[0] >= self.threshold else None
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            self.tracks[tid] = box
            used.add(tid)
            ids.append(tid)
        self.tracks = {t: b for t, b in self.tracks.items() if t in used}
        return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--endpoint", default="tcp://localhost:5555")
    ap.add_argument("--gallery", default="gallery.db")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    fc, gallery, voter, tracker = FaceClient(args.endpoint), Gallery(args.gallery), IdentityVoter(), IoUTracker()
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = fc.detect(frame)
        matches = gallery.identify(np.stack([f.embedding for f in faces]), fc.space, fc.threshold) if faces else []
        for face, match, tid in zip(faces, matches, tracker.update([f.box for f in faces]), strict=True):
            decision = voter.update(tid, match.name, match.score, frame_no)
            label = f"#{tid} {decision.name or '...'}"
            x1, y1, x2, y2 = (int(v) for v in face.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120) if decision.name else (160, 160, 160), 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        voter.prune(frame_no)
        frame_no += 1
        if args.show:
            cv2.imshow("maia-face", frame)
            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":
    main()
