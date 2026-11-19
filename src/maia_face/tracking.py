"""Per-track smoothing: decide identity and attributes over many frames, not one.

A single frame is noisy (blur, profile view, occlusion). Video pipelines track
people anyway, so votes are collected per track id and a decision is only made
when the evidence is consistent.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass


@dataclass
class TrackDecision:
    name: str | None
    confidence: float  # share of the voting weight held by `name`
    votes: int


class IdentityVoter:
    def __init__(self, window: int = 30, min_votes: int = 3, min_share: float = 0.6, max_age: int = 150) -> None:
        self.window, self.min_votes, self.min_share, self.max_age = window, min_votes, min_share, max_age
        self._votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._last_seen: dict[int, int] = {}

    def update(self, track_id: int, name: str | None, score: float, frame: int) -> TrackDecision:
        """Add one observation (``name=None`` = seen but not recognised) and return the current decision."""
        self._votes[track_id].append((name, max(0.0, float(score))))
        self._last_seen[track_id] = frame
        return self.decision(track_id)

    def decision(self, track_id: int) -> TrackDecision:
        votes = self._votes.get(track_id)
        if not votes:
            return TrackDecision(None, 0.0, 0)
        weights: Counter = Counter()
        for name, score in votes:
            weights[name] += score if name is not None else 0.5  # unknowns still count against a weak match
        total = sum(weights.values()) or 1.0
        best, weight = max(((n, w) for n, w in weights.items() if n is not None), key=lambda t: t[1], default=(None, 0))
        known_votes = sum(1 for n, _ in votes if n == best)
        if best is None or known_votes < self.min_votes or weight / total < self.min_share:
            return TrackDecision(None, weight / total, len(votes))
        return TrackDecision(best, weight / total, len(votes))

    def prune(self, frame: int) -> list[int]:
        """Forget tracks not seen for ``max_age`` frames. Returns the removed ids."""
        stale = [t for t, last in self._last_seen.items() if frame - last > self.max_age]
        for t in stale:
            self._votes.pop(t, None)
            self._last_seen.pop(t, None)
        return stale


class AttributeSmoother:
    """Median age and confidence-weighted gender over the last ``window`` estimates of a track."""

    def __init__(self, window: int = 15) -> None:
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, age: float, gender: str, gender_conf: float) -> tuple[float, str, float]:
        h = self._hist[track_id]
        h.append((float(age), gender, float(gender_conf)))
        ages = sorted(a for a, _, _ in h)
        mid = len(ages) // 2
        age_med = ages[mid] if len(ages) % 2 else (ages[mid - 1] + ages[mid]) / 2
        g = Counter()
        for _, gen, conf in h:
            g[gen] += conf
        gender_best, w = g.most_common(1)[0]
        return age_med, gender_best, w / (sum(g.values()) or 1.0)

    def forget(self, track_id: int) -> None:
        self._hist.pop(track_id, None)
