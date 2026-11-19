"""Identity gallery: enrolled faces in SQLite, searched in memory with NumPy.

* Safe to share between processes (SQLite WAL): several workers or a web API can
  enroll and search the same file. Each process notices other processes' writes
  through ``PRAGMA data_version`` and reloads its in-memory matrix only then.
* Embeddings remember which model produced them; searches only compare within
  the same embedding space, so changing backends can never return false matches.
* Search is a single matrix product, so tens of thousands of embeddings are
  searched in milliseconds. An identity's score is its best-matching sample.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    space TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS embeddings_space ON embeddings(space, identity_id);
"""


@dataclass
class Match:
    name: str | None  # None = unknown
    score: float  # best cosine similarity found (even if below threshold)
    candidate: str | None = None  # best identity regardless of threshold


@dataclass
class _Matrix:
    vectors: np.ndarray  # (M, D), rows grouped by identity
    starts: np.ndarray  # row index where each identity's block starts
    names: list[str]


class Gallery:
    def __init__(self, path: str | Path = "gallery.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        self._cache: dict[str, _Matrix] = {}
        self._version = None

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Gallery:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- writes
    def enroll(self, name: str, embeddings: np.ndarray, space: str, meta: dict | None = None) -> int:
        """Add one or more embeddings (N, D) for ``name``; creates the identity if needed."""
        vectors = np.atleast_2d(np.asarray(embeddings, np.float32))
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "INSERT INTO identities(name, meta, created) VALUES (?, ?, ?) ON CONFLICT(name) DO NOTHING",
                    (name, json.dumps(meta or {}), now),
                )
                (identity_id,) = self._db.execute("SELECT id FROM identities WHERE name = ?", (name,)).fetchone()
                self._db.executemany(
                    "INSERT INTO embeddings(identity_id, space, dim, vector, created) VALUES (?, ?, ?, ?, ?)",
                    [(identity_id, space, vectors.shape[1], v.tobytes(), now) for v in vectors],
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._cache.clear()
        return identity_id

    def remove(self, name: str) -> bool:
        with self._lock:
            removed = self._db.execute("DELETE FROM identities WHERE name = ?", (name,)).rowcount > 0
            self._cache.clear()
        return removed

    # ----------------------------------------------------------------- reads
    def identities(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT i.name, i.meta, COUNT(e.id), GROUP_CONCAT(DISTINCT e.space) FROM identities i "
            "LEFT JOIN embeddings e ON e.identity_id = i.id GROUP BY i.id ORDER BY i.name"
        ).fetchall()
        return [
            {"name": n, "meta": json.loads(m), "samples": c, "spaces": (s or "").split(",") if s else []}
            for n, m, c, s in rows
        ]

    def _matrix(self, space: str) -> _Matrix:
        with self._lock:
            version = self._db.execute("PRAGMA data_version").fetchone()[0]
            if version != self._version:  # another process wrote to the gallery
                self._cache.clear()
                self._version = version
            if space not in self._cache:
                rows = self._db.execute(
                    "SELECT i.name, e.vector, e.dim FROM embeddings e JOIN identities i ON i.id = e.identity_id "
                    "WHERE e.space = ? ORDER BY e.identity_id, e.id",
                    (space,),
                ).fetchall()
                if rows:
                    dim = rows[0][2]
                    vectors = np.frombuffer(b"".join(r[1] for r in rows), np.float32).reshape(-1, dim)
                    names, starts = [], []
                    for i, (name, _, _) in enumerate(rows):
                        if not names or names[-1] != name:
                            names.append(name)
                            starts.append(i)
                    self._cache[space] = _Matrix(vectors, np.asarray(starts), names)
                else:
                    self._cache[space] = _Matrix(np.zeros((0, 1), np.float32), np.zeros(0, int), [])
            return self._cache[space]

    def search(self, embeddings: np.ndarray, space: str) -> tuple[np.ndarray, list[str]]:
        """(N, identities) best cosine similarity of each query to each identity, plus their names."""
        m = self._matrix(space)
        queries = np.atleast_2d(np.asarray(embeddings, np.float32))
        if not m.names:
            return np.zeros((len(queries), 0), np.float32), []
        if queries.shape[1] != m.vectors.shape[1]:
            raise ValueError(f"query dim {queries.shape[1]} != gallery dim {m.vectors.shape[1]} for '{space}'")
        sims = queries @ m.vectors.T
        return np.maximum.reduceat(sims, m.starts, axis=1), m.names

    def identify(self, embeddings: np.ndarray, space: str, threshold: float) -> list[Match]:
        if len(np.atleast_2d(embeddings)) == 0 or np.asarray(embeddings).size == 0:
            return []
        scores, names = self.search(embeddings, space)
        if not names:
            return [Match(None, 0.0) for _ in range(len(scores))]
        best = scores.argmax(axis=1)
        out = []
        for row, idx in zip(scores, best, strict=True):
            score = float(row[idx])
            out.append(Match(names[idx] if score >= threshold else None, score, names[idx]))
        return out
