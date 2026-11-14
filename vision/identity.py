"""SFace owner match and ID-swap re-checks (WP6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import config
from contracts import FACE, Policy, TrackedBox

# (raw BGR image, YuNet face row) -> SFace feature. Injectable so tests need no real faces.
Embedder = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class _TrackCheck:
    embedding: np.ndarray | None  # unit length; None when the last embedding failed
    checked_at: float


class OwnerIdentity:
    """Remembers the owner's embedding and the embedding of every unblurred person.

    Embeddings live in memory only; nothing here writes an image or an embedding to disk.
    """

    def __init__(self, model_path: Path = config.SFACE_MODEL, *, embed: Embedder | None = None) -> None:
        self._embed = embed if embed is not None else _sface_embedder(model_path)
        self._owner_embedding: np.ndarray | None = None
        self._owner_track: str | None = None
        self._checks: dict[str, _TrackCheck] = {}  # latest embedding per face track in frame
        self._override_refs: dict[str, np.ndarray] = {}  # embedding saved when a "show" override was first seen

    @property
    def enrolled(self) -> bool:
        return self._owner_embedding is not None

    def enroll(self, image: np.ndarray, face_row: np.ndarray) -> None:
        """Store the owner's embedding (memory only) from one aligned face."""
        self._owner_embedding = _unit(self._embed(image, face_row))
        self._owner_track = None

    def update(self, image: np.ndarray, boxes: list[TrackedBox], policy: Policy, now: float) -> None:
        """Set is_owner on at most one face box, and re-check about once a second.

        A "show" override whose face no longer matches its saved embedding is
        removed from the policy, so a swapped ID blurs again (Never-leaks rule 6).
        """
        faces = [box for box in boxes if box.label == FACE]
        for box in boxes:
            box.is_owner = False
        present = {box.id for box in faces}
        self._checks = {track_id: check for track_id, check in self._checks.items() if track_id in present}

        rechecked: set[str] = set()
        for box in faces:
            if self._recheck_due(box, now):
                self._checks[box.id] = _TrackCheck(self._embed_or_none(image, box), now)
                rechecked.add(box.id)

        self._check_show_overrides(faces, rechecked, policy)
        self._mark_owner(faces)

    def _recheck_due(self, box: TrackedBox, now: float) -> bool:
        if box.face_row is None:
            return False
        check = self._checks.get(box.id)
        return check is None or now - check.checked_at >= config.ID_RECHECK_S

    def _embed_or_none(self, image: np.ndarray, box: TrackedBox) -> np.ndarray | None:
        """Fail closed: a box we cannot embed is never the owner and loses any show override."""
        try:
            return _unit(self._embed(image, box.face_row))
        except Exception:
            return None

    def _check_show_overrides(self, faces: list[TrackedBox], rechecked: set[str], policy: Policy) -> None:
        overrides = policy.snapshot().track_overrides
        shown = {track_id for track_id, action in overrides.items() if action == "show"}
        self._override_refs = {t: ref for t, ref in self._override_refs.items() if t in shown}
        for box in faces:
            if box.id not in shown:
                continue
            check = self._checks.get(box.id)
            if check is None or check.embedding is None:
                self._revoke(box.id, policy)
                continue
            reference = self._override_refs.setdefault(box.id, check.embedding)
            if box.id in rechecked and not _same_person(reference, check.embedding):
                self._revoke(box.id, policy)

    def _revoke(self, track_id: str, policy: Policy) -> None:
        policy.clear_track_override(track_id)
        self._override_refs.pop(track_id, None)

    def _mark_owner(self, faces: list[TrackedBox]) -> None:
        """The best match at or above the threshold. A box without a fresh face row can only stay owner."""
        owner_box, best_score = None, -1.0
        if self._owner_embedding is not None:
            for box in faces:
                check = self._checks.get(box.id)
                if check is None or check.embedding is None:
                    continue
                if box.face_row is None and box.id != self._owner_track:
                    continue
                score = _cosine(self._owner_embedding, check.embedding)
                if score >= config.SFACE_COSINE_MATCH and score > best_score:
                    owner_box, best_score = box, score
        self._owner_track = owner_box.id if owner_box is not None else None
        if owner_box is not None:
            owner_box.is_owner = True


def _sface_embedder(model_path: Path) -> Embedder:
    recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        return recognizer.feature(recognizer.alignCrop(image, face_row))

    return embed


def _unit(feature: np.ndarray) -> np.ndarray:
    """Flattened and L2-normalized, so the dot product is SFace's FR_COSINE score."""
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("empty face embedding")
    return vector / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _same_person(a: np.ndarray, b: np.ndarray) -> bool:
    return _cosine(a, b) >= config.SFACE_COSINE_MATCH
