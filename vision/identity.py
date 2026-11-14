"""SFace owner match and ID-swap re-checks (WP6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import config
from contracts import FACE, Policy, TrackedBox

_MAX_STRIKES = 3  # consecutive failed per-frame checks before the owner or a "show" override is revoked
_RETRY_WINDOW_S = 0.5  # how long failed checks are retried every frame before giving up

# (raw BGR image, YuNet face row) -> SFace feature. Injectable so tests need no real faces.
Embedder = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class _Strikes:
    """Consecutive failed checks of one identity, and when the first of them happened."""

    count: int = 0
    since: float | None = None

    @property
    def pending(self) -> bool:
        return self.count > 0

    def fail(self, now: float) -> bool:
        """Counts a failed check. True when the identity must be revoked."""
        if self.since is None:
            self.since = now
        self.count += 1
        return self.count >= _MAX_STRIKES or self.expired(now)

    def expired(self, now: float) -> bool:
        """True when no check has succeeded within the retry window since the first failure."""
        return self.since is not None and now - self.since >= _RETRY_WINDOW_S

    def reset(self) -> None:
        self.count, self.since = 0, None


@dataclass
class _Track:
    """What we know about one face track that is in frame."""

    checked_at: float | None = None  # None until the first check
    embedding: np.ndarray | None = None  # unit length; None when the last embedding failed
    owner_score: float | None = None  # cosine with the owner while this track counts as an owner match
    retry_until: float = float("-inf")  # a new track that missed the owner is checked every frame until then
    owner_strikes: _Strikes = field(default_factory=_Strikes)
    show_strikes: _Strikes = field(default_factory=_Strikes)

    def check_due(self, now: float) -> bool:
        if self.checked_at is None or self.owner_strikes.pending or self.show_strikes.pending:
            return True
        return now < self.retry_until or now - self.checked_at >= config.ID_RECHECK_S


class OwnerIdentity:
    """Remembers the owner's embedding and the embedding of every unblurred person.

    Embeddings live in memory only; nothing here writes an image or an embedding to disk.
    """

    def __init__(self, model_path: Path = config.SFACE_MODEL, *, embed: Embedder | None = None) -> None:
        self._embed = embed if embed is not None else _sface_embedder(model_path)
        self._owner_embedding: np.ndarray | None = None
        self._owner_track: str | None = None
        self._tracks: dict[str, _Track] = {}  # face tracks in the latest frame
        self._show_refs: dict[str, np.ndarray] = {}  # embedding saved when a "show" override was first seen

    @property
    def enrolled(self) -> bool:
        return self._owner_embedding is not None

    def enroll(self, image: np.ndarray, face_row: np.ndarray) -> None:
        """Store the owner's embedding (memory only) from one aligned face."""
        self._owner_embedding = _unit(self._embed(image, face_row))
        self._owner_track = None
        self._tracks.clear()  # a new owner: every track is checked again

    def update(self, image: np.ndarray, boxes: list[TrackedBox], policy: Policy, now: float) -> None:
        """Set is_owner on at most one face box, and re-check about once a second.

        A failed re-check of the owner or of a "show" override is retried on every
        following frame with a fresh face row; the identity is revoked after
        _MAX_STRIKES failures in a row, or when nothing matched for _RETRY_WINDOW_S.
        A revoked override is removed from the policy, so a swapped ID blurs again
        (Never-leaks rule 6).
        """
        faces = [box for box in boxes if box.label == FACE]
        for box in boxes:
            box.is_owner = False
        self._tracks = {box.id: self._tracks.get(box.id) or _Track() for box in faces}

        checked: set[str] = set()
        for box in faces:
            track = self._tracks[box.id]
            if box.face_row is not None and track.check_due(now):
                self._check_owner(box.id, track, self._embed_or_none(image, box), now)
                checked.add(box.id)
            elif box.id == self._owner_track and track.owner_strikes.expired(now):
                _revoke_owner(track)

        self._check_show_overrides(faces, checked, policy, now)
        owner_box = self._mark_owner(faces)
        if owner_box is not None and owner_box.face_row is not None:
            # The owner is in view, so no other track needs a fast retry to find them.
            for track in self._tracks.values():
                track.retry_until = float("-inf")

    def _embed_or_none(self, image: np.ndarray, box: TrackedBox) -> np.ndarray | None:
        """Fail closed: an embedding that cannot be computed is a failed check."""
        try:
            return _unit(self._embed(image, box.face_row))
        except Exception:
            return None

    def _check_owner(self, track_id: str, track: _Track, embedding: np.ndarray | None, now: float) -> None:
        first_check = track.checked_at is None
        track.checked_at, track.embedding = now, embedding
        if self._owner_embedding is None:
            return
        if embedding is not None and _same_person(self._owner_embedding, embedding):
            track.owner_score = _cosine(self._owner_embedding, embedding)
            track.owner_strikes.reset()
            track.retry_until = float("-inf")
        elif track_id == self._owner_track:
            if track.owner_strikes.fail(now):  # stays owner while strikes are pending
                _revoke_owner(track)
        else:
            track.owner_score = None
            if first_check:
                track.retry_until = now + _RETRY_WINDOW_S

    def _check_show_overrides(self, faces: list[TrackedBox], checked: set[str], policy: Policy, now: float) -> None:
        overrides = policy.snapshot().track_overrides
        shown = {track_id for track_id, action in overrides.items() if action == "show"}
        self._show_refs = {t: ref for t, ref in self._show_refs.items() if t in shown}
        for box in faces:
            track = self._tracks[box.id]
            if box.id not in shown:
                track.show_strikes.reset()
                continue
            reference = self._show_refs.get(box.id)
            if reference is None:
                if track.embedding is None:  # nothing to compare a later face with
                    self._revoke_show(box.id, track, policy)
                else:
                    self._show_refs[box.id] = track.embedding
                continue
            if box.id not in checked:
                revoke = track.show_strikes.expired(now)
            elif track.embedding is not None and _same_person(reference, track.embedding):
                track.show_strikes.reset()
                revoke = False
            else:
                revoke = track.show_strikes.fail(now)
            if revoke:
                self._revoke_show(box.id, track, policy)

    def _revoke_show(self, track_id: str, track: _Track, policy: Policy) -> None:
        policy.clear_track_override(track_id)
        self._show_refs.pop(track_id, None)
        track.show_strikes.reset()

    def _mark_owner(self, faces: list[TrackedBox]) -> TrackedBox | None:
        """The best match at or above the threshold. A box without a fresh face row can only stay owner."""
        owner_box, best_score = None, -1.0
        for box in faces:
            score = self._tracks[box.id].owner_score
            if score is None or (box.face_row is None and box.id != self._owner_track):
                continue
            if score > best_score:
                owner_box, best_score = box, score
        for box in faces:
            track = self._tracks[box.id]
            if box is not owner_box and track.owner_strikes.pending:
                _revoke_owner(track)  # outscored while failing: no stale score to win with later
        self._owner_track = owner_box.id if owner_box is not None else None
        if owner_box is not None:
            owner_box.is_owner = True
        return owner_box


def _revoke_owner(track: _Track) -> None:
    track.owner_score = None
    track.owner_strikes.reset()


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
