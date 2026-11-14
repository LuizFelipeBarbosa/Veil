"""SFace owner match, ID-swap re-checks (WP6), remembered unblurred people (WP15) and far-away faces (WP17)."""

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
_MAX_REMEMBERED = 16  # unblurred people remembered for the session; the oldest is dropped beyond this
# WP17 measurements (faces shrunk in memory): same-person misses are 6% at 35-80 px, 9% at 30, 21% at 20, 47% at 18.
_MIN_CHECK_PX = 32  # a face box shorter than this gives an embedding that is no evidence either way
_UPGRADE_RATIO = 1.5  # a matching face this many times taller than the saved reference replaces it
_HANDOVER_S = 0.4  # a lost shown or owner track may hand over to a new small track this soon after it was last seen
_HANDOVER_MIN_IOU = 0.5  # overlap needed between the lost track's last box and the new track's box
_HANDOVER_MAX_SIZE_RATIO = 1.25  # taller box height over shorter box height, at most
_CROWD_RADIUS_FACES = 3.0  # another face this many lost-face heights away (centre to centre) blocks a hand-over

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
class _Reference:
    """A unit embedding and the height in pixels of the face it was taken from."""

    embedding: np.ndarray
    face_px: float


@dataclass
class _LostTrack:
    """A shown or owner track that just left, which a new small track at the same spot may take over."""

    track_id: str
    xyxy: tuple[float, float, float, float]
    seen_at: float
    shown: bool
    owner_score: float | None
    reference: _Reference | None


@dataclass
class _Track:
    """What we know about one face track that is in frame."""

    checked_at: float | None = None  # None until the first check
    embedding: np.ndarray | None = None  # unit length; None when the last embedding failed
    embedded_px: float = 0.0  # height of the face the embedding was taken from
    xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # box in the latest frame
    seen_at: float = float("-inf")  # time of the latest frame
    owner_score: float | None = None  # cosine with the owner while this track counts as an owner match
    retry_until: float = float("-inf")  # a new track that missed the owner is checked every frame until then
    # A new track that matched no remembered person is checked every frame until then.
    # None until its first fresh embedding; -inf once it may no longer be re-shown automatically.
    reshow_until: float | None = None
    owner_strikes: _Strikes = field(default_factory=_Strikes)
    show_strikes: _Strikes = field(default_factory=_Strikes)

    def check_due(self, now: float) -> bool:
        if self.checked_at is None or self.owner_strikes.pending or self.show_strikes.pending:
            return True
        return now < self.retry_until or self.may_reshow(now) or now - self.checked_at >= config.ID_RECHECK_S

    def may_reshow(self, now: float) -> bool:
        """True while this new track may still be matched to a remembered person."""
        return self.reshow_until is None or now < self.reshow_until


class _RememberedPeople:
    """Unit embeddings of the people who were unblurred this session, oldest first. Memory only."""

    def __init__(self) -> None:
        self._people: list[_Reference] = []

    def __len__(self) -> int:
        return len(self._people)

    def add(self, reference: _Reference) -> None:
        if self.match(reference.embedding) is not None:
            return
        self._people.append(reference)
        del self._people[:-_MAX_REMEMBERED]

    def match(self, embedding: np.ndarray | None) -> _Reference | None:
        """The best remembered person at or above the match threshold, or None."""
        if embedding is None:
            return None
        best, best_score = None, -1.0
        for person in self._people:
            score = _cosine(person.embedding, embedding)
            if score >= config.SFACE_COSINE_MATCH and score > best_score:
                best, best_score = person, score
        return best

    def upgrade(self, old: np.ndarray, better: _Reference) -> None:
        """Replace the remembered embedding `old` (that very array) with a better one of the same person."""
        for person in self._people:
            if person.embedding is old:
                person.embedding, person.face_px = better.embedding, better.face_px

    def forget_matching(self, embedding: np.ndarray) -> None:
        self._people = [p for p in self._people if not _same_person(p.embedding, embedding)]

    def clear(self) -> None:
        self._people.clear()


class OwnerIdentity:
    """Remembers the owner's embedding and the embedding of every unblurred person.

    An unblurred person stays unblurred for the session: a later face track that
    matches them gets a "show" override again (this replaces the Brief's Never-leaks
    rule 7, by the lead's request), until they are blurred again or all faces are.
    Embeddings live in memory only; nothing here writes an image or an embedding to disk.

    A face shorter than _MIN_CHECK_PX is too small for its embedding to be evidence:
    it is not embedded, so its track keeps its status, is never newly owner or
    re-shown, and gives no reference. A shown or owner track that is lost can hand
    its status to a new track too small to check, at the same spot right after.
    """

    def __init__(self, model_path: Path = config.SFACE_MODEL, *, embed: Embedder | None = None) -> None:
        self._embed = embed if embed is not None else _sface_embedder(model_path)
        self._owner_embedding: np.ndarray | None = None
        self._owner_track: str | None = None
        self._tracks: dict[str, _Track] = {}  # face tracks in the latest frame
        self._show_refs: dict[str, _Reference] = {}  # saved when a "show" override was first seen
        self._lost: list[_LostTrack] = []  # shown or owner tracks lost within _HANDOVER_S
        self._people = _RememberedPeople()
        self._face_reset_count = 0  # policy face_reset_count seen on the previous update

    @property
    def enrolled(self) -> bool:
        return self._owner_embedding is not None

    def enroll(self, image: np.ndarray, face_row: np.ndarray) -> bool:
        """Store the owner's embedding (memory only) from one aligned face.

        False, and nothing changes, when the face is too small to give a usable embedding.
        """
        if face_row[3] < _MIN_CHECK_PX:
            return False
        self._owner_embedding = _unit(self._embed(image, face_row))
        self._owner_track = None
        self._tracks.clear()  # a new owner: every track is checked again
        self._lost.clear()
        return True

    def update(self, image: np.ndarray, boxes: list[TrackedBox], policy: Policy, now: float) -> None:
        """Set is_owner on at most one face box, and re-check about once a second.

        A failed re-check of the owner or of a "show" override is retried on every
        following frame with a fresh face row; the identity is revoked after
        _MAX_STRIKES failures in a row, or when nothing matched for _RETRY_WINDOW_S.
        A revoked override is removed from the policy, so a swapped ID blurs again
        (Never-leaks rule 6); the revoked track is not re-shown automatically.

        A new face track without an override that matches a remembered person gets a
        "show" override, checked against that person from then on.
        """
        reset_count = policy.snapshot().face_reset_count
        if reset_count != self._face_reset_count:  # "blur all faces": forget everyone first
            self._people.clear()
            self._lost.clear()
            self._face_reset_count = reset_count

        faces = [box for box in boxes if box.label == FACE]
        for box in boxes:
            box.is_owner = False
        previous = self._tracks
        self._tracks = {box.id: previous.get(box.id) or _Track() for box in faces}
        self._remember_lost_tracks(previous, policy)

        checked: set[str] = set()
        for box in faces:
            track = self._tracks[box.id]
            track.xyxy, track.seen_at = box.xyxy, now
            if _checkable(box) and track.check_due(now):
                self._check_owner(box.id, track, self._embed_or_none(image, box), now)
                track.embedded_px = _height(box.xyxy)
                checked.add(box.id)
            elif box.id == self._owner_track and track.owner_strikes.expired(now):
                _revoke_owner(track)

        self._forget_blurred(policy)
        self._hand_over_lost_tracks([box for box in faces if box.id not in previous], policy, now)
        self._reshow_remembered(faces, checked, policy, now)
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

    def _forget_blurred(self, policy: Policy) -> None:
        """A person whose track is blurred on request is not re-shown under a later track id."""
        for track_id, action in policy.snapshot().track_overrides.items():
            if action != "blur":
                continue
            track = self._tracks.get(track_id)
            reference = self._show_refs.get(track_id)
            for embedding in (reference.embedding if reference else None, track.embedding if track else None):
                if embedding is not None:
                    self._people.forget_matching(embedding)

    def _remember_lost_tracks(self, previous: dict[str, _Track], policy: Policy) -> None:
        """Note each shown or owner track that left since the last frame, unless another face was near it."""
        overrides = policy.snapshot().track_overrides
        for track_id, track in previous.items():
            shown = overrides.get(track_id) == "show"
            owner = track_id == self._owner_track
            if track_id in self._tracks or not (shown or owner):
                continue
            if track.owner_strikes.pending or track.show_strikes.pending:
                continue  # it was failing its checks: nothing to hand over
            if any(other_id != track_id and _near(track.xyxy, other.xyxy) for other_id, other in previous.items()):
                continue
            self._lost.append(_LostTrack(
                track_id, track.xyxy, track.seen_at, shown,
                track.owner_score if owner else None, self._show_refs.get(track_id),
            ))

    def _hand_over_lost_tracks(self, new_faces: list[TrackedBox], policy: Policy, now: float) -> None:
        """A new small track at the very spot of a just-lost shown or owner track takes over its status.

        Only when exactly one new track qualifies and it is too small to embed: a face
        that can be checked must match by embedding instead.
        """
        self._lost = [lost for lost in self._lost if now - lost.seen_at <= _HANDOVER_S]
        overrides = policy.snapshot().track_overrides
        for lost in list(self._lost):
            candidates = [
                box for box in new_faces
                if box.face_row is not None and box.id not in overrides and _same_spot(lost.xyxy, box.xyxy)
            ]
            if len(candidates) != 1 or _checkable(candidates[0]):
                continue
            self._lost.remove(lost)
            self._inherit(candidates[0].id, lost, policy)

    def _inherit(self, track_id: str, lost: _LostTrack, policy: Policy) -> None:
        track = self._tracks[track_id]
        if lost.owner_score is not None and self._owner_embedding is not None:
            track.owner_score = lost.owner_score
        if not lost.shown or policy.snapshot().track_overrides.get(lost.track_id) != "show":
            return  # not shown, or blurred or forgotten since it was lost
        policy.set_track_override(track_id, "show")
        if policy.snapshot().face_reset_count != self._face_reset_count:
            policy.clear_track_override(track_id)  # "blur all faces" landed meanwhile: fail closed
            return
        if lost.reference is not None:
            self._show_refs[track_id] = lost.reference
        track.reshow_until = float("-inf")

    def _reshow_remembered(self, faces: list[TrackedBox], checked: set[str], policy: Policy, now: float) -> None:
        """Give a new track a "show" override when its fresh embedding matches a remembered person."""
        snapshot = policy.snapshot()
        if snapshot.face_reset_count != self._face_reset_count:
            return  # all faces were blurred during this update: forget everyone on the next one first
        for box in faces:
            track = self._tracks[box.id]
            if box.id in snapshot.track_overrides:
                track.reshow_until = float("-inf")  # it has its own choice; never touched by this
                continue
            if box.id not in checked or not track.may_reshow(now):
                continue
            person = self._people.match(track.embedding)
            if person is not None:
                policy.set_track_override(box.id, "show")
                if policy.snapshot().face_reset_count != self._face_reset_count:
                    policy.clear_track_override(box.id)  # "blur all faces" landed meanwhile: fail closed
                    return
                self._show_refs[box.id] = person  # the swap check compares with the remembered face
                track.reshow_until = float("-inf")
            elif track.reshow_until is None:  # first sight missed: retry while anyone is remembered
                track.reshow_until = now + _RETRY_WINDOW_S if self._people else float("-inf")

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
                if track.embedding is not None:
                    reference = _Reference(track.embedding, track.embedded_px)
                    self._show_refs[box.id] = reference
                    self._people.add(reference)
                elif not _waiting_for_first_check(box, track):  # nothing to compare a later face with
                    self._revoke_show(box.id, track, policy)
                continue
            if box.id not in checked:
                revoke = track.show_strikes.expired(now)
            elif track.embedding is not None and _same_person(reference.embedding, track.embedding):
                track.show_strikes.reset()
                self._upgrade_reference(reference, track)
                revoke = False
            else:
                revoke = track.show_strikes.fail(now)
            if revoke:
                self._revoke_show(box.id, track, policy)

    def _upgrade_reference(self, reference: _Reference, track: _Track) -> None:
        """A clearly larger matching face makes a better reference, here and in memory."""
        if track.embedding is None or track.embedded_px < _UPGRADE_RATIO * reference.face_px:
            return
        better = _Reference(track.embedding, track.embedded_px)
        self._people.upgrade(reference.embedding, better)
        reference.embedding, reference.face_px = better.embedding, better.face_px

    def _revoke_show(self, track_id: str, track: _Track, policy: Policy) -> None:
        policy.clear_track_override(track_id)
        self._show_refs.pop(track_id, None)
        track.show_strikes.reset()
        track.reshow_until = float("-inf")  # this track is somebody else now; the person stays remembered

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


def _height(xyxy: tuple[float, float, float, float]) -> float:
    return xyxy[3] - xyxy[1]


def _checkable(box: TrackedBox) -> bool:
    """A fresh face large enough for its embedding to be evidence."""
    return box.face_row is not None and _height(box.xyxy) >= _MIN_CHECK_PX


def _waiting_for_first_check(box: TrackedBox, track: _Track) -> bool:
    """A shown track never checked yet because its face has been too small so far."""
    return track.checked_at is None and box.face_row is not None and not _checkable(box)


def _same_spot(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Strong overlap and a similar size."""
    taller, shorter = max(_height(a), _height(b)), min(_height(a), _height(b))
    return shorter > 0 and taller / shorter <= _HANDOVER_MAX_SIZE_RATIO and _iou(a, b) >= _HANDOVER_MIN_IOU


def _near(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """b's centre is within _CROWD_RADIUS_FACES face heights of a's centre."""
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return float(np.hypot(dx, dy)) < _CROWD_RADIUS_FACES * _height(a)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union


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
