"""WP6: owner match and ID-swap check. WP15: unblurred people are remembered for the session.

Unit tests use a fake embedder: a face row's first value names a person, and each
person has a fixed vector. Clip tests use real YuNet + SFace on recordings from the
main checkout and skip while those clips are missing.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from contracts import Policy, TrackedBox
from vision.identity import OwnerIdentity, _RememberedPeople
from vision.policy import decide

OWNER, TWIN, STRANGER, OTHER, BROKEN = 0, 1, 2, 3, 4
PEOPLE = {
    OWNER: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    TWIN: np.array([0.6, 0.8, 0.0], dtype=np.float32),  # cosine 0.6 with the owner: a match, but a weaker one
    STRANGER: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    OTHER: np.array([0.0, 0.0, 1.0], dtype=np.float32),
}
IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME_S = 1 / 15


class FakeEmbedder:
    """Looks the person up from the face row and counts calls per person."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        person = int(face_row[0])
        self.calls.append(person)
        if person == BROKEN:
            raise RuntimeError("alignCrop failed")
        return PEOPLE[person]


def _row(person: int) -> np.ndarray:
    return np.array([person] + [0] * 14, dtype=np.float32)


def _face(track_id: str, person: int | None) -> TrackedBox:
    """person None is a held-over box without a fresh face row."""
    return TrackedBox(
        id=track_id, label="face", xyxy=(100, 100, 200, 220), conf=0.9, frame_index=0, last_seen=0.0,
        face_row=None if person is None else _row(person),
    )


def _identity(enroll: bool = True) -> tuple[OwnerIdentity, FakeEmbedder]:
    embedder = FakeEmbedder()
    identity = OwnerIdentity(embed=embedder)
    if enroll:
        identity.enroll(IMAGE, _row(OWNER))
        embedder.calls.clear()
    return identity, embedder


def _owners(identity: OwnerIdentity, faces: dict[str, int | None], policy: Policy, now: float) -> list[str]:
    boxes = [_face(track_id, person) for track_id, person in faces.items()]
    identity.update(IMAGE, boxes, policy, now)
    return [box.id for box in boxes if box.is_owner]


# --- owner ---


def test_nobody_is_owner_before_enrolment() -> None:
    identity, _ = _identity(enroll=False)
    assert not identity.enrolled
    assert _owners(identity, {"face#1": OWNER, "face#2": STRANGER}, Policy(), 0.0) == []


def test_enrolled_owner_is_marked() -> None:
    identity, _ = _identity()
    assert identity.enrolled
    assert _owners(identity, {"face#1": STRANGER, "face#2": OWNER}, Policy(), 0.0) == ["face#2"]


def test_only_one_owner_when_two_faces_match() -> None:
    identity, _ = _identity()
    assert _owners(identity, {"face#1": TWIN, "face#2": OWNER}, Policy(), 0.0) == ["face#2"]
    identity, _ = _identity()
    assert len(_owners(identity, {"face#1": OWNER, "face#2": OWNER}, Policy(), 0.0)) == 1


def test_owner_that_stops_matching_loses_it_after_three_failed_checks() -> None:
    identity, _ = _identity()
    policy = Policy()
    assert _owners(identity, {"face#1": OWNER}, policy, 0.0) == ["face#1"]
    assert _owners(identity, {"face#1": STRANGER}, policy, 0.5) == ["face#1"]  # cached until the re-check
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0) == ["face#1"]  # strike 1
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0 + FRAME_S) == ["face#1"]  # strike 2
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0 + 2 * FRAME_S) == []  # strike 3


def test_held_over_box_is_never_newly_owner_and_is_not_rechecked() -> None:
    identity, embedder = _identity()
    policy = Policy()
    _owners(identity, {"face#1": STRANGER, "face#2": OWNER}, policy, 0.0)
    calls = len(embedder.calls)
    # Held over past the re-check time: face#2 stays owner, and neither box is embedded again.
    assert _owners(identity, {"face#1": None, "face#2": None}, policy, 5.0) == ["face#2"]
    assert len(embedder.calls) == calls
    # A brand-new held-over track cannot become owner.
    identity, _ = _identity()
    _owners(identity, {"face#3": OWNER}, policy, 0.0)
    assert _owners(identity, {"face#4": None}, policy, 0.1) == []


def test_owner_that_leaves_and_returns_as_a_held_over_box_is_not_owner() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#1": OWNER}, policy, 0.0)
    _owners(identity, {}, policy, 0.1)
    assert _owners(identity, {"face#1": None}, policy, 0.2) == []


def test_non_face_boxes_are_ignored_and_never_owner() -> None:
    identity, embedder = _identity()
    plate = TrackedBox(
        id="obj#1", label="license plate", xyxy=(0, 0, 10, 10), conf=0.9, frame_index=0, last_seen=0.0,
        is_owner=True, face_row=_row(OWNER),
    )
    identity.update(IMAGE, [plate], Policy(), 0.0)
    assert plate.is_owner is False
    assert embedder.calls == []


def test_rechecks_happen_at_most_once_per_second_per_track(monkeypatch) -> None:
    monkeypatch.setattr(config, "ID_RECHECK_S", 1.0)
    times: dict[int, list[float]] = {OWNER: [], STRANGER: []}
    now = 0.0

    def embed(image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        times[int(face_row[0])].append(now)
        return PEOPLE[int(face_row[0])]

    identity = OwnerIdentity(embed=embed)
    identity.enroll(IMAGE, _row(OWNER))
    times[OWNER].clear()
    policy = Policy()
    policy.set_track_override("face#2", "show")
    for i in range(45):  # 3 s at 15 FPS
        now = i * FRAME_S
        _owners(identity, {"face#1": OWNER, "face#2": STRANGER}, policy, now)
    for track_times in times.values():
        assert 3 <= len(track_times) <= 4
        assert all(b - a >= 1.0 - 1e-9 for a, b in zip(track_times, track_times[1:]))
    assert policy.snapshot().track_overrides == {"face#2": "show"}


# --- ID-swap check ---


def _run(identity: OwnerIdentity, policy: Policy, faces: dict[str, int | None], start: float, end: float) -> float:
    """Feed frames at 15 FPS from start up to end (exclusive); returns the next frame time."""
    now = start
    while now < end - 1e-9:
        _owners(identity, faces, policy, now)
        now += FRAME_S
    return now


def test_unchanged_show_override_is_kept() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#7": STRANGER}, policy, 0.0)
    policy.set_track_override("face#7", "show")
    _run(identity, policy, {"face#7": STRANGER}, FRAME_S, 5.0)
    assert policy.snapshot().track_overrides == {"face#7": "show"}


def test_swapped_show_override_is_cleared_within_one_interval_and_not_before() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#7": STRANGER}, policy, 0.0)  # embedded at t=0
    policy.set_track_override("face#7", "show")
    _owners(identity, {"face#7": STRANGER}, policy, 0.1)  # reference saved; re-checked (new-track retry window)
    swap_at = 0.6  # after the new track's 0.5 s retry window
    now = _run(identity, policy, {"face#7": OTHER}, swap_at, 1.1)  # the tracker hands face#7 to someone else
    assert policy.snapshot().track_overrides == {"face#7": "show"}  # not before the re-check
    for _ in range(3):  # first frame at or after t=1.1, then two more per-frame retries
        assert policy.snapshot().track_overrides == {"face#7": "show"}
        _owners(identity, {"face#7": OTHER}, policy, now)
        now += FRAME_S
    assert policy.snapshot().track_overrides == {}
    assert now - FRAME_S - swap_at <= config.ID_RECHECK_S
    box = _face("face#7", OTHER)
    identity.update(IMAGE, [box], policy, now + FRAME_S)
    assert decide(box, policy.snapshot()) == "blur"


def test_override_reference_is_forgotten_when_the_override_goes() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#7": STRANGER}, policy, 0.0)
    policy.set_track_override("face#7", "show")
    _owners(identity, {"face#7": STRANGER}, policy, 0.1)
    policy.clear_track_override("face#7")
    now = _run(identity, policy, {"face#7": OTHER}, 0.2, 1.5)  # re-checked as OTHER with no override
    policy.set_track_override("face#7", "show")  # the operator now unblurs the person they see
    _run(identity, policy, {"face#7": OTHER}, now, 4.0)
    assert policy.snapshot().track_overrides == {"face#7": "show"}


def test_owner_with_show_override_is_also_swap_checked() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#2": OWNER}, policy, 0.0)
    policy.set_track_override("face#2", "show")
    now = _run(identity, policy, {"face#2": OWNER}, FRAME_S, 0.5)
    _run(identity, policy, {"face#2": STRANGER}, now, 1.3)  # re-check near t=1.0 plus two retries
    assert policy.snapshot().track_overrides == {}


def test_object_overrides_are_untouched() -> None:
    identity, _ = _identity()
    policy = Policy()
    policy.set_track_override("obj#3", "show")
    _run(identity, policy, {"face#1": STRANGER}, 0.0, 2.0)
    assert policy.snapshot().track_overrides == {"obj#3": "show"}


# --- remembered people (WP15; replaces the Brief's rule 7) ---


def _overrides(policy: Policy) -> dict[str, str]:
    return dict(policy.snapshot().track_overrides)


def _unblur_then_leave(identity: OwnerIdentity, policy: Policy, track_id: str, person: int, start: float) -> float:
    """The person is unblurred on track_id, stays a second, then leaves the frame for a second."""
    _owners(identity, {track_id: person}, policy, start)
    policy.set_track_override(track_id, "show")
    now = _run(identity, policy, {track_id: person}, start + FRAME_S, start + 1.0)
    return _run(identity, policy, {}, now, now + 1.0)


def test_returning_person_under_a_new_track_id_is_shown_and_nobody_else() -> None:
    identity, _ = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    boxes = [_face("face#5", STRANGER), _face("face#6", OTHER)]
    identity.update(IMAGE, boxes, policy, now)  # first frame of both new tracks
    assert _overrides(policy) == {"face#1": "show", "face#5": "show"}
    assert [decide(box, policy.snapshot()) for box in boxes] == ["show", "blur"]
    assert not any(box.is_owner for box in boxes)
    _run(identity, policy, {"face#5": STRANGER, "face#6": OTHER}, now + FRAME_S, now + 4.0)
    assert _overrides(policy) == {"face#1": "show", "face#5": "show"}


@pytest.mark.parametrize("owner_in_view", [False, True])
def test_blurry_first_frames_of_a_returning_person_are_retried(owner_in_view: bool) -> None:
    identity, _ = _identity(enroll=owner_in_view)
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    owner = {"face#9": OWNER} if owner_in_view else {}
    _owners(identity, {**owner, "face#5": BROKEN}, policy, now)  # embedding fails
    _owners(identity, {**owner, "face#5": OTHER}, policy, now + FRAME_S)  # motion blur: looks like nobody known
    assert "face#5" not in _overrides(policy)
    _owners(identity, {**owner, "face#5": STRANGER}, policy, now + 2 * FRAME_S)
    assert _overrides(policy)["face#5"] == "show"


def test_a_new_track_is_only_matched_while_it_is_new() -> None:
    identity, _ = _identity(enroll=False)
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    now = _run(identity, policy, {"face#5": OTHER}, now, now + 0.6)  # past the retry window
    _run(identity, policy, {"face#5": STRANGER}, now, now + 3.0)
    assert "face#5" not in _overrides(policy)


def test_blurring_a_remembered_person_forgets_them() -> None:
    identity, _ = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    _owners(identity, {"face#5": STRANGER}, policy, now)
    assert _overrides(policy)["face#5"] == "show"
    policy.set_track_override("face#5", "blur")  # "blur her again"
    now = _run(identity, policy, {"face#5": STRANGER}, now + FRAME_S, now + 0.5)
    now = _run(identity, policy, {}, now, now + 1.0)
    _run(identity, policy, {"face#8": STRANGER}, now, now + 3.0)
    assert "face#8" not in _overrides(policy)


def test_blurring_all_faces_forgets_everyone_in_the_same_update() -> None:
    identity, _ = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    now = _unblur_then_leave(identity, policy, "face#2", OTHER, now)
    policy.set_class_rule("face", "blur")  # bumps face_reset_count and clears face overrides
    _owners(identity, {"face#5": STRANGER, "face#6": OTHER}, policy, now)
    assert _overrides(policy) == {}
    _run(identity, policy, {"face#5": STRANGER, "face#6": OTHER}, now + FRAME_S, now + 3.0)
    assert _overrides(policy) == {}
    now = _unblur_then_leave(identity, policy, "face#7", OTHER, now + 3.0)  # remembering works again
    _owners(identity, {"face#8": OTHER}, policy, now)
    assert _overrides(policy) == {"face#7": "show", "face#8": "show"}


def test_re_enrolling_the_owner_keeps_remembered_people() -> None:
    identity, _ = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    identity.enroll(IMAGE, _row(OWNER))
    _owners(identity, {"face#5": STRANGER}, policy, now)
    assert _overrides(policy)["face#5"] == "show"


def test_id_swap_on_a_re_shown_track_is_revoked_and_the_person_stays_remembered() -> None:
    identity, _ = _identity()
    policy = Policy()
    start = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    _owners(identity, {"face#5": STRANGER}, policy, start)
    assert _overrides(policy)["face#5"] == "show"
    _owners(identity, {"face#5": OTHER}, policy, start + 0.6)  # the tracker hands face#5 to someone else
    first_failure = start + 1.05  # the next scheduled re-check
    for i in range(3):
        assert _overrides(policy)["face#5"] == "show"
        _owners(identity, {"face#5": OTHER}, policy, first_failure + i * FRAME_S)
    assert "face#5" not in _overrides(policy)  # revoked on the third failed frame
    now = _run(identity, policy, {"face#5": OTHER}, first_failure + 3 * FRAME_S, start + 4.0)
    assert "face#5" not in _overrides(policy)  # the person who swapped in is not re-shown
    _run(identity, policy, {"face#5": STRANGER}, now, now + 2.0)
    assert "face#5" not in _overrides(policy)  # a revoked track is not re-shown automatically
    _owners(identity, {"face#5": OTHER, "face#9": STRANGER}, policy, now + 2.0)
    assert _overrides(policy) == {"face#1": "show", "face#9": "show"}  # the real person, on a new track


def test_held_over_boxes_are_never_matched_to_a_remembered_person() -> None:
    identity, embedder = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    calls = len(embedder.calls)
    now = _run(identity, policy, {"face#5": None}, now, now + 1.0)
    assert "face#5" not in _overrides(policy)
    assert len(embedder.calls) == calls
    _owners(identity, {"face#5": STRANGER}, policy, now)  # its first fresh face row
    assert _overrides(policy)["face#5"] == "show"


def test_embedding_errors_never_unblur() -> None:
    identity, _ = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    _run(identity, policy, {"face#5": BROKEN}, now, now + 3.0)
    assert "face#5" not in _overrides(policy)


def test_a_track_with_its_own_blur_override_is_not_re_shown() -> None:
    identity, _ = _identity()
    policy = Policy()
    now = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    policy.set_track_override("face#5", "blur")
    _run(identity, policy, {"face#5": STRANGER}, now, now + 2.0)
    assert _overrides(policy)["face#5"] == "blur"


def test_remembered_people_are_capped_and_the_oldest_is_dropped() -> None:
    people = _RememberedPeople()
    faces = list(np.eye(20, dtype=np.float32))
    for face in faces[:17]:
        people.add(face)
    people.add(faces[16])  # already remembered: not added twice
    assert len(people) == 16
    assert people.match(faces[0]) is None
    assert all(people.match(face) is not None for face in faces[1:17])
    assert people.match(faces[17]) is None


def test_remembered_people_add_no_steady_state_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(config, "ID_RECHECK_S", 1.0)
    times: dict[int, list[float]] = {OWNER: [], STRANGER: [], OTHER: []}
    now = 0.0

    def embed(image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        times[int(face_row[0])].append(now)
        return PEOPLE[int(face_row[0])]

    identity = OwnerIdentity(embed=embed)
    identity.enroll(IMAGE, _row(OWNER))
    policy = Policy()
    start = _unblur_then_leave(identity, policy, "face#1", STRANGER, 0.0)
    for track_times in times.values():
        track_times.clear()
    for i in range(60):  # 4 s at 15 FPS: the owner, a remembered person and a stranger
        now = start + i * FRAME_S
        _owners(identity, {"face#2": OWNER, "face#5": STRANGER, "face#6": OTHER}, policy, now)
    assert _overrides(policy) == {"face#1": "show", "face#5": "show"}
    # The new track that matches nobody remembered is retried for 0.5 s (8 frames), like the owner retry.
    assert len([t for t in times[OTHER] if t < start + 0.5]) == 8
    for track_times in times.values():
        steady = [t for t in track_times if t >= start + 0.5]
        assert 3 <= len(steady) <= 4
        assert all(b - a >= 1.0 - 1e-9 for a, b in zip(steady, steady[1:]))


# --- fail closed ---


def test_embedding_failure_is_not_owner_and_clears_show_override() -> None:
    identity, _ = _identity()
    policy = Policy()
    policy.set_track_override("face#7", "show")
    owners = _owners(identity, {"face#2": BROKEN, "face#7": BROKEN, "face#8": STRANGER}, policy, 0.0)
    assert owners == []
    assert policy.snapshot().track_overrides == {}


def test_owner_whose_rechecks_fail_loses_owner() -> None:
    identity, _ = _identity()
    policy = Policy()
    assert _owners(identity, {"face#2": OWNER}, policy, 0.0) == ["face#2"]
    assert _owners(identity, {"face#2": BROKEN}, policy, 1.0) == ["face#2"]
    assert _owners(identity, {"face#2": BROKEN}, policy, 1.0 + FRAME_S) == ["face#2"]
    assert _owners(identity, {"face#2": BROKEN}, policy, 1.0 + 2 * FRAME_S) == []


def test_zero_embedding_fails_closed() -> None:
    identity = OwnerIdentity(embed=lambda image, row: np.zeros(128, dtype=np.float32))
    with pytest.raises(ValueError):
        identity.enroll(IMAGE, _row(OWNER))
    policy = Policy()
    policy.set_track_override("face#1", "show")
    assert _owners(identity, {"face#1": OWNER}, policy, 0.0) == []
    assert policy.snapshot().track_overrides == {}


# --- one bad frame is retried, not trusted ---


def test_one_bad_frame_keeps_the_owner() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#1": OWNER}, policy, 0.0)
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0) == ["face#1"]  # motion blur at the re-check
    assert _owners(identity, {"face#1": OWNER}, policy, 1.0 + FRAME_S) == ["face#1"]
    now = _run(identity, policy, {"face#1": OWNER}, 1.0 + 2 * FRAME_S, 3.0)
    assert _owners(identity, {"face#1": OWNER}, policy, now) == ["face#1"]


def test_three_bad_frames_in_a_row_revoke_the_owner() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#1": OWNER}, policy, 0.0)
    owners = [_owners(identity, {"face#1": STRANGER}, policy, 1.0 + i * FRAME_S) for i in range(4)]
    assert owners == [["face#1"], ["face#1"], [], []]


def test_bad_good_bad_bad_keeps_the_owner() -> None:
    identity, embedder = _identity()
    policy = Policy()
    _owners(identity, {"face#1": OWNER}, policy, 0.0)
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0) == ["face#1"]  # bad
    assert _owners(identity, {"face#1": OWNER}, policy, 1.1) == ["face#1"]  # good: back to once a second
    calls = len(embedder.calls)
    _run(identity, policy, {"face#1": STRANGER}, 1.2, 2.05)  # not re-checked until 2.1, so no strikes
    assert len(embedder.calls) == calls
    assert _owners(identity, {"face#1": STRANGER}, policy, 2.1) == ["face#1"]  # bad
    assert _owners(identity, {"face#1": STRANGER}, policy, 2.1 + FRAME_S) == ["face#1"]  # bad
    assert _owners(identity, {"face#1": STRANGER}, policy, 2.1 + 2 * FRAME_S) == []  # third in a row


def test_swapped_face_on_a_shown_track_is_revoked_after_three_frames() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#7": STRANGER}, policy, 0.0)
    policy.set_track_override("face#7", "show")
    _owners(identity, {"face#7": STRANGER}, policy, FRAME_S)  # reference saved, last check
    first_failure = 1.2  # the next scheduled re-check
    for i in range(3):
        assert policy.snapshot().track_overrides == {"face#7": "show"}
        _owners(identity, {"face#7": OTHER}, policy, first_failure + i * FRAME_S)
    assert policy.snapshot().track_overrides == {}
    assert 2 * FRAME_S < 0.5  # revoked on the third failed frame, well inside the 0.5 s window
    box = _face("face#7", OTHER)
    identity.update(IMAGE, [box], policy, first_failure + 3 * FRAME_S)
    assert decide(box, policy.snapshot()) == "blur"


def test_one_bad_frame_keeps_a_show_override() -> None:
    identity, embedder = _identity()
    policy = Policy()
    _owners(identity, {"face#7": STRANGER}, policy, 0.0)
    policy.set_track_override("face#7", "show")
    _owners(identity, {"face#7": STRANGER}, policy, FRAME_S)
    calls = len(embedder.calls)
    _owners(identity, {"face#7": OTHER}, policy, 1.2)  # turned head at the re-check
    assert len(embedder.calls) == calls + 1
    _run(identity, policy, {"face#7": STRANGER}, 1.2 + FRAME_S, 4.0)
    assert policy.snapshot().track_overrides == {"face#7": "show"}


def test_new_track_that_misses_first_and_matches_next_frame_is_owner_then() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#1": OWNER}, policy, 0.0)
    _owners(identity, {}, policy, 0.5)  # the tracker lost the owner
    assert _owners(identity, {"face#2": STRANGER}, policy, 0.6) == []  # blurry first frame of the new id
    assert _owners(identity, {"face#2": OWNER}, policy, 0.6 + FRAME_S) == ["face#2"]


def test_new_track_retries_for_half_a_second_then_once_a_second() -> None:
    identity, embedder = _identity()
    policy = Policy()
    now = _run(identity, policy, {"face#2": STRANGER}, 0.0, 0.5)
    assert len(embedder.calls) == 8  # every frame inside the retry window
    _run(identity, policy, {"face#2": STRANGER}, now, 1.0)
    assert len(embedder.calls) == 8  # back to once a second


def test_held_over_boxes_neither_add_strikes_nor_rescue_a_failing_track() -> None:
    identity, embedder = _identity()
    policy = Policy()
    _owners(identity, {"face#1": OWNER}, policy, 0.0)
    policy.set_track_override("face#1", "show")
    _owners(identity, {"face#1": OWNER}, policy, FRAME_S)  # reference saved
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0) == ["face#1"]  # strike 1
    calls = len(embedder.calls)
    for i in range(1, 7):  # held over until 1.4: not embedded, no strikes
        assert _owners(identity, {"face#1": None}, policy, 1.0 + i * FRAME_S) == ["face#1"]
    assert len(embedder.calls) == calls
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0 + 7 * FRAME_S) == ["face#1"]  # strike 2
    assert policy.snapshot().track_overrides == {"face#1": "show"}
    # Held over again past 0.5 s after the first failure: revoked without a third check.
    assert _owners(identity, {"face#1": None}, policy, 1.0 + 8 * FRAME_S) == []
    assert policy.snapshot().track_overrides == {}


def test_steady_state_embeds_each_track_at_most_once_per_second(monkeypatch) -> None:
    monkeypatch.setattr(config, "ID_RECHECK_S", 1.0)
    times: dict[int, list[float]] = {OWNER: [], STRANGER: [], OTHER: []}
    now = 0.0

    def embed(image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        times[int(face_row[0])].append(now)
        return PEOPLE[int(face_row[0])]

    identity = OwnerIdentity(embed=embed)
    identity.enroll(IMAGE, _row(OWNER))
    times[OWNER].clear()
    policy = Policy()
    policy.set_track_override("face#3", "show")
    for i in range(60):  # 4 s at 15 FPS, with the owner in view
        now = i * FRAME_S
        _owners(identity, {"face#1": OWNER, "face#2": STRANGER, "face#3": OTHER}, policy, now)
    for track_times in times.values():
        assert 4 <= len(track_times) <= 5
        assert all(b - a >= 1.0 - 1e-9 for a, b in zip(track_times, track_times[1:]))


# --- real SFace ---

CLIPS_DIR = Path("/Users/luiz/Developer/privacy-cam/tests/data")
CROSSING_CLIP = CLIPS_DIR / "crossing.mov"
FACES_CLIP = CLIPS_DIR / "faces.mov"

needs_models = pytest.mark.skipif(
    not (config.SFACE_MODEL.exists() and config.YUNET_MODEL.exists()), reason="model weights missing"
)


@needs_models
def test_unit_dot_product_equals_sface_cosine_match() -> None:
    from vision.identity import _cosine, _unit

    recognizer = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL), "")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(1, 128)).astype(np.float32), rng.normal(size=(1, 128)).astype(np.float32)
    expected = recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)
    assert _cosine(_unit(a), _unit(b)) == pytest.approx(expected, abs=1e-5)


def _clip_faces(path: Path, step: int = 3, max_frames: int = 600) -> list[tuple[np.ndarray, np.ndarray]]:
    """(frame, YuNet rows sorted by box area, largest first) for every step-th frame with a face."""
    if not path.exists():
        pytest.skip(f"clip missing: {path}")
    from fakes.capture import FileSource

    detector = cv2.FaceDetectorYN.create(
        str(config.YUNET_MODEL), "", (config.FRAME_WIDTH, config.FRAME_HEIGHT), config.FACE_SCORE
    )
    source = FileSource(path, loop=False)
    frames = []
    try:
        for index in range(max_frames):
            image = source.read()
            if image is None:
                break
            if index % step:
                continue
            _, rows = detector.detect(image)
            if rows is not None and len(rows):
                rows = rows[np.argsort(-(rows[:, 2] * rows[:, 3]))]
                frames.append((image, rows))
    finally:
        source.release()
    return frames


def _sface() -> OwnerIdentity:
    return OwnerIdentity()


@needs_models
@pytest.mark.parametrize("clip", [CROSSING_CLIP], ids=["crossing"])  # in faces.mov nobody stays in frame
def test_clip_same_person_matches_across_frames(clip: Path) -> None:
    frames = _clip_faces(clip)
    # confident faces only: a half-visible face at the frame edge is not a fair identity sample
    frames = [(image, rows[rows[:, 14] >= 0.8]) for image, rows in frames]
    frames = [(image, rows) for image, rows in frames if len(rows)]
    assert len(frames) >= 2
    identity = _sface()
    first_image, first_rows = frames[0]
    later_image, later_rows = frames[min(5, len(frames) - 1)]
    first_row = first_rows[0]
    # rows are not ordered by person: follow the first face by position
    later_row = min(later_rows, key=lambda row: abs(row[0] - first_row[0]) + abs(row[1] - first_row[1]))
    identity.enroll(first_image, first_row)
    box = TrackedBox(id="face#1", label="face", xyxy=(0, 0, 1, 1), conf=1.0, frame_index=0, last_seen=0.0,
                     face_row=later_row)
    identity.update(later_image, [box], Policy(), 0.0)
    assert box.is_owner


def _two_people_frame(clip: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for image, rows in _clip_faces(clip):
        if len(rows) >= 2 and rows[1][14] >= 0.8:
            return image, rows[0], rows[1]
    pytest.skip(f"no frame with two confident faces in {clip.name}")


@needs_models
@pytest.mark.parametrize("clip", [CROSSING_CLIP, FACES_CLIP], ids=["crossing", "faces"])
def test_clip_two_different_people_do_not_match(clip: Path) -> None:
    image, person_a, person_b = _two_people_frame(clip)
    identity = _sface()
    identity.enroll(image, person_a)
    box = TrackedBox(id="face#1", label="face", xyxy=(0, 0, 1, 1), conf=1.0, frame_index=0, last_seen=0.0,
                     face_row=person_b)
    identity.update(image, [box], Policy(), 0.0)
    assert not box.is_owner


@needs_models
@pytest.mark.parametrize("clip", [CROSSING_CLIP, FACES_CLIP], ids=["crossing", "faces"])
def test_clip_swapped_face_clears_the_override(clip: Path) -> None:
    image, person_a, person_b = _two_people_frame(clip)
    identity = _sface()
    policy = Policy()

    def frame(row: np.ndarray, now: float) -> None:
        box = TrackedBox(id="face#7", label="face", xyxy=(0, 0, 1, 1), conf=1.0, frame_index=0, last_seen=now,
                         face_row=row)
        identity.update(image, [box], policy, now)

    frame(person_a, 0.0)
    policy.set_track_override("face#7", "show")
    frame(person_a, 0.1)
    frame(person_a, 1.0)
    assert policy.snapshot().track_overrides == {"face#7": "show"}
    frame(person_b, 1.5)
    assert policy.snapshot().track_overrides == {"face#7": "show"}  # next re-check is at 2.0
    frame(person_b, 2.0)
    frame(person_b, 2.0 + 1 / 15)
    assert policy.snapshot().track_overrides == {"face#7": "show"}  # two strikes so far
    frame(person_b, 2.0 + 2 / 15)
    assert policy.snapshot().track_overrides == {}


@needs_models
def test_clip_remembered_person_is_shown_under_a_new_track_id() -> None:
    frames = _clip_faces(CROSSING_CLIP)
    frames = [(image, rows[rows[:, 14] >= 0.8]) for image, rows in frames]
    frames = [(image, rows) for image, rows in frames if len(rows) >= 2]
    if len(frames) < 2:
        pytest.skip("crossing.mov has fewer than two frames with two confident faces")
    first_image, first_rows = frames[0]
    later_image, later_rows = frames[-1]
    # rows are not ordered by person: follow the first face by position through every frame
    person = first_rows[0]
    for _, rows in frames[1:]:
        nearest = int(np.argmin(np.abs(rows[:, 0] - person[0]) + np.abs(rows[:, 1] - person[1])))
        person = rows[nearest]
    other = later_rows[1 if nearest == 0 else 0]
    identity = _sface()
    policy = Policy()

    def frame(image: np.ndarray, rows: dict[str, np.ndarray], now: float) -> None:
        boxes = [TrackedBox(id=track_id, label="face", xyxy=(0, 0, 1, 1), conf=1.0, frame_index=0, last_seen=now,
                            face_row=row) for track_id, row in rows.items()]
        identity.update(image, boxes, policy, now)

    frame(first_image, {"face#1": first_rows[0]}, 0.0)
    policy.set_track_override("face#1", "show")
    frame(first_image, {"face#1": first_rows[0]}, 0.1)  # unblurred: remembered
    frame(later_image, {}, 1.0)  # left the frame
    frame(later_image, {"face#5": person, "face#6": other}, 2.0)  # several seconds later in the clip
    assert dict(policy.snapshot().track_overrides) == {"face#1": "show", "face#5": "show"}


@needs_models
def test_measure_sface_ms_per_face() -> None:
    """Prints alignCrop + feature time; run with -s to see it."""
    clip = next((c for c in (CROSSING_CLIP, FACES_CLIP) if c.exists()), None)
    if clip is None:
        pytest.skip("no clip to measure on")
    image, rows = _clip_faces(clip)[0]
    recognizer = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL), "")
    recognizer.feature(recognizer.alignCrop(image, rows[0]))  # warm up
    runs = 50
    start = time.perf_counter()
    for _ in range(runs):
        recognizer.feature(recognizer.alignCrop(image, rows[0]))
    ms = (time.perf_counter() - start) / runs * 1000
    print(f"\nSFace alignCrop + feature: {ms:.2f} ms per face")
    assert ms < 1000
