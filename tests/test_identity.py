"""WP6: owner match and ID-swap check.

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
from vision.identity import OwnerIdentity
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


def test_owner_that_stops_matching_loses_it_at_the_next_recheck() -> None:
    identity, _ = _identity()
    policy = Policy()
    assert _owners(identity, {"face#1": OWNER}, policy, 0.0) == ["face#1"]
    assert _owners(identity, {"face#1": STRANGER}, policy, 0.5) == ["face#1"]  # cached until the re-check
    assert _owners(identity, {"face#1": STRANGER}, policy, 1.0) == []


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
    _owners(identity, {"face#7": STRANGER}, policy, 0.1)  # reference saved
    swap_at = 0.3
    now = _run(identity, policy, {"face#7": OTHER}, swap_at, 1.0)  # the tracker hands face#7 to someone else
    assert policy.snapshot().track_overrides == {"face#7": "show"}  # not before the re-check
    _owners(identity, {"face#7": OTHER}, policy, now)  # first frame at or after t=1.0
    assert policy.snapshot().track_overrides == {}
    assert now - swap_at <= config.ID_RECHECK_S
    box = _face("face#7", OTHER)
    identity.update(IMAGE, [box], policy, now + FRAME_S)
    assert decide(box, policy.snapshot()) == "blur"


def test_returning_person_under_a_new_track_id_is_blurred() -> None:
    identity, _ = _identity()
    policy = Policy()
    _owners(identity, {"face#7": STRANGER}, policy, 0.0)
    policy.set_track_override("face#7", "show")
    now = _run(identity, policy, {"face#7": STRANGER}, FRAME_S, 1.0)
    now = _run(identity, policy, {}, now, 3.0)  # left the frame
    box = _face("face#9", STRANGER)
    identity.update(IMAGE, [box], policy, now)
    assert box.is_owner is False
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
    _run(identity, policy, {"face#2": STRANGER}, now, 1.1)
    assert policy.snapshot().track_overrides == {}


def test_object_overrides_are_untouched() -> None:
    identity, _ = _identity()
    policy = Policy()
    policy.set_track_override("obj#3", "show")
    _run(identity, policy, {"face#1": STRANGER}, 0.0, 2.0)
    assert policy.snapshot().track_overrides == {"obj#3": "show"}


# --- fail closed ---


def test_embedding_failure_is_not_owner_and_clears_show_override() -> None:
    identity, _ = _identity()
    policy = Policy()
    policy.set_track_override("face#7", "show")
    owners = _owners(identity, {"face#2": BROKEN, "face#7": BROKEN, "face#8": STRANGER}, policy, 0.0)
    assert owners == []
    assert policy.snapshot().track_overrides == {}


def test_owner_whose_recheck_fails_loses_owner() -> None:
    identity, _ = _identity()
    policy = Policy()
    assert _owners(identity, {"face#2": OWNER}, policy, 0.0) == ["face#2"]
    assert _owners(identity, {"face#2": BROKEN}, policy, 1.0) == []


def test_zero_embedding_fails_closed() -> None:
    identity = OwnerIdentity(embed=lambda image, row: np.zeros(128, dtype=np.float32))
    with pytest.raises(ValueError):
        identity.enroll(IMAGE, _row(OWNER))
    policy = Policy()
    policy.set_track_override("face#1", "show")
    assert _owners(identity, {"face#1": OWNER}, policy, 0.0) == []
    assert policy.snapshot().track_overrides == {}


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
@pytest.mark.parametrize("clip", [CROSSING_CLIP, FACES_CLIP], ids=["crossing", "faces"])
def test_clip_same_person_matches_across_frames(clip: Path) -> None:
    frames = _clip_faces(clip)
    largest = [(image, rows[0]) for image, rows in frames]
    assert len(largest) >= 2
    identity = _sface()
    (first_image, first_row), (later_image, later_row) = largest[0], largest[min(5, len(largest) - 1)]
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
    assert policy.snapshot().track_overrides == {}


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
