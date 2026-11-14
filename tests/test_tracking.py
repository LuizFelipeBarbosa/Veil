"""WP2: the face tracker. Synthetic boxes only."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from contracts import FACE, Detection
from vision.tracking import make_face_tracker, make_object_tracker


def face(x: int, y: int, size: int = 80, conf: float = 0.9) -> Detection:
    row = np.full(15, float(x), dtype=np.float32)
    return Detection(label=FACE, xyxy=(x, y, x + size, y + size), conf=conf, face_row=row)


def test_every_detection_comes_back_even_before_confirmation() -> None:
    tracker = make_face_tracker()
    detections = [face(10, 10), face(300, 200, conf=0.6)]
    boxes = tracker.update(detections, frame_index=0, now=1.0)
    assert [box.xyxy for box in boxes] == [d.xyxy for d in detections]
    assert all(box.id.startswith("face#") for box in boxes)
    assert len({box.id for box in boxes}) == 2


def test_fields_are_carried_through() -> None:
    tracker = make_face_tracker()
    detection = face(10, 10, conf=0.75)
    (box,) = tracker.update([detection], frame_index=42, now=12.5)
    assert box.label == FACE
    assert box.conf == pytest.approx(0.75)
    assert box.frame_index == 42 and box.last_seen == 12.5
    assert box.face_row is detection.face_row


def test_a_steady_face_keeps_its_id() -> None:
    tracker = make_face_tracker()
    ids = [tracker.update([face(100 + 2 * i, 100)], frame_index=i, now=i / 15)[0].id for i in range(20)]
    assert len(set(ids[1:])) == 1  # confirmed on the second frame, stable after that
    assert ids[0] != ids[1]  # the first frame's ID was provisional


def test_two_faces_keep_distinct_ids_whatever_the_input_order() -> None:
    tracker = make_face_tracker()
    for i in range(5):
        left, right = face(50 + i, 100), face(400 - i, 100)
        detections = [left, right] if i % 2 else [right, left]
        boxes = tracker.update(detections, frame_index=i, now=i / 15)
        by_x = {box.xyxy[0]: box.id for box in boxes}
        assert [box.xyxy for box in boxes] == [d.xyxy for d in detections]
    assert by_x[54] != by_x[396]


def test_ids_are_never_reused_and_a_returning_face_is_new() -> None:
    tracker = make_face_tracker()
    seen: list[str] = []
    for i in range(5):
        seen += [box.id for box in tracker.update([face(100, 100)], frame_index=i, now=i / 15)]
    first_track = seen[-1]
    for i in range(5, 40):  # gone for more than a second: the track expires
        assert tracker.update([], frame_index=i, now=i / 15) == []
    returned = tracker.update([face(100, 100)], frame_index=40, now=40 / 15)[0].id
    assert returned != first_track and returned not in seen


class _UnconfirmingByteTrack:
    """Stands in for ByteTrack: returns the first detection unconfirmed and drops the rest."""

    def update(self, detections: sv.Detections) -> sv.Detections:
        kept = detections[:1]
        kept.tracker_id = np.full(len(kept), -1, dtype=int)
        return kept


def test_unconfirmed_and_dropped_detections_get_fresh_provisional_ids() -> None:
    tracker = make_face_tracker()
    tracker._bytetrack = _UnconfirmingByteTrack()
    ids = []
    for i in range(3):
        boxes = tracker.update([face(0, 0), face(200, 0), face(400, 0)], frame_index=i, now=i / 15)
        assert [b.xyxy[0] for b in boxes] == [0, 200, 400]
        ids += [b.id for b in boxes]
    assert len(ids) == len(set(ids)) == 9


def test_object_tracker_belongs_to_wp5() -> None:
    with pytest.raises(NotImplementedError):
        make_object_tracker()
