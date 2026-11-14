"""WP5: the object tracker. Synthetic boxes only, fed at YOLOE's low, irregular rate."""

from __future__ import annotations

import pytest

from contracts import OBJECT_CLASSES, Detection
from vision.tracking import make_object_tracker

# Gaps between finished YOLOE results, in seconds: between 15 and 3 per second.
_IRREGULAR_GAPS = [0.07, 0.3, 0.12, 0.25, 0.08, 0.33, 0.1, 0.2] * 3


def plate(x: int, y: int, conf: float = 0.4, label: str = "license plate") -> Detection:
    return Detection(label=label, xyxy=(x, y, x + 90, y + 30), conf=conf)


def test_every_detection_comes_back_with_an_obj_id() -> None:
    tracker = make_object_tracker()
    detections = [plate(10, 10, conf=0.25), plate(300, 200, conf=0.9, label="street sign")]
    boxes = tracker.update(detections, frame_index=3, now=1.0)
    assert [(box.label, box.xyxy, box.conf) for box in boxes] == [(d.label, d.xyxy, pytest.approx(d.conf)) for d in detections]
    assert all(box.id.startswith("obj#") and box.label in OBJECT_CLASSES for box in boxes)
    assert all(box.frame_index == 3 and box.last_seen == 1.0 for box in boxes)
    assert len({box.id for box in boxes}) == 2


def test_weak_detections_at_the_conf_floor_are_never_dropped() -> None:
    tracker = make_object_tracker()
    for i in range(10):
        detections = [plate(50 + 3 * i, 50, conf=0.25), plate(400, 300, conf=0.26, label="house number")]
        assert len(tracker.update(detections, frame_index=i, now=i * 0.2)) == 2


def test_a_slowly_moving_box_keeps_its_id_at_an_irregular_rate() -> None:
    tracker = make_object_tracker()
    ids, now, x = [], 0.0, 100
    for i, gap in enumerate(_IRREGULAR_GAPS):
        now += gap
        x += round(40 * gap)  # 40 px per second
        ids.append(tracker.update([plate(x, 200)], frame_index=i, now=now)[0].id)
    assert len(set(ids[1:])) == 1  # the first ID is provisional; stable once confirmed


def test_a_missed_result_does_not_change_the_id() -> None:
    tracker = make_object_tracker()
    ids = [tracker.update([plate(100 + i, 100)], frame_index=i, now=i * 0.1)[0].id for i in range(4)]
    tracker.update([], frame_index=4, now=0.4)  # YOLOE missed it once
    ids.append(tracker.update([plate(105, 100)], frame_index=5, now=0.5)[0].id)
    assert len(set(ids[1:])) == 1


def test_ids_are_never_reused() -> None:
    tracker = make_object_tracker()
    seen: list[str] = []
    for round_ in range(3):
        for i in range(4):
            boxes = tracker.update([plate(100 + 200 * round_, 100)], frame_index=10 * round_ + i, now=round_ * 10 + i * 0.1)
            seen.extend(box.id for box in boxes)
        for i in range(30):  # gone long enough for the track to be retired
            tracker.update([], frame_index=10 * round_ + 4 + i, now=round_ * 10 + 1 + i * 0.2)
    first_ids = [seen[4 * r] for r in range(3)]
    stable_ids = [seen[4 * r + 1] for r in range(3)]
    assert len(set(first_ids + stable_ids)) == 6
    numbers = [int(track_id.split("#")[1]) for track_id in dict.fromkeys(seen)]
    assert numbers == sorted(numbers)
