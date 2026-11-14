"""ByteTrack IDs (WP2 faces, WP5 objects). One tracker instance per kind."""

from __future__ import annotations

import itertools

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

import config
from contracts import FACE_ID_PREFIX, Detection, TrackedBox

_UNCONFIRMED = -1


class Tracker:
    """Wraps one trackers.ByteTrackTracker and names its tracks "<id_prefix>#<n>"."""

    def __init__(
        self,
        id_prefix: str,
        *,
        min_conf: float = config.FACE_SCORE,
        frame_rate: float = config.FPS,
        lost_track_s: float = 1.0,
    ) -> None:
        self._id_prefix = id_prefix
        # Every detection the detector passes is confident enough to start a track and
        # to count as "high" in ByteTrack's first matching stage. lost_track_buffer is
        # counted in 30 FPS frames whatever the real frame rate is.
        self._bytetrack = ByteTrackTracker(
            lost_track_buffer=round(lost_track_s * 30),
            frame_rate=float(frame_rate),
            track_activation_threshold=min_conf,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=min_conf,
        )
        self._numbers = itertools.count(1)  # shared by confirmed and provisional IDs, never reused
        self._names: dict[int, str] = {}  # ByteTrack tracker_id -> our ID

    def update(self, detections: list[Detection], frame_index: int, now: float) -> list[TrackedBox]:
        """Tracked boxes for this frame. IDs are never reused, so a returning person is a new track.

        A detection the tracker has not confirmed yet still comes back, under a
        fresh provisional ID, so no detected face is ever dropped.
        """
        tracker_ids = self._tracker_ids_by_detection(detections)
        return [
            TrackedBox(
                id=self._name(tracker_ids.get(i, _UNCONFIRMED)),
                label=detection.label,
                xyxy=detection.xyxy,
                conf=detection.conf,
                frame_index=frame_index,
                last_seen=now,
                face_row=detection.face_row,
            )
            for i, detection in enumerate(detections)
        ]

    def _tracker_ids_by_detection(self, detections: list[Detection]) -> dict[int, int]:
        """ByteTrack's tracker_id for each detection, keyed by its index in `detections`.

        ByteTrack may reorder its output, so each detection carries its own index through it.
        A detection missing from the output is simply absent here and gets a provisional ID.
        """
        if detections:
            xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
            confidence = np.array([d.conf for d in detections], dtype=np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
        indices = np.arange(len(detections))
        tracked = self._bytetrack.update(sv.Detections(xyxy=xyxy, confidence=confidence, data={"index": indices}))
        if tracked.tracker_id is None or len(tracked) == 0:
            return {}
        return {int(index): int(tracker_id) for index, tracker_id in zip(tracked.data["index"], tracked.tracker_id)}

    def _name(self, tracker_id: int) -> str:
        if tracker_id == _UNCONFIRMED:
            return self._new_name()
        if tracker_id not in self._names:
            self._names[tracker_id] = self._new_name()
        return self._names[tracker_id]

    def _new_name(self) -> str:
        return f"{self._id_prefix}#{next(self._numbers)}"


def make_face_tracker() -> Tracker:
    return Tracker(FACE_ID_PREFIX, min_conf=config.FACE_SCORE, frame_rate=config.FPS)


def make_object_tracker() -> Tracker:
    raise NotImplementedError("WP5")
