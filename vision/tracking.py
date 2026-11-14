"""ByteTrack IDs (WP2 faces, WP5 objects). One tracker instance per kind."""

from __future__ import annotations

from contracts import Detection, TrackedBox


class Tracker:
    """Wraps one trackers.ByteTrackTracker and names its tracks "<id_prefix>#<n>"."""

    def __init__(self, id_prefix: str) -> None:
        raise NotImplementedError("WP2")

    def update(self, detections: list[Detection], frame_index: int, now: float) -> list[TrackedBox]:
        """Tracked boxes for this frame. IDs are never reused, so a returning person is a new track.

        A detection the tracker has not confirmed yet still comes back, under a
        fresh provisional ID, so no detected face is ever dropped.
        """
        raise NotImplementedError("WP2")


def make_face_tracker() -> Tracker:
    raise NotImplementedError("WP2")


def make_object_tracker() -> Tracker:
    raise NotImplementedError("WP5")
