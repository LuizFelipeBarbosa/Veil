"""Pixelation, padding, hold-over, backfill and the watchdog (WP2).

The only module besides pipeline.fail_closed_frame that builds a CensoredFrame.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Sequence

import cv2
import numpy as np

import config
from contracts import Action, BufferedFrame, CensoredFrame, PolicySnapshot, TrackedBox

Decide = Callable[[TrackedBox, PolicySnapshot], Action]


def pad_box(xyxy: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Grow the box by config.BOX_PADDING per side, clipped to the frame."""
    x1, y1, x2, y2 = xyxy
    pad_x = (x2 - x1) * config.BOX_PADDING
    pad_y = (y2 - y1) * config.BOX_PADDING
    return (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    )


def pixelate_full(image: np.ndarray) -> np.ndarray:
    """A pixelated copy of the whole frame, config.PIXEL_BLOCK blocks."""
    block = config.PIXEL_BLOCK
    height, width = image.shape[:2]
    # Grow the frame to whole blocks so every block is exactly block x block and grid-aligned.
    grown = cv2.copyMakeBorder(
        image, 0, -height % block, 0, -width % block, cv2.BORDER_REPLICATE
    )
    grown_height, grown_width = grown.shape[:2]
    small = cv2.resize(grown, (grown_width // block, grown_height // block), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (grown_width, grown_height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(pixelated[:height, :width])


def pixelate_boxes(image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """A copy of the frame with every (already padded) box pixelated."""
    censored = image.copy()
    if not boxes:
        return censored
    # Copy each box out of one pixelated frame: one resize however many boxes there are,
    # and blocks stay on the frame-wide grid.
    pixelated = pixelate_full(image)
    for x1, y1, x2, y2 in boxes:
        censored[y1:y2, x1:x2] = pixelated[y1:y2, x1:x2]
    return censored


def backfill(earlier_frames: Sequence[BufferedFrame], new_boxes: Sequence[TrackedBox]) -> None:
    """Add each newly found track's box to the buffered frames before it (Never-leaks rule 4)."""
    for frame in earlier_frames:
        ids_on_frame = {box.id for box in frame.boxes}
        for box in new_boxes:
            if box.id not in ids_on_frame:
                frame.boxes.append(dataclasses.replace(box, frame_index=frame.index))


class HoldOver:
    """Keeps a lost track's last box alive for config.HOLD_OVER_S (Never-leaks rule 3)."""

    def __init__(self, hold_s: float = config.HOLD_OVER_S) -> None:
        self._hold_s = hold_s
        self._last_box: dict[str, TrackedBox] = {}

    def apply(self, boxes: list[TrackedBox], now: float, frame_index: int | None = None) -> list[TrackedBox]:
        """This frame's boxes plus the last box of every track lost less than hold_s ago.

        A track exactly hold_s old is still held (fail closed). When frame_index is
        given, held boxes are re-labelled with it, since they join that frame's list.
        """
        for box in boxes:
            self._last_box[box.id] = box
        self._last_box = {
            track_id: box for track_id, box in self._last_box.items() if now - box.last_seen <= self._hold_s
        }

        current_ids = {box.id for box in boxes}
        held = [box for track_id, box in self._last_box.items() if track_id not in current_ids]
        if frame_index is not None:
            held = [dataclasses.replace(box, frame_index=frame_index) for box in held]
        return [*boxes, *held]


class Watchdog:
    """Trips when detections stop arriving (Never-leaks rule 5). Times are time.monotonic()."""

    def __init__(
        self,
        face_limit_s: float = config.FACE_WATCHDOG_S,
        object_limit_s: float = config.OBJECT_WATCHDOG_S,
        watch_objects: bool = True,
    ) -> None:
        self._face_limit_s = face_limit_s
        self._object_limit_s = object_limit_s
        self._watch_objects = watch_objects
        self._last_faces: float | None = None
        self._last_objects: float | None = None

    def beat_faces(self, now: float) -> None:
        self._last_faces = now

    def beat_objects(self, now: float) -> None:
        self._last_objects = now

    def tripped(self, now: float) -> bool:
        """True before the first beat and whenever either detector is overdue."""
        if _overdue(self._last_faces, self._face_limit_s, now):
            return True
        return self._watch_objects and _overdue(self._last_objects, self._object_limit_s, now)


def _overdue(last_beat: float | None, limit_s: float, now: float) -> bool:
    return last_beat is None or now - last_beat > limit_s


def censor(
    frame: BufferedFrame,
    policy: PolicySnapshot,
    decide: Decide,
    watchdog_tripped: bool,
) -> CensoredFrame:
    """Turn one buffered frame into the frame that may be shown and recorded.

    Whole-frame pixelation when panic is on, the watchdog tripped or the frame's
    face detection never finished; otherwise every box that decide() blurs is
    padded and pixelated, using this frame's own boxes only.
    """
    if policy.panic or watchdog_tripped or not frame.faces_done:
        return CensoredFrame(image=pixelate_full(frame.image), index=frame.index, fail_closed=True)

    height, width = frame.image.shape[:2]
    # Anything other than an explicit "show" blurs.
    blurred = [pad_box(box.xyxy, width, height) for box in frame.boxes if decide(box, policy) != "show"]
    return CensoredFrame(image=pixelate_boxes(frame.image, blurred), index=frame.index, fail_closed=False)
