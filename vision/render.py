"""Pixelation, padding, hold-over, backfill and the watchdog (WP2).

The only module besides pipeline.fail_closed_frame that builds a CensoredFrame.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import config
from contracts import Action, BufferedFrame, CensoredFrame, PolicySnapshot, TrackedBox

Decide = Callable[[TrackedBox, PolicySnapshot], Action]


def pad_box(xyxy: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Grow the box by config.BOX_PADDING per side, clipped to the frame."""
    raise NotImplementedError("WP2")


def pixelate_full(image: np.ndarray) -> np.ndarray:
    """A pixelated copy of the whole frame, config.PIXEL_BLOCK blocks."""
    raise NotImplementedError("WP2")


def pixelate_boxes(image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """A copy of the frame with every (already padded) box pixelated."""
    raise NotImplementedError("WP2")


def backfill(earlier_frames: Sequence[BufferedFrame], new_boxes: Sequence[TrackedBox]) -> None:
    """Add each newly found track's box to the buffered frames before it (Never-leaks rule 4)."""
    raise NotImplementedError("WP2")


class HoldOver:
    """Keeps a lost track's last box alive for config.HOLD_OVER_S (Never-leaks rule 3)."""

    def __init__(self, hold_s: float = config.HOLD_OVER_S) -> None:
        raise NotImplementedError("WP2")

    def apply(self, boxes: list[TrackedBox], now: float) -> list[TrackedBox]:
        """This frame's boxes plus the last box of every track lost less than hold_s ago."""
        raise NotImplementedError("WP2")


class Watchdog:
    """Trips when detections stop arriving (Never-leaks rule 5). Times are time.monotonic()."""

    def __init__(
        self,
        face_limit_s: float = config.FACE_WATCHDOG_S,
        object_limit_s: float = config.OBJECT_WATCHDOG_S,
        watch_objects: bool = True,
    ) -> None:
        raise NotImplementedError("WP2")

    def beat_faces(self, now: float) -> None:
        raise NotImplementedError("WP2")

    def beat_objects(self, now: float) -> None:
        raise NotImplementedError("WP2")

    def tripped(self, now: float) -> bool:
        """True before the first beat and whenever either detector is overdue."""
        raise NotImplementedError("WP2")


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
    raise NotImplementedError("WP2")
