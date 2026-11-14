"""Camera capture and the delay ring buffer (WP1).

Each buffered frame carries its own box list (Never-leaks rule 2).
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

import config
from contracts import BufferedFrame


class FrameSource(Protocol):
    """A camera or a clip. fakes.capture.FileSource is the offline one."""

    def read(self) -> np.ndarray | None:
        """Next BGR frame at config.FRAME_WIDTH x config.FRAME_HEIGHT, or None when the source ends."""
        ...

    def release(self) -> None: ...


def open_source(source: str | None) -> FrameSource:
    """A clip path gives a file source; None opens the camera at config.CAMERA_INDEX."""
    raise NotImplementedError("WP1")


class RingBuffer:
    """Thread-safe delay buffer of raw frames. Raw frames never leave it except to vision code."""

    def __init__(self, capacity: int = config.DELAY_BUFFER_FRAMES) -> None:
        raise NotImplementedError("WP1")

    def push(self, image: np.ndarray, timestamp: float) -> BufferedFrame:
        """Wrap a new raw frame with the next index and an empty box list, and append it."""
        raise NotImplementedError("WP1")

    def pop_oldest(self) -> BufferedFrame | None:
        """Remove and return the oldest frame once the buffer holds `capacity` frames, else None."""
        raise NotImplementedError("WP1")

    def newest(self) -> BufferedFrame | None:
        raise NotImplementedError("WP1")

    def recent(self, count: int) -> list[BufferedFrame]:
        """The newest `count` frames still buffered, oldest first (used for backfill)."""
        raise NotImplementedError("WP1")


class CaptureThread(threading.Thread):
    """Grabs frames at config.FPS into the ring buffer and wakes the vision loop."""

    def __init__(self, source: FrameSource, ring: RingBuffer) -> None:
        raise NotImplementedError("WP1")

    def wait_for_frame(self, after_index: int, timeout: float) -> BufferedFrame | None:
        """Block until a frame newer than after_index is buffered; None on timeout or end of source."""
        raise NotImplementedError("WP1")

    def stop(self) -> None:
        raise NotImplementedError("WP1")
