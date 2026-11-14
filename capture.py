"""Camera capture and the delay ring buffer (WP1).

Each buffered frame carries its own box list (Never-leaks rule 2).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Protocol

import cv2
import numpy as np

import config
from contracts import BufferedFrame
from fakes.capture import FileSource, cover

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    """A camera or a clip. fakes.capture.FileSource is the offline one."""

    def read(self) -> np.ndarray | None:
        """Next BGR frame at config.FRAME_WIDTH x config.FRAME_HEIGHT, or None when the source ends."""
        ...

    def release(self) -> None: ...


def open_source(source: str | None) -> FrameSource:
    """A clip path gives a file source; None opens the camera at config.CAMERA_INDEX."""
    if source is not None:
        return FileSource(source)
    return _CameraSource(config.CAMERA_INDEX)


class _CameraSource:
    """The live camera, asked for the config size and FPS and resized if it ignores the request."""

    def __init__(self, index: int) -> None:
        self._video = cv2.VideoCapture(index)
        if not self._video.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._video.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._video.set(cv2.CAP_PROP_FPS, config.FPS)

    def read(self) -> np.ndarray | None:
        ok, image = self._video.read()
        if not ok or image is None:
            return None
        # A 16:9 camera is cropped to the frame's shape, never stretched: stretched faces detect badly.
        return cover(image, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    def release(self) -> None:
        self._video.release()


class RingBuffer:
    """Thread-safe delay buffer of raw frames. Raw frames never leave it except to vision code.

    It holds at most 2 x capacity frames: if the vision loop falls that far behind, the oldest
    raw frames are dropped unseen, which is safe (they are never shown) and bounds memory.
    """

    def __init__(self, capacity: int = config.DELAY_BUFFER_FRAMES) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._frames: deque[BufferedFrame] = deque(maxlen=2 * capacity)
        self._next_index = 0
        self._changed = threading.Condition()

    def push(self, image: np.ndarray, timestamp: float) -> BufferedFrame:
        """Wrap a new raw frame with the next index and an empty box list, and append it."""
        with self._changed:
            frame = BufferedFrame(index=self._next_index, timestamp=timestamp, image=image)
            self._next_index += 1
            self._frames.append(frame)
            self._changed.notify_all()
            return frame

    def pop_oldest(self) -> BufferedFrame | None:
        """Remove and return the oldest frame once the buffer holds `capacity` frames, else None."""
        with self._changed:
            if len(self._frames) < self._capacity:
                return None
            return self._frames.popleft()

    def newest(self) -> BufferedFrame | None:
        with self._changed:
            return self._frames[-1] if self._frames else None

    def recent(self, count: int) -> list[BufferedFrame]:
        """The newest `count` frames still buffered, oldest first (used for backfill)."""
        if count <= 0:
            return []
        with self._changed:
            return list(self._frames)[-count:]

    def __len__(self) -> int:
        with self._changed:
            return len(self._frames)

    def _wait_for_frame_after(
        self, after_index: int, timeout: float, ended: Callable[[], bool]
    ) -> BufferedFrame | None:
        """The oldest buffered frame newer than after_index, waiting up to timeout for one."""
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                frame = next((f for f in self._frames if f.index > after_index), None)
                if frame is not None:
                    return frame
                remaining = deadline - time.monotonic()
                if ended() or remaining <= 0:
                    return None
                self._changed.wait(remaining)

    def _wake_waiters(self) -> None:
        with self._changed:
            self._changed.notify_all()


class CaptureThread(threading.Thread):
    """Grabs frames at config.FPS into the ring buffer and wakes the vision loop.

    The thread owns the source once started and releases it when it ends.
    """

    def __init__(self, source: FrameSource, ring: RingBuffer, fps: float = config.FPS) -> None:
        super().__init__(name="capture", daemon=True)
        self._source = source
        self._ring = ring
        self._frame_period = 1.0 / fps
        self._stop_requested = threading.Event()
        self._ended = threading.Event()

    def run(self) -> None:
        next_due = time.monotonic()
        try:
            while not self._stop_requested.is_set():
                image = self._source.read()
                if image is None:
                    break
                now = time.monotonic()
                self._ring.push(image, now)
                # Absolute deadlines so sleep overshoot doesn't add up; never burst to catch up.
                next_due = max(next_due + self._frame_period, now)
                self._stop_requested.wait(max(0.0, next_due - now))
        except Exception:
            log.exception("capture failed; no more frames")
        finally:
            self._ended.set()
            self._ring._wake_waiters()
            self._source.release()

    def wait_for_frame(self, after_index: int, timeout: float) -> BufferedFrame | None:
        """Block until a frame newer than after_index is buffered; None on timeout or end of source.

        Returns the oldest such frame, so a vision loop that passes the last index it handled
        sees every frame in order (faces every frame).
        """
        return self._ring._wait_for_frame_after(after_index, timeout, self._ended.is_set)

    def stop(self) -> None:
        self._stop_requested.set()
