"""The vision loop: raw frame in, CensoredFrame out. Orchestrator-owned (WP0, WP8).

This is the one place raw frames turn into output, and it fails closed: while a
detector is missing, and whenever censoring raises, the whole frame is pixelated.

Per captured frame: detect on the newest frame (so late detections can be
backfilled), then censor the oldest buffered frame, one second behind, with its
own boxes, and hand it to the preview and the recorder.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

import config
from capture import CaptureThread, FrameSource, RingBuffer
from contracts import BufferedFrame, CensoredFrame, Policy
from recorder import Recorder

log = logging.getLogger(__name__)

_FRAME_WAIT_S = 0.1


def fail_closed_frame(image: np.ndarray, index: int) -> CensoredFrame:
    """Pixelate the whole frame. The last resort when nothing else can be trusted."""
    height, width = image.shape[:2]
    small_size = (max(1, width // config.PIXEL_BLOCK), max(1, height // config.PIXEL_BLOCK))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return CensoredFrame(image=pixelated, index=index, fail_closed=True)


class Pipeline(threading.Thread):
    """Owns the capture thread and the delay buffer; publishes the latest censored frame."""

    def __init__(self, source: FrameSource, policy: Policy, recorder: Recorder | None = None) -> None:
        super().__init__(name="vision", daemon=True)
        self._policy = policy
        self._recorder = recorder
        self._ring = RingBuffer()
        self._capture = CaptureThread(source, self._ring)
        self._stop_requested = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest: CensoredFrame | None = None

    def latest(self) -> CensoredFrame | None:
        with self._latest_lock:
            return self._latest

    def stop(self) -> None:
        self._stop_requested.set()
        self._capture.stop()

    def run(self) -> None:
        self._capture.start()
        last_index = -1
        while not self._stop_requested.is_set():
            newest = self._capture.wait_for_frame(last_index, _FRAME_WAIT_S)
            if newest is None:
                if not self._capture.is_alive():
                    break
                continue
            last_index = newest.index
            self._detect_or_leave_unchecked(newest)

            oldest = self._ring.pop_oldest()
            if oldest is not None:
                self._publish(self._censor_or_fail_closed(oldest))
        self._capture.stop()

    def _publish(self, censored: CensoredFrame) -> None:
        with self._latest_lock:
            self._latest = censored
        if self._recorder is not None:
            self._recorder.write(censored)

    def _detect_or_leave_unchecked(self, frame: BufferedFrame) -> None:
        """A frame whose detection fails keeps faces_done False, so it leaves the buffer pixelated."""
        try:
            self._detect(frame)
        except NotImplementedError:
            pass
        except Exception:
            log.exception("detection failed on frame %d; it will be pixelated whole", frame.index)

    def _censor_or_fail_closed(self, frame: BufferedFrame) -> CensoredFrame:
        if not frame.faces_done:
            return fail_closed_frame(frame.image, frame.index)
        try:
            return self._censor(frame)
        except Exception:
            log.exception("censoring failed on frame %d; failing closed", frame.index)
            return fail_closed_frame(frame.image, frame.index)

    def _detect(self, frame: BufferedFrame) -> None:
        raise NotImplementedError("wired when WP2 merges: faces, tracking, hold-over, backfill, watchdog")

    def _censor(self, frame: BufferedFrame) -> CensoredFrame:
        raise NotImplementedError("wired when WP2 merges: render.censor with policy.decide")
