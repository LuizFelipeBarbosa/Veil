"""The vision loop: raw frame in, CensoredFrame out. Orchestrator-owned (WP0, WP8).

This is the one place raw frames turn into output, and it fails closed: until
the detectors exist, and whenever censoring raises, the whole frame is pixelated.
WP8 wires capture.RingBuffer, vision/* and the recorder in here.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

import config
from capture import FrameSource
from contracts import CensoredFrame, Policy

log = logging.getLogger(__name__)


def fail_closed_frame(image: np.ndarray, index: int) -> CensoredFrame:
    """Pixelate the whole frame. The last resort when nothing else can be trusted."""
    height, width = image.shape[:2]
    small_size = (max(1, width // config.PIXEL_BLOCK), max(1, height // config.PIXEL_BLOCK))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return CensoredFrame(image=pixelated, index=index, fail_closed=True)


class Pipeline(threading.Thread):
    """Reads the source at config.FPS and publishes the latest censored frame."""

    def __init__(self, source: FrameSource, policy: Policy) -> None:
        super().__init__(name="vision", daemon=True)
        self._source = source
        self._policy = policy
        self._stop_requested = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest: CensoredFrame | None = None

    def latest(self) -> CensoredFrame | None:
        with self._latest_lock:
            return self._latest

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        frame_period = 1.0 / config.FPS
        index = 0
        while not self._stop_requested.is_set():
            started = time.monotonic()
            image = self._source.read()
            if image is None:
                break
            censored = self._censor_or_fail_closed(image, index)
            with self._latest_lock:
                self._latest = censored
            index += 1
            self._stop_requested.wait(max(0.0, frame_period - (time.monotonic() - started)))

    def _censor_or_fail_closed(self, image: np.ndarray, index: int) -> CensoredFrame:
        try:
            return self._censor(image, index)
        except NotImplementedError:
            return fail_closed_frame(image, index)
        except Exception:
            log.exception("censoring failed on frame %d; failing closed", index)
            return fail_closed_frame(image, index)

    def _censor(self, image: np.ndarray, index: int) -> CensoredFrame:
        raise NotImplementedError("WP8 wires faces, tracking, policy and render here")
