"""YOLOE-26n on its own thread (WP5): PyTorch on the Mac, NCNN on the Pi, chosen by config."""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from contracts import BufferedFrame, Detection


class ObjectDetector:
    """Loads the model for config.YOLOE_RUNTIME with the 4 fixed classes."""

    def __init__(self) -> None:
        raise NotImplementedError("WP5")

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError("WP5")


class ObjectsThread(threading.Thread):
    """Runs the detector on the newest buffered frame whenever it is free."""

    def __init__(self, detector: ObjectDetector, newest_frame: Callable[[], BufferedFrame | None]) -> None:
        raise NotImplementedError("WP5")

    def latest(self) -> tuple[int, float, list[Detection]] | None:
        """(frame_index, finished_at, detections) of the last finished detection, or None."""
        raise NotImplementedError("WP5")

    def stop(self) -> None:
        raise NotImplementedError("WP5")
