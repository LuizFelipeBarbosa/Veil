"""A capture source that plays a video file, for offline tests and `app.py --source`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config


class FileSource:
    """Plays a clip frame by frame, resized to the config frame size. Pacing is the caller's job."""

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self._path = str(path)
        self._loop = loop
        self._video = cv2.VideoCapture(self._path)
        if not self._video.isOpened():
            raise FileNotFoundError(f"cannot open clip {self._path}")

    def read(self) -> np.ndarray | None:
        ok, image = self._video.read()
        if not ok and self._loop:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._video.read()
        if not ok:
            return None
        return cv2.resize(image, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

    def release(self) -> None:
        self._video.release()
