"""A capture source that plays a video file, for offline tests and `app.py --source`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config


class FileSource:
    """Plays a clip frame by frame at the config frame size. Pacing is the caller's job.

    A clip with another aspect ratio (a portrait phone video) is scaled to cover
    the frame and centre-cropped, so faces are never stretched.
    """

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
        return cover(image, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    def release(self) -> None:
        self._video.release()


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover width x height, then crop the centre."""
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    scaled_width, scaled_height = round(source_width * scale), round(source_height * scale)
    scaled = cv2.resize(image, (scaled_width, scaled_height))
    left, top = (scaled_width - width) // 2, (scaled_height - height) // 2
    return scaled[top : top + height, left : left + width]
