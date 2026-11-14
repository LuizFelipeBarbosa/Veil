"""YuNet face detection (WP2). Runs on every frame in the output path."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

import config
from contracts import FACE, Detection

# YuNet row layout: x, y, w, h, then 5 landmarks as (x, y) pairs, then the score.
_X_COLUMNS = [0, 2, 4, 6, 8, 10, 12]
_Y_COLUMNS = [1, 3, 5, 7, 9, 11, 13]
_SCORE_COLUMN = 14


class FaceDetector:
    def __init__(
        self,
        model_path: Path = config.YUNET_MODEL,
        input_size: tuple[int, int] = config.YUNET_INPUT_SIZE,
        score: float = config.FACE_SCORE,
    ) -> None:
        self._input_size = tuple(input_size)  # (width, height)
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", self._input_size, score)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Faces in a BGR frame, boxes in frame pixels, label "face", face_row set to the YuNet row."""
        height, width = image.shape[:2]
        if (width, height) == self._input_size:
            model_input = image
        else:
            model_input = cv2.resize(image, self._input_size, interpolation=cv2.INTER_AREA)

        _, rows = self._detector.detect(model_input)
        if rows is None:
            return []

        rows = _to_frame_pixels(rows, scale_x=width / self._input_size[0], scale_y=height / self._input_size[1])
        return [_to_detection(row, width, height) for row in rows]


def _to_frame_pixels(rows: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """A copy of the YuNet rows with every coordinate scaled from model input to frame pixels."""
    scaled = rows.astype(np.float32, copy=True)
    scaled[:, _X_COLUMNS] *= scale_x
    scaled[:, _Y_COLUMNS] *= scale_y
    return scaled


def _to_detection(row: np.ndarray, width: int, height: int) -> Detection:
    """Round the box outward and clip it to the frame, so rounding never shrinks a face."""
    x, y, w, h = (float(v) for v in row[:4])
    x1 = min(max(math.floor(x), 0), width)
    y1 = min(max(math.floor(y), 0), height)
    x2 = min(max(math.ceil(x + w), 0), width)
    y2 = min(max(math.ceil(y + h), 0), height)
    return Detection(label=FACE, xyxy=(x1, y1, x2, y2), conf=float(row[_SCORE_COLUMN]), face_row=row.copy())
