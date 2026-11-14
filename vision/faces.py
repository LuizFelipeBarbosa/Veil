"""YuNet face detection (WP2). Runs on every frame in the output path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from contracts import Detection


class FaceDetector:
    def __init__(
        self,
        model_path: Path = config.YUNET_MODEL,
        input_size: tuple[int, int] = config.YUNET_INPUT_SIZE,
        score: float = config.FACE_SCORE,
    ) -> None:
        raise NotImplementedError("WP2")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Faces in a BGR frame, boxes in frame pixels, label "face", face_row set to the YuNet row."""
        raise NotImplementedError("WP2")
