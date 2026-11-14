"""WP2: YuNet face detection. No camera; clips from tests/data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
from contracts import FACE
from fakes.capture import FileSource
from vision.faces import FaceDetector, _to_detection, _to_frame_pixels

DATA = Path(__file__).parent / "data"
SYNTHETIC_CLIP = DATA / "synthetic.mp4"
MAIN_CHECKOUT_DATA = Path("/Users/luiz/Developer/privacy-cam/tests/data")


def faces_clip() -> Path | None:
    for candidate in (DATA / "faces.mov", MAIN_CHECKOUT_DATA / "faces.mov"):
        if candidate.exists():
            return candidate
    return None


def test_no_faces_in_the_synthetic_clip() -> None:
    detector = FaceDetector()
    source = FileSource(SYNTHETIC_CLIP, loop=False)
    while (image := source.read()) is not None:
        assert detector.detect(image) == []
    source.release()


def test_rows_scale_from_model_input_to_frame_pixels() -> None:
    row = np.arange(15, dtype=np.float32)  # x, y, w, h, 5 landmarks, score
    scaled = _to_frame_pixels(row[None, :], scale_x=2.0, scale_y=3.0)[0]
    assert scaled[:4].tolist() == [0.0, 3.0, 4.0, 9.0]
    assert scaled[4:14].tolist() == [8.0, 15.0, 12.0, 21.0, 16.0, 27.0, 20.0, 33.0, 24.0, 39.0]
    assert scaled[14] == row[14]  # the score is not a coordinate
    assert row[0] == 0.0 and row[1] == 1.0  # the original row is untouched


def test_detection_box_rounds_outward_and_clips_to_the_frame() -> None:
    row = np.zeros(15, dtype=np.float32)
    row[:4] = (-3.5, 10.2, 50.0, 20.6)
    row[14] = 0.87
    detection = _to_detection(row, width=40, height=480)
    assert detection.label == FACE
    assert detection.xyxy == (0, 10, 40, 31)
    assert detection.conf == pytest.approx(0.87)
    assert detection.face_row is not None and detection.face_row is not row
    np.testing.assert_array_equal(detection.face_row, row)


def test_faces_found_in_the_faces_clip_at_both_input_sizes() -> None:
    clip = faces_clip()
    if clip is None:
        pytest.skip("tests/data/faces.mov not recorded yet")
    full = FaceDetector(input_size=(config.FRAME_WIDTH, config.FRAME_HEIGHT))
    half = FaceDetector(input_size=(config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2))
    source = FileSource(clip, loop=False)
    frames_with_faces = 0
    while (image := source.read()) is not None:
        found = full.detect(image)
        frames_with_faces += bool(found)
        for detection in found + half.detect(image):
            x1, y1, x2, y2 = detection.xyxy
            assert 0 <= x1 < x2 <= config.FRAME_WIDTH and 0 <= y1 < y2 <= config.FRAME_HEIGHT
            assert detection.label == FACE and detection.conf >= config.FACE_SCORE
            assert detection.face_row is not None and detection.face_row.shape == (15,)
            # Landmarks (the eyes) sit inside the box, in frame pixels.
            assert x1 <= detection.face_row[4] <= x2 and y1 <= detection.face_row[5] <= y2
    source.release()
    assert frames_with_faces > 0
