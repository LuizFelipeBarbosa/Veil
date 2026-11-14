"""WP2: YuNet face detection. No camera; clips from tests/data only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from contracts import FACE, Detection
from fakes.capture import FileSource
from vision.faces import (
    _ROTATIONS,
    FaceDetector,
    _apply,
    _map_rows,
    _merge_same_faces,
    _on_pixel_centres,
    _rotate,
    _to_detection,
    _to_frame_pixels,
)

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
            # face_row is in frame pixels too: its box, clipped, is the detection's box.
            x, y, w, h = detection.face_row[:4]
            assert abs(max(x, 0) - x1) <= 1 and abs(min(x + w, config.FRAME_WIDTH) - x2) <= 1
            assert abs(max(y, 0) - y1) <= 1 and abs(min(y + h, config.FRAME_HEIGHT) - y2) <= 1
    source.release()
    assert frames_with_faces > 0


# --- WP12: faces at an angle -------------------------------------------------

TEST_ANGLES = (0, 15, 30, 45, 60, 90, 135, 180, -15, -30, -45, -60, -90, -135)
EVERY_NTH_FRAME = 12  # about 35 frames of crossing.mov, enough faces, a few seconds per angle sweep


def crossing_clip() -> Path | None:
    for candidate in (DATA / "crossing.mov", MAIN_CHECKOUT_DATA / "crossing.mov"):
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def upright_faces() -> list[tuple[np.ndarray, list[Detection]]]:
    """In-memory frames of crossing.mov with their confident upright faces (the reference).

    A reference face scores >= 0.8 at full size, is at least 20 px and does not touch the frame
    edge, so its box is a trustworthy ground truth for where the face is.
    """
    clip = crossing_clip()
    if clip is None:
        pytest.skip("tests/data/crossing.mov not recorded yet")
    reference = FaceDetector(input_size=(config.FRAME_WIDTH, config.FRAME_HEIGHT), rotations=())
    source = FileSource(clip, loop=False)
    frames = []
    index = 0
    while (image := source.read()) is not None:
        if index % EVERY_NTH_FRAME == 0:
            faces = [d for d in reference.detect(image) if _is_reference_face(d)]
            if faces:
                frames.append((image, faces))
        index += 1
    source.release()
    assert sum(len(faces) for _, faces in frames) >= 30
    return frames


def _is_reference_face(detection: Detection) -> bool:
    x1, y1, x2, y2 = detection.xyxy
    on_edge = x1 <= 0 or y1 <= 0 or x2 >= config.FRAME_WIDTH or y2 >= config.FRAME_HEIGHT
    return detection.conf >= 0.8 and min(x2 - x1, y2 - y1) >= 20 and not on_edge


def _turn_frame(image: np.ndarray, angle: int) -> tuple[np.ndarray, np.ndarray]:
    """The frame turned about its centre, same canvas (as a tilted camera sees it), and its 2x3 map."""
    height, width = image.shape[:2]
    to_turned = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, to_turned, (width, height)), to_turned


def _turned_box(xyxy: tuple[int, int, int, int], to_turned: np.ndarray) -> np.ndarray | None:
    """Corners of a box after the turn, or None when any corner left the frame."""
    x1, y1, x2, y2 = xyxy
    corners = _apply(to_turned, np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float64))
    inside = (corners >= 0).all() and (corners[:, 0] <= config.FRAME_WIDTH).all()
    return corners if inside and (corners[:, 1] <= config.FRAME_HEIGHT).all() else None


def _matching(detections: list[Detection], corners: np.ndarray) -> list[Detection]:
    """Detections whose box centre lies near the turned face's centre and whose size is about right."""
    centre = corners.mean(axis=0)
    size = float(np.linalg.norm(corners[1] - corners[0]))  # the face box's original width
    found = []
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        offset = np.hypot((x1 + x2) / 2 - centre[0], (y1 + y2) / 2 - centre[1])
        if detection.conf >= config.FACE_SCORE and offset <= 0.35 * max(size, 1) and 0.5 * size <= x2 - x1 <= 3 * size:
            found.append(detection)
    return found


def _recall(detector: FaceDetector, frames, angle: int) -> float:
    found = total = 0
    for image, faces in frames:
        turned, to_turned = _turn_frame(image, angle)
        detections = detector.detect(turned)
        for face in faces:
            corners = _turned_box(face.xyxy, to_turned)
            if corners is not None:
                total += 1
                found += bool(_matching(detections, corners))
    assert total >= 20, f"too few reference faces stay in frame at {angle} degrees"
    return found / total


def _iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


@pytest.mark.parametrize("input_size", [(640, 480), (320, 240)], ids=["mac-640", "pi-320"])
def test_recall_at_every_angle_is_close_to_upright_recall(upright_faces, input_size) -> None:
    detector = FaceDetector(input_size=input_size, rotations_per_call=None)
    upright = _recall(detector, upright_faces, 0)
    assert upright >= 0.9
    for angle in TEST_ANGLES:
        assert _recall(detector, upright_faces, angle) >= 0.9 * upright, f"{angle} degrees"


def _single_pass(detector: FaceDetector, image: np.ndarray, angle: int) -> list[Detection]:
    """Only the pass turned `angle` degrees, its rows mapped back into `image`."""
    pass_image, to_pass = _rotate(image, angle)
    rows = _map_rows(detector._detect_rows(pass_image), cv2.invertAffineTransform(to_pass))
    return [_to_detection(row, image.shape[1], image.shape[0]) for row in rows]


@pytest.mark.parametrize("pass_angle", [*_ROTATIONS, 90, 270])
def test_rotated_hit_maps_back_onto_the_face(upright_faces, pass_angle) -> None:
    """A frame turned by -pass_angle, seen only through the pass that turns it upright again: the
    mapped-back box contains the turned face centre and has IoU >= 0.5 with the reference box
    turned the same way (as its axis-aligned box).

    Both boxes are the axis-aligned box of the same turn applied to a box of the same face, so
    the turn inflates them alike and what is left is YuNet's jitter after resampling (IoU 0.7-0.9
    in practice). A wrong map (sign, canvas, centre) puts the box elsewhere: IoU near 0.
    """
    detector = FaceDetector(input_size=(640, 480))
    checked = 0
    for image, faces in upright_faces[::2]:
        turned, to_turned = _turn_frame(image, -pass_angle)
        detections = _single_pass(detector, turned, pass_angle % 360)
        for face in faces:
            corners = _turned_box(face.xyxy, to_turned)
            if corners is None or not (match := _matching(detections, corners)):
                continue
            (cx, cy), box = corners.mean(axis=0), match[0].xyxy
            assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
            assert _iou(box, (*corners.min(axis=0), *corners.max(axis=0))) >= 0.5
            checked += 1
    assert checked >= 15


def test_an_upright_face_is_one_detection_not_one_per_pass(upright_faces) -> None:
    detector = FaceDetector(input_size=(640, 480), rotations_per_call=None)
    for image, faces in upright_faces:
        detections = detector.detect(image)
        for face in faces:
            x1, y1, x2, y2 = face.xyxy
            centres_inside = [
                d for d in detections if x1 <= (d.xyxy[0] + d.xyxy[2]) / 2 <= x2 and y1 <= (d.xyxy[1] + d.xyxy[3]) / 2 <= y2
            ]
            assert len(centres_inside) == 1


def test_landmarks_from_a_rotated_pass_still_align_for_sface(upright_faces) -> None:
    """The same face, upright and in a turned frame found through a rotated pass, embeds alike."""
    if not config.SFACE_MODEL.exists():
        pytest.skip("SFace weights not installed")
    recognizer = cv2.FaceRecognizerSF.create(str(config.SFACE_MODEL), "")
    detector = FaceDetector(input_size=(640, 480))

    def unit_embedding(image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        feature = recognizer.feature(recognizer.alignCrop(image, face_row)).reshape(-1)
        return feature / np.linalg.norm(feature)

    image, faces = max(upright_faces, key=lambda frame: max(_area(f) for f in frame[1]))
    face = max(faces, key=_area)
    reference = unit_embedding(image, face.face_row)
    for angle in (60, 90, 135, 180, -60, -90, -135):
        turned, to_turned = _turn_frame(image, angle)
        corners = _turned_box(face.xyxy, to_turned)
        assert corners is not None
        # Only the rotated pass that turns the face upright again, its rows mapped back to `turned`.
        found = _matching(_single_pass(detector, turned, -angle % 360), corners)
        assert found, f"{angle} degrees: the rotated pass missed the face"
        cosine = float(unit_embedding(turned, found[0].face_row) @ reference)
        assert cosine >= 0.6, f"{angle} degrees: cosine {cosine:.2f}"  # match threshold is 0.363


def _area(detection: Detection) -> int:
    x1, y1, x2, y2 = detection.xyxy
    return (x2 - x1) * (y2 - y1)


@pytest.mark.parametrize("angle", [90, 180, 270, 60, 300])
def test_rotate_map_matches_the_pixels_and_maps_back_exactly(angle) -> None:
    image = np.random.default_rng(angle).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    rotated, to_rotated = _rotate(image, angle)
    canvas = (rotated.shape[1], rotated.shape[0])
    if angle % 90 == 0:  # cv2.rotate's pixels are what the returned matrix says
        np.testing.assert_array_equal(rotated, cv2.warpAffine(image, _on_pixel_centres(to_rotated), canvas, flags=cv2.INTER_NEAREST))
        assert canvas == ((48, 64) if angle % 180 else (64, 48))
    row = np.zeros(15, dtype=np.float32)
    row[:4] = (20, 10, 8, 12)
    row[4:14] = (21, 13, 26, 13, 24, 16, 22, 19, 26, 19)
    there = _map_rows(row[None, :], to_rotated)
    back = _map_rows(there, cv2.invertAffineTransform(to_rotated))[0]
    np.testing.assert_allclose(back[4:14], row[4:14], atol=1e-3)  # landmarks are exact
    x, y, w, h = back[:4]  # the box only grows: it is the axis-aligned box around the turned box
    assert x <= 20 + 1e-3 and y <= 10 + 1e-3 and x + w >= 28 - 1e-3 and y + h >= 22 - 1e-3
    assert back[14] == row[14]


def test_hits_from_different_passes_merge_to_the_highest_score_and_neighbours_stay() -> None:
    def row(x: float, y: float, size: float, score: float, nose: tuple[float, float]) -> np.ndarray:
        values = np.zeros(15, dtype=np.float32)
        values[:4], values[8:10], values[14] = (x, y, size, size), nose, score
        return values

    face = row(10, 10, 40, 0.6, nose=(30, 32))
    inner = row(20, 20, 20, 0.5, nose=(29, 31))  # same pass: YuNet's NMS already decided
    assert [float(r[14]) for r in _merge_same_faces([np.stack([face, inner])])] == pytest.approx([0.6, 0.5])

    tilted_duplicate = row(0, 0, 60, 0.9, nose=(31, 31))  # inflated box, same nose
    neighbour = row(60, 10, 40, 0.7, nose=(80, 32))
    merged = _merge_same_faces([np.stack([face, inner]), np.stack([tilted_duplicate, neighbour])])
    assert [float(r[14]) for r in merged] == pytest.approx([0.9, 0.7])

    spanning = row(25, 0, 70, 0.95, nose=(58, 30))  # a stray wide hit between two faces, over 5/8 of this one
    merged = _merge_same_faces([np.stack([face]), np.stack([spanning])])
    assert [float(r[14]) for r in merged] == pytest.approx([0.95, 0.6])  # the real face is not swallowed


def test_rotated_passes_round_robin_and_cover_every_angle() -> None:
    detector = FaceDetector(rotations=(60, 180, 300, 120, 240), rotations_per_call=2)
    calls = [detector._rotations_this_call() for _ in range(5)]
    assert calls == [[60, 180], [300, 120], [240, 60], [180, 300], [120, 240]]
    assert {angle for call in calls[:3] for angle in call} == {60, 180, 300, 120, 240}
    assert FaceDetector(rotations_per_call=None)._rotations_this_call() == list(_ROTATIONS)
