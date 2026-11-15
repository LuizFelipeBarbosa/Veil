"""YuNet face detection (WP2, WP12). Runs on every frame in the output path.

YuNet is trained on roughly upright faces and loses them past about 30 degrees of
in-plane rotation (a tilted head, a camera held at an angle). So besides the upright
frame it also runs on rotated copies, maps every hit back into the original frame and
merges the hits that are the same face.
"""

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

# Extra passes, degrees counter-clockwise, besides upright: YuNet holds about +-30, so 60-degree steps.
# Ordered so consecutive calls cover far-apart angles.
_ROTATIONS = (60, 180, 300, 120, 240)
_NOSE_COLUMNS = [8, 9]  # the third landmark; two passes' hits are one face when their noses agree
# Or when the stronger hit's box already covers this share of the weaker one's box.
_COVERED_SHARE = 0.7
# A rotated-pass hit that is no upright hit's face needs this score (WP18). On intersection.mov such
# hits on facades scored at most 0.65 (640x480) / 0.70 (320x240); on rotated crossing.mov faces,
# 0.75 dropped recall at 320x240 to 0.90 at 45 and -135 degrees, 0.7 kept it at 0.91 or better.
_ROTATED_ONLY_SCORE = 0.7
# Passes that look for faces turned more than 90 degrees from upright, near upside down (WP26).
# When the upright pass finds a face this sure, the camera is held upright and every face in the
# scene shares its "up", so these passes are skipped. On ad.mp4 they put 0.70-0.81 hits on the
# printed adverts (the bus-shelter panel at 120 degrees in about one frame in two) while the
# speaker's upright face scored 0.94-0.95. The upright pass also finds faces turned 180 degrees,
# at up to 0.92 on crossing.mov (0.86 at 135), so a lower bar would wrongly skip these passes
# when the camera itself is upside down.
_UPSIDE_DOWN_PASSES = frozenset({120, 180, 240})
_UPRIGHT_SCENE_SCORE = 0.93
# cv2.rotate codes for the exact quarter turns; any other angle goes through warpAffine.
_QUARTER_TURNS = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


class FaceDetector:
    def __init__(
        self,
        model_path: Path = config.YUNET_MODEL,
        input_size: tuple[int, int] = config.YUNET_INPUT_SIZE,
        score: float = config.FACE_SCORE,
        rotations: tuple[int, ...] = _ROTATIONS,
        rotations_per_call: int | None = config.FACE_ROTATIONS_PER_CALL,
    ) -> None:
        self._input_size = tuple(input_size)  # (width, height)
        self._rotations = tuple(rotations)
        self._rotations_per_call = len(self._rotations) if rotations_per_call is None else rotations_per_call
        self._next_rotation = 0  # round-robin position in self._rotations
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", self._input_size, score)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Faces in a BGR frame, boxes in frame pixels, label "face", face_row set to the YuNet row.

        Runs upright on every call, plus the next few rotated passes in round-robin order (all of
        them when rotations_per_call is None). A tilted face found only by a rotated pass is
        therefore seen every few frames; the tracker keeps its ID and the render hold-over keeps
        it blurred in between. A face seen by several passes is one detection (the highest
        score), its face_row landmarks in frame pixels so SFace can align it.
        """
        height, width = image.shape[:2]
        upright = self._detect_rows(image)
        camera_upright = bool(len(upright)) and float(upright[:, _SCORE_COLUMN].max()) >= _UPRIGHT_SCENE_SCORE
        rows = [upright]
        for angle in self._rotations_this_call():
            if camera_upright and angle in _UPSIDE_DOWN_PASSES:
                continue
            rotated, to_rotated = _rotate(image, angle)
            mapped = _map_rows(self._detect_rows(rotated), cv2.invertAffineTransform(to_rotated))
            rows.append(_without_phantoms(mapped, upright))
        merged = _merge_same_faces(rows)
        return [_to_detection(row, width, height) for row in merged]

    def _rotations_this_call(self) -> list[int]:
        count = min(self._rotations_per_call, len(self._rotations))
        start = self._next_rotation
        self._next_rotation = (start + count) % max(len(self._rotations), 1)
        return [self._rotations[(start + i) % len(self._rotations)] for i in range(count)]

    def _detect_rows(self, image: np.ndarray) -> np.ndarray:
        """YuNet rows in this image's pixels; the model input keeps the image's orientation."""
        height, width = image.shape[:2]
        input_width, input_height = self._input_size
        if (width > height) != (input_width > input_height):
            input_width, input_height = input_height, input_width
        if (width, height) == (input_width, input_height):
            model_input = image
        else:
            model_input = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)

        self._detector.setInputSize((input_width, input_height))
        _, rows = self._detector.detect(model_input)
        if rows is None:
            return np.empty((0, 15), dtype=np.float32)
        return _to_frame_pixels(rows, scale_x=width / input_width, scale_y=height / input_height)


def _rotate(image: np.ndarray, angle: int) -> tuple[np.ndarray, np.ndarray]:
    """The image turned `angle` degrees counter-clockwise about its centre, and the 2x3 map into it.

    The map works on YuNet's coordinates, where pixel i spans [i, i + 1). Quarter turns keep every
    pixel (the canvas swaps width and height). Other angles keep the canvas size and lose the
    corners; a face there is left to the neighbouring passes.
    """
    height, width = image.shape[:2]
    if angle % 180 == 90:
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height
    to_rotated = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    to_rotated[:, 2] += (out_width / 2 - width / 2, out_height / 2 - height / 2)  # recentre on the canvas
    if angle in _QUARTER_TURNS:
        return cv2.rotate(image, _QUARTER_TURNS[angle]), to_rotated
    return cv2.warpAffine(image, _on_pixel_centres(to_rotated), (out_width, out_height)), to_rotated


def _on_pixel_centres(affine: np.ndarray) -> np.ndarray:
    """The same map for warpAffine, which puts pixel i's centre at i rather than i + 0.5."""
    shifted = affine.copy()
    shifted[:, 2] += affine[:, :2] @ (0.5, 0.5) - 0.5
    return shifted


def _map_rows(rows: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """YuNet rows moved through a 2x3 affine map: landmarks exactly, the box as the axis-aligned
    box around its four mapped corners, so the box still contains the whole face."""
    mapped = rows.astype(np.float32, copy=True)
    for row in mapped:
        x, y, w, h = row[:4]
        corners = _apply(affine, np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float64))
        (x1, y1), (x2, y2) = corners.min(axis=0), corners.max(axis=0)
        row[:4] = (x1, y1, x2 - x1, y2 - y1)
        row[4:14] = _apply(affine, row[4:14].reshape(5, 2).astype(np.float64)).reshape(-1)
    return mapped


def _apply(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ affine[:, :2].T + affine[:, 2]


def _without_phantoms(rotated_rows: np.ndarray, upright_rows: np.ndarray) -> np.ndarray:
    """The rotated pass's rows minus those that are likely scenery, not faces.

    A row that is the same face as an upright row (the merge's own test, the rotated row as the
    weaker one) is kept whatever its score. Any other row must score _ROTATED_ONLY_SCORE: facades
    and windows turned 60 or 120 degrees look face-like to YuNet. That includes a big facade hit
    around a small upright face: it merely overlaps the face, and the face keeps its own box.
    """
    keep = [
        float(row[_SCORE_COLUMN]) >= _ROTATED_ONLY_SCORE or any(_same_face(row, upright) for upright in upright_rows)
        for row in rotated_rows
    ]
    return rotated_rows[np.array(keep, dtype=bool)] if len(rotated_rows) else rotated_rows


def _merge_same_faces(passes: list[np.ndarray]) -> list[np.ndarray]:
    """One row per face across passes, the highest score kept.

    A weaker row from another pass is a duplicate of a kept row when their noses agree (each nose
    inside the other's box; landmarks map back exactly while boxes inflate), or when the kept box
    already covers _COVERED_SHARE of it, so dropping it uncovers little (fail closed). A stray wide
    hit spanning two faces therefore swallows neither. Rows of the same pass never suppress each
    other: YuNet's own NMS already ran, so the upright pass alone gives exactly what it gave before.
    """
    candidates = [(index, row) for index, rows in enumerate(passes) for row in rows]
    candidates.sort(key=lambda candidate: -float(candidate[1][_SCORE_COLUMN]))
    kept: list[tuple[int, np.ndarray]] = []
    for index, row in candidates:
        if not any(other_index != index and _same_face(row, other) for other_index, other in kept):
            kept.append((index, row))
    return [row for _, row in kept]


def _same_face(weaker: np.ndarray, kept: np.ndarray) -> bool:
    noses_agree = _nose_inside(weaker, kept) and _nose_inside(kept, weaker)
    return noses_agree or _share_covered(weaker, kept) >= _COVERED_SHARE


def _share_covered(row: np.ndarray, by: np.ndarray) -> float:
    """The share of row's (x, y, w, h) box that lies inside by's box."""
    width = min(row[0] + row[2], by[0] + by[2]) - max(row[0], by[0])
    height = min(row[1] + row[3], by[1] + by[3]) - max(row[1], by[1])
    if width <= 0 or height <= 0:
        return 0.0
    return float(width * height) / max(float(row[2] * row[3]), 1e-6)


def _nose_inside(row: np.ndarray, box_row: np.ndarray) -> bool:
    nose_x, nose_y = row[_NOSE_COLUMNS]
    x, y, w, h = box_row[:4]
    return bool(x <= nose_x <= x + w and y <= nose_y <= y + h)


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
