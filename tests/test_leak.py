"""WP2 leak test: a clip through detect -> track -> hold-over -> backfill -> censor -> MJPG file.

A checker YuNet, at a lower score than the pipeline uses, finds the faces in every
RAW frame; the test then looks at the pixels of the censored frame at each of those
faces. A face whose middle is unchanged was left visible: that is a leak.

Changed with the lead's approval on 2026-11-14 (15:25). The old check ran the detector
on the censored frames, and with rotated passes it "found" faces inside regions that
were already pixelated, so it could not tell a leak from a pixelated blob.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from contracts import BufferedFrame, CensoredFrame, Policy, PolicySnapshot, TrackedBox
from fakes.capture import FileSource
from vision.faces import FaceDetector
from vision.render import HoldOver, Watchdog, backfill, censor
from vision.tracking import make_face_tracker

LEAK_CHECK_SCORE = 0.3
CONFIDENT_FACE_SCORE = 0.6  # the Brief's own bar for "this is a face"
MAX_WEAK_LEAK_FRAME_SHARE = 0.10
# A face counts as pixelated when at least this share of the pixels in the middle half of its box changed.
MIN_CHANGED_SHARE = 0.5
CLIP_DIRS = (
    Path(__file__).parent / "data",
    Path("/Users/luiz/Developer/privacy-cam/tests/data"),  # the lead records into the main checkout
)


def _clip(name: str) -> Path:
    for directory in CLIP_DIRS:
        if (directory / name).exists():
            return directory / name
    pytest.skip(f"tests/data/{name} not recorded yet")


def _always_blur(_box: TrackedBox, _policy: PolicySnapshot) -> str:
    return "blur"


def _is_pixelated(raw: np.ndarray, censored: np.ndarray, xyxy: tuple[int, int, int, int]) -> bool:
    """Did censoring change the middle half of this box?"""
    x1, y1, x2, y2 = xyxy
    quarter_w, quarter_h = (x2 - x1) // 4, (y2 - y1) // 4
    middle = np.s_[y1 + quarter_h : max(y2 - quarter_h, y1 + quarter_h + 1), x1 + quarter_w : max(x2 - quarter_w, x1 + quarter_w + 1)]
    changed = (raw[middle] != censored[middle]).any(axis=-1)
    return changed.size == 0 or changed.mean() >= MIN_CHANGED_SHARE


def _censor_clip(clip: Path) -> tuple[list[CensoredFrame], int, list[tuple[int, float]]]:
    """Every frame of the clip, censored with a delay buffer; how many raw frames had faces;
    and the leaks: (frame index, checker score) of every raw face left unpixelated."""
    detector = FaceDetector()
    checker = FaceDetector(score=LEAK_CHECK_SCORE, rotations=())  # upright pass: what a viewer would recognise
    raw_faces: dict[int, list] = {}
    leaks: list[tuple[int, float]] = []
    tracker = make_face_tracker()
    hold_over = HoldOver()
    watchdog = Watchdog(watch_objects=False)
    policy = Policy().snapshot()
    buffer: deque[BufferedFrame] = deque()
    seen_ids: set[str] = set()
    censored: list[CensoredFrame] = []
    raw_frames_with_faces = 0

    def release_oldest(now: float) -> None:
        frame = buffer.popleft()
        out = censor(frame, policy, _always_blur, watchdog.tripped(now))
        censored.append(out)
        leaks.extend(
            (frame.index, round(face.conf, 2))
            for face in raw_faces.pop(frame.index)
            if not _is_pixelated(frame.image, out.image, face.xyxy)
        )

    source = FileSource(clip, loop=False)
    index = 0
    while (image := source.read()) is not None:
        now = index / config.FPS
        frame = BufferedFrame(index=index, timestamp=now, image=image)

        raw_faces[index] = checker.detect(image)
        detections = detector.detect(image)
        raw_frames_with_faces += bool(detections)
        boxes = hold_over.apply(tracker.update(detections, index, now), now, frame_index=index)
        new_boxes = [box for box in boxes if box.id not in seen_ids]
        seen_ids.update(box.id for box in new_boxes)
        backfill(list(buffer)[-config.BACKFILL_FRAMES :], new_boxes)
        frame.boxes = boxes
        frame.faces_done = True
        watchdog.beat_faces(now)

        buffer.append(frame)
        if len(buffer) > config.DELAY_BUFFER_FRAMES:
            release_oldest(now)
        index += 1
    source.release()

    while buffer:
        release_oldest(now)
    return censored, raw_frames_with_faces, leaks


def _write_mjpg(frames: list[CensoredFrame], path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_FOURCC)
    writer = cv2.VideoWriter(str(path), fourcc, config.FPS, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame.image)
    writer.release()


def _read_back(path: Path) -> list[np.ndarray]:
    video = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = video.read()
        if not ok:
            break
        frames.append(image)
    video.release()
    return frames


@pytest.mark.parametrize("clip_name", ["faces.mov", "crossing.mov"])
def test_no_face_survives_into_the_recording(clip_name: str, tmp_path: Path) -> None:
    clip = _clip(clip_name)
    censored, raw_frames_with_faces, leaks = _censor_clip(clip)
    assert raw_frames_with_faces > 0, "the clip should contain faces"
    assert [frame.index for frame in censored] == list(range(len(censored)))

    # The recording holds exactly these frames.
    recording = tmp_path / f"leak{config.RECORDING_EXTENSION}"
    _write_mjpg(censored, recording)
    assert len(_read_back(recording)) == len(censored)

    confident = [(i, conf) for i, conf in leaks if conf >= CONFIDENT_FACE_SCORE]
    leak_frames = {i for i, _conf in leaks}
    print(f"leak check {clip_name}: unpixelated raw faces in {len(leak_frames)} of {len(censored)} frames, {len(confident)} confident")
    assert confident == [], f"confident faces left unpixelated (frame, score): {confident}"
    assert len(leak_frames) <= MAX_WEAK_LEAK_FRAME_SHARE * len(censored), (
        f"weak faces left unpixelated in {len(leak_frames)} of {len(censored)} frames"
    )
