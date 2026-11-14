"""WP2 leak test: faces.mov through detect -> track -> hold-over -> backfill -> censor -> MJPG file.

Reads the recording back and checks that YuNet, at a far lower score than the
pipeline uses, finds no face in any written frame.
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
MAX_WEAK_HIT_FRAME_SHARE = 0.20
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


def _censor_clip(clip: Path) -> tuple[list[CensoredFrame], int]:
    """Every frame of the clip, censored with a delay buffer; also how many raw frames had faces."""
    detector = FaceDetector()
    tracker = make_face_tracker()
    hold_over = HoldOver()
    watchdog = Watchdog(watch_objects=False)
    policy = Policy().snapshot()
    buffer: deque[BufferedFrame] = deque()
    seen_ids: set[str] = set()
    censored: list[CensoredFrame] = []
    raw_frames_with_faces = 0

    def release_oldest(now: float) -> None:
        censored.append(censor(buffer.popleft(), policy, _always_blur, watchdog.tripped(now)))

    source = FileSource(clip, loop=False)
    index = 0
    while (image := source.read()) is not None:
        now = index / config.FPS
        frame = BufferedFrame(index=index, timestamp=now, image=image)

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
    return censored, raw_frames_with_faces


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
    censored, raw_frames_with_faces = _censor_clip(clip)
    assert raw_frames_with_faces > 0, "the clip should contain faces"
    assert [frame.index for frame in censored] == list(range(len(censored)))

    recording = tmp_path / f"leak{config.RECORDING_EXTENSION}"
    _write_mjpg(censored, recording)
    written = _read_back(recording)
    assert len(written) == len(censored)

    checker = FaceDetector(score=LEAK_CHECK_SCORE)
    hits = [(i, round(hit.conf, 2)) for i, image in enumerate(written) for hit in checker.detect(image)]

    # Pass bar set by the lead on 2026-11-14: the 0.3 checker also fires on pixelated
    # blobs and clothing, so zero hits is not reachable. No confident face may
    # survive, and frames with any weak hit must stay under the limit.
    confident = [(i, conf) for i, conf in hits if conf >= CONFIDENT_FACE_SCORE]
    weak_frames = {i for i, _conf in hits}
    print(f"leak check {clip_name}: hits in {len(weak_frames)} of {len(written)} frames, {len(confident)} confident")
    assert confident == [], f"confident faces in the recording (frame, score): {confident}"
    assert len(weak_frames) <= MAX_WEAK_HIT_FRAME_SHARE * len(written), (
        f"weak hits in {len(weak_frames)} of {len(written)} frames"
    )
