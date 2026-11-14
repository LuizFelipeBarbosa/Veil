"""WP1: the MJPG recorder takes CensoredFrame only."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from contracts import BufferedFrame, CensoredFrame
from recorder import Recorder


def _censored(index: int) -> CensoredFrame:
    image = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), index * 8 % 256, dtype=np.uint8)
    return CensoredFrame(image=image, index=index, fail_closed=False)


def test_recording_reopens_as_mjpg_with_every_frame(tmp_path: Path) -> None:
    directory = tmp_path / "recordings"  # does not exist yet
    recorder = Recorder(directory)
    assert not recorder.is_recording and recorder.elapsed_s == 0.0

    path = recorder.start()
    assert recorder.is_recording
    assert path.parent == directory and path.suffix == config.RECORDING_EXTENSION
    for index in range(12):
        recorder.write(_censored(index))
    assert recorder.stop() == path
    assert not recorder.is_recording

    video = cv2.VideoCapture(str(path))
    try:
        assert video.isOpened()
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
        frame_count = 0
        while video.read()[0]:
            frame_count += 1
    finally:
        video.release()
    assert (width, height) == (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    assert fourcc == config.RECORDING_FOURCC
    assert frame_count == 12


@pytest.mark.parametrize(
    "not_censored",
    [
        np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8),
        BufferedFrame(index=0, timestamp=0.0, image=np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)),
    ],
    ids=["numpy array", "BufferedFrame"],
)
def test_raw_frames_are_refused(tmp_path: Path, not_censored: object) -> None:
    recorder = Recorder(tmp_path)
    with pytest.raises(TypeError):
        recorder.write(not_censored)  # refused even when not recording
    recorder.start()
    try:
        with pytest.raises(TypeError):
            recorder.write(not_censored)
    finally:
        recorder.stop()


def test_write_is_ignored_when_not_recording_and_stop_without_start(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.write(_censored(0))
    assert recorder.stop() is None
    assert list(tmp_path.iterdir()) == []


def test_two_recordings_get_two_files_and_elapsed_counts_up(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    first = recorder.start()
    time.sleep(0.05)
    assert recorder.elapsed_s >= 0.04
    recorder.stop()
    second = recorder.start()
    recorder.stop()
    assert first != second and first.exists() and second.exists()
