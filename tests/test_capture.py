"""WP1: ring buffer, capture thread and sources. Never opens the camera."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import capture
import config
from capture import CaptureThread, RingBuffer, open_source
from contracts import TrackedBox
from fakes.capture import FileSource

SYNTHETIC_CLIP = Path(__file__).parent / "data" / "synthetic.mp4"


def _image(value: int = 0) -> np.ndarray:
    return np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), value, dtype=np.uint8)


def _box(frame_index: int) -> TrackedBox:
    return TrackedBox(id="face#1", label="face", xyxy=(0, 0, 10, 10), conf=0.9, frame_index=frame_index, last_seen=0.0)


# --- ring buffer ---


def test_pop_oldest_waits_for_a_full_delay_then_returns_frames_in_order() -> None:
    ring = RingBuffer()
    for i in range(config.DELAY_BUFFER_FRAMES - 1):
        ring.push(_image(), timestamp=float(i))
        assert ring.pop_oldest() is None

    ring.push(_image(), timestamp=99.0)
    assert ring.pop_oldest().index == 0
    assert ring.pop_oldest() is None  # back below capacity

    ring.push(_image(), timestamp=100.0)
    ring.push(_image(), timestamp=101.0)
    assert [ring.pop_oldest().index, ring.pop_oldest().index] == [1, 2]


def test_each_frame_has_its_own_box_list() -> None:
    ring = RingBuffer(capacity=3)
    frames = [ring.push(_image(), timestamp=float(i)) for i in range(3)]
    assert all(frame.boxes == [] and not frame.faces_done for frame in frames)

    frames[1].boxes.append(_box(frames[1].index))
    assert frames[0].boxes == [] and frames[2].boxes == []
    assert frames[0].boxes is not frames[2].boxes

    popped = ring.pop_oldest()
    assert popped is frames[0] and popped.boxes == []


def test_recent_and_newest() -> None:
    ring = RingBuffer(capacity=12)
    assert ring.newest() is None and ring.recent(10) == []
    for i in range(12):
        ring.push(_image(), timestamp=float(i))

    assert [frame.index for frame in ring.recent(10)] == list(range(2, 12))
    assert [frame.index for frame in ring.recent(50)] == list(range(12))
    assert ring.recent(0) == []
    assert ring.newest().index == 11


def test_ring_is_bounded_when_vision_falls_behind() -> None:
    ring = RingBuffer(capacity=4)
    for i in range(20):
        ring.push(_image(), timestamp=float(i))
    assert len(ring) == 8
    assert ring.pop_oldest().index == 12


# --- capture thread ---


def test_wait_for_frame_times_out_without_frames() -> None:
    thread = CaptureThread(_NeverSource(), RingBuffer())  # not started: nothing arrives
    started = time.monotonic()
    assert thread.wait_for_frame(after_index=-1, timeout=0.05) is None
    assert 0.04 <= time.monotonic() - started < 1.0


def test_wait_for_frame_returns_the_next_frame_in_order() -> None:
    ring = RingBuffer()
    thread = CaptureThread(_NeverSource(), ring)
    for i in range(3):
        ring.push(_image(), timestamp=float(i))
    assert thread.wait_for_frame(after_index=-1, timeout=0.01).index == 0
    assert thread.wait_for_frame(after_index=1, timeout=0.01).index == 2
    assert thread.wait_for_frame(after_index=2, timeout=0.01) is None


def test_capture_thread_fills_the_ring_from_a_file_source() -> None:
    ring = RingBuffer()
    thread = CaptureThread(FileSource(SYNTHETIC_CLIP), ring, fps=200)
    thread.start()
    try:
        last = -1
        seen = []
        while len(seen) < config.DELAY_BUFFER_FRAMES:
            frame = thread.wait_for_frame(last, timeout=2.0)
            assert frame is not None
            seen.append(frame.index)
            last = frame.index
    finally:
        thread.stop()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert seen == list(range(config.DELAY_BUFFER_FRAMES))
    oldest = ring.pop_oldest()
    assert oldest.index == 0 and oldest.image.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)


def test_wait_for_frame_returns_none_at_end_of_source() -> None:
    ring = RingBuffer()
    thread = CaptureThread(FileSource(SYNTHETIC_CLIP, loop=False), ring, fps=500)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    newest = ring.newest()
    assert newest.index == 29  # 30-frame clip
    started = time.monotonic()
    assert thread.wait_for_frame(after_index=newest.index, timeout=5.0) is None
    assert time.monotonic() - started < 1.0  # returns at once, not after the timeout


def test_capture_thread_paces_at_config_fps() -> None:
    ring = RingBuffer()
    thread = CaptureThread(FileSource(SYNTHETIC_CLIP), ring)
    thread.start()
    time.sleep(0.5)
    thread.stop()
    thread.join(timeout=2.0)
    expected = 0.5 * config.FPS
    assert expected - 3 <= ring.newest().index + 1 <= expected + 3


# --- sources ---


def test_open_source_with_a_path_gives_a_file_source() -> None:
    source = open_source(str(SYNTHETIC_CLIP))
    try:
        assert isinstance(source, FileSource)
        assert source.read().shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)
    finally:
        source.release()


def test_open_source_none_asks_the_camera_for_config_size_and_resizes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeVideoCapture()
    monkeypatch.setattr(capture.cv2, "VideoCapture", fake)
    source = open_source(None)

    assert fake.opened_index == config.CAMERA_INDEX
    assert fake.settings == {
        capture.cv2.CAP_PROP_FRAME_WIDTH: config.FRAME_WIDTH,
        capture.cv2.CAP_PROP_FRAME_HEIGHT: config.FRAME_HEIGHT,
        capture.cv2.CAP_PROP_FPS: config.FPS,
    }
    assert source.read().shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)  # camera gave 1280x720
    assert source.read() is None  # a failed grab ends the source
    source.release()
    assert fake.released


class _NeverSource:
    def read(self) -> np.ndarray | None:
        return None

    def release(self) -> None:
        pass


class _FakeVideoCapture:
    """Stands in for cv2.VideoCapture so the test never touches the camera."""

    def __init__(self) -> None:
        self.opened_index: int | None = None
        self.settings: dict[int, float] = {}
        self.released = False
        self._reads = [(True, np.zeros((720, 1280, 3), dtype=np.uint8)), (False, None)]

    def __call__(self, index: int) -> "_FakeVideoCapture":
        self.opened_index = index
        return self

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.settings[prop] = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._reads.pop(0)

    def release(self) -> None:
        self.released = True
