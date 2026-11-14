"""WP5: YOLOE-26n detector (PyTorch and the NCNN export) and the objects thread."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import yaml

import config
from contracts import OBJECT_CLASSES, BufferedFrame, Detection
from fakes.capture import FileSource
from vision.objects import _PROMPT_LABELS, PROMPTS, ObjectDetector, ObjectsThread

# Lives only in the main checkout, never in git. Frames from it stay in memory.
INTERSECTION_CLIP = Path("/Users/luiz/Developer/privacy-cam/tests/data/intersection.mov")
SAMPLE_EVERY = 10  # frames; the clip has about 320
MIN_SHARE_WITH_DETECTIONS = 0.25  # PyTorch measured 21/33, NCNN at 320 px 12/33


# --- prompts ---------------------------------------------------------------


def test_every_prompt_reports_a_contract_label_and_every_label_has_a_prompt() -> None:
    assert set(_PROMPT_LABELS.values()) == set(OBJECT_CLASSES)
    assert set(OBJECT_CLASSES) <= set(PROMPTS)


def test_the_ncnn_export_bakes_in_the_same_prompts() -> None:
    metadata = Path(config.YOLOE_NCNN_MODEL) / "metadata.yaml"
    if not metadata.exists():
        pytest.skip("no NCNN export; run scripts/export_ncnn.py")
    names = yaml.safe_load(metadata.read_text())["names"]
    assert [names[i] for i in sorted(names)] == list(PROMPTS)


# --- detector on the intersection clip -------------------------------------


@pytest.fixture(scope="module")
def intersection_frames() -> list[np.ndarray]:
    if not INTERSECTION_CLIP.exists():
        pytest.skip("intersection.mov is not in the main checkout")
    source = FileSource(INTERSECTION_CLIP, loop=False)
    frames, index = [], 0
    while (image := source.read()) is not None:
        if index % SAMPLE_EVERY == 0:
            frames.append(image)
        index += 1
    source.release()
    return frames


def _check_detections_on_clip(detector: ObjectDetector, frames: list[np.ndarray]) -> None:
    frames_with_detections = 0
    for image in frames:
        detections = detector.detect(image)
        frames_with_detections += bool(detections)
        for detection in detections:
            assert detection.label in OBJECT_CLASSES
            x1, y1, x2, y2 = detection.xyxy
            assert all(isinstance(v, int) for v in detection.xyxy)
            assert 0 <= x1 < x2 <= config.FRAME_WIDTH and 0 <= y1 < y2 <= config.FRAME_HEIGHT
            assert config.YOLOE_CONF <= detection.conf <= 1.0
    assert frames_with_detections >= MIN_SHARE_WITH_DETECTIONS * len(frames)


def test_pytorch_detector_finds_signs_and_plates_on_the_clip(intersection_frames: list[np.ndarray]) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "YOLOE_RUNTIME", "pytorch")
        detector = ObjectDetector()
    _check_detections_on_clip(detector, intersection_frames)


def test_ncnn_export_loads_and_finds_signs_and_plates_on_the_clip(intersection_frames: list[np.ndarray]) -> None:
    """Slow-tolerant: NCNN on the Mac CPU. Skipped until scripts/export_ncnn.py has run."""
    if not Path(config.YOLOE_NCNN_MODEL).is_dir():
        pytest.skip("no NCNN export; run scripts/export_ncnn.py")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "YOLOE_RUNTIME", "ncnn")
        detector = ObjectDetector()
    _check_detections_on_clip(detector, intersection_frames)


# --- the objects thread, with a fake detector ------------------------------


class FakeDetector:
    """Returns one plate per call and counts calls; can be told to raise. Keeps no image."""

    def __init__(self) -> None:
        self.calls = 0
        self.raise_next = False

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("detector exploded")
        return [Detection(label="license plate", xyxy=(1, 2, 30, 12), conf=0.5)]


class NewestFrame:
    def __init__(self) -> None:
        self.frame: BufferedFrame | None = None

    def __call__(self) -> BufferedFrame | None:
        return self.frame

    def set(self, index: int) -> None:
        image = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        self.frame = BufferedFrame(index=index, timestamp=time.monotonic(), image=image)


def wait_for(condition: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


@pytest.fixture
def running() -> tuple[ObjectsThread, FakeDetector, NewestFrame]:
    detector, newest = FakeDetector(), NewestFrame()
    thread = ObjectsThread(detector, newest)  # type: ignore[arg-type]
    thread.start()
    yield thread, detector, newest
    thread.stop()
    thread.join(timeout=1.0)


def test_thread_is_a_daemon_named_objects() -> None:
    thread = ObjectsThread(FakeDetector(), NewestFrame())  # type: ignore[arg-type]
    assert thread.name == "objects" and thread.daemon


def test_nothing_is_published_before_a_frame_arrives(running) -> None:
    thread, detector, _ = running
    time.sleep(0.05)
    assert thread.latest() is None and detector.calls == 0


def test_results_are_tagged_with_their_frame_index(running) -> None:
    thread, _, newest = running
    newest.set(7)
    before = time.monotonic()
    assert wait_for(lambda: thread.latest() is not None)
    frame_index, finished_at, detections = thread.latest()
    assert frame_index == 7
    assert before <= finished_at <= time.monotonic()
    assert [d.label for d in detections] == ["license plate"]

    newest.set(8)
    assert wait_for(lambda: thread.latest()[0] == 8)


def test_an_unchanged_frame_is_not_detected_twice(running) -> None:
    thread, detector, newest = running
    newest.set(1)
    assert wait_for(lambda: detector.calls == 1)
    time.sleep(0.1)
    assert detector.calls == 1
    newest.set(2)
    assert wait_for(lambda: detector.calls == 2)


def test_a_raising_detector_stores_nothing_and_the_thread_survives(running) -> None:
    thread, detector, newest = running
    newest.set(1)
    assert wait_for(lambda: thread.latest() is not None)

    detector.raise_next = True
    newest.set(2)
    assert wait_for(lambda: detector.calls == 2)
    time.sleep(0.05)
    assert thread.latest()[0] == 1  # frame 2 failed: nothing new, so the watchdog can trip
    assert thread.is_alive()

    newest.set(3)
    assert wait_for(lambda: thread.latest()[0] == 3)


def test_stop_ends_the_thread_promptly(running) -> None:
    thread, _, newest = running
    newest.set(1)
    assert wait_for(lambda: thread.latest() is not None)
    started = time.monotonic()
    thread.stop()
    thread.join(timeout=0.5)
    assert not thread.is_alive()
    assert time.monotonic() - started < 0.5


def test_the_thread_keeps_no_reference_to_the_image(running) -> None:
    thread, _, newest = running
    newest.set(1)
    image_ref = weakref.ref(newest.frame.image)
    assert wait_for(lambda: thread.latest() is not None)
    newest.frame = None
    time.sleep(0.05)  # let the loop move past the frame
    gc.collect()
    assert image_ref() is None
