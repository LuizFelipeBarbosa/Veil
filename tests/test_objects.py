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
from vision.objects import _PROMPT_LABELS, _STREET_SIGN_CONF, PROMPTS, ObjectDetector, ObjectsThread

# Lives only in the main checkout, never in git. Frames from it stay in memory.
INTERSECTION_CLIP = Path("/Users/luiz/Developer/privacy-cam/tests/data/intersection.mov")
SAMPLE_EVERY = 10  # frames; the clip has about 320
IGNORED_PROMPTS = {prompt for prompt, label in _PROMPT_LABELS.items() if label is None}

# WP16 measurements, every 10th frame (33 frames). Before, "traffic sign" reported as
# "street sign" on 19 frames, square stop/regulatory signs included. After, PyTorch at 640
# finds 3 boxes, all wide blades high in the frame (w/h 2.7 to 3.3); NCNN at 320 finds none
# at YOLOE_CONF (the same blades score 0.1 to 0.25 there).
# WP19: street-name signs are kept from _STREET_SIGN_CONF, and ignored prompts win overlaps.
# WP16 measured 9/33; WP19 6/33: the 3 lost frames held only the square traffic sign under
# the lights that "sign with text" reported as a storefront sign.
PYTORCH_MIN_SHARE_WITH_DETECTIONS = 0.15
MAX_STREET_SIGN_BOXES = 8  # WP16: 3 PyTorch, 0 NCNN; was 29 PyTorch, 17 NCNN
MIN_STREET_SIGN_ASPECT = 1.5  # a street-name blade is wide; stop and most regulatory signs are not
WP16_PYTORCH_STREET_SIGN_FRAMES = 3  # of 33, at YOLOE_CONF, before WP19
# WP19: the largest object box on the clip at the new cuts is 0.6% of the frame. The "big
# areas of buildings" the lead saw came from the face detector, not from here.
MAX_OBJECT_BOX_SHARE = 0.04


# --- prompts ---------------------------------------------------------------


def test_every_prompt_reports_a_contract_label_or_is_ignored_and_every_label_has_a_prompt() -> None:
    assert set(_PROMPT_LABELS.values()) - {None} == set(OBJECT_CLASSES)
    assert PROMPTS == tuple(_PROMPT_LABELS)


def test_traffic_signs_are_ignored_and_street_signs_are_named_streets() -> None:
    assert {"stop sign", "one way sign", "traffic light", "speed limit sign"} <= IGNORED_PROMPTS
    street_prompts = [prompt for prompt, label in _PROMPT_LABELS.items() if label == "street sign"]
    assert street_prompts and all("street name" in prompt for prompt in street_prompts)
    assert "traffic sign" not in street_prompts


class _FakeBoxes:
    def __init__(self, rows: list[tuple[tuple[float, float, float, float], float, int]]) -> None:
        self.xyxy = np.array([xyxy for xyxy, _, _ in rows], dtype=np.float32).reshape(-1, 4)
        self.conf = np.array([conf for _, conf, _ in rows], dtype=np.float32)
        self.cls = np.array([cls for _, _, cls in rows], dtype=np.float32)


class _FakeModel:
    """Returns the given (xyxy, conf, prompt) rows at or above the asked conf, like ultralytics."""

    def __init__(self, rows: list[tuple[tuple[float, float, float, float], float, str]]) -> None:
        self._rows = rows

    def predict(self, image: np.ndarray, conf: float, **_kwargs) -> list:
        rows = [(xyxy, c, PROMPTS.index(prompt)) for xyxy, c, prompt in self._rows if c >= conf]
        result = type("Result", (), {"names": dict(enumerate(PROMPTS)), "boxes": _FakeBoxes(rows)})()
        return [result]


def _fake_detect(rows: list[tuple[tuple[float, float, float, float], float, str]]) -> list[Detection]:
    detector = object.__new__(ObjectDetector)  # skip loading a real model
    detector._model, detector._device, detector._imgsz = _FakeModel(rows), "cpu", 640
    return detector.detect(np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8))


def test_ignored_prompts_are_dropped_from_the_output() -> None:
    # One box per prompt, side by side so none overlaps another.
    rows = [((10.0 + 40 * i, 20.0, 40.0 + 40 * i, 40.0), 0.5, prompt) for i, prompt in enumerate(PROMPTS)]
    detections = _fake_detect(rows)
    assert len(detections) == len(PROMPTS) - len(IGNORED_PROMPTS)
    assert {d.label for d in detections} == set(OBJECT_CLASSES)


def test_a_more_confident_ignored_box_on_the_same_object_drops_the_kept_box() -> None:
    sign = (100.0, 100.0, 200.0, 200.0)
    # A 0.9 stop sign drops a 0.3 storefront box, and a 0.15 street-name box inside it.
    assert _fake_detect([(sign, 0.9, "stop sign"), (sign, 0.3, "storefront sign")]) == []
    assert _fake_detect([(sign, 0.9, "stop sign"), ((110.0, 140.0, 190.0, 165.0), 0.15, "street name sign")]) == []
    # A less confident ignored box does not.
    [kept] = _fake_detect([(sign, 0.2, "stop sign"), (sign, 0.3, "storefront sign")])
    assert kept.label == "storefront sign" and kept.conf == pytest.approx(0.3)
    # Nor does a confident one elsewhere, or one that only grazes the kept box.
    [kept] = _fake_detect([((300.0, 300.0, 400.0, 400.0), 0.9, "stop sign"), (sign, 0.3, "storefront sign")])
    assert kept.label == "storefront sign"
    [kept] = _fake_detect([((170.0, 100.0, 270.0, 200.0), 0.9, "stop sign"), (sign, 0.3, "storefront sign")])
    assert kept.label == "storefront sign"


def test_street_signs_have_their_own_lower_confidence_cut() -> None:
    assert _STREET_SIGN_CONF < config.YOLOE_CONF
    between = (_STREET_SIGN_CONF + config.YOLOE_CONF) / 2
    rows = [
        ((10.0, 10.0, 70.0, 30.0), between, "street name sign"),
        ((100.0, 10.0, 160.0, 30.0), between, "street name plate"),
        ((200.0, 10.0, 260.0, 30.0), between, "license plate"),
        ((300.0, 10.0, 360.0, 70.0), between, "storefront sign"),
        ((400.0, 10.0, 460.0, 30.0), _STREET_SIGN_CONF * 0.9, "street name sign"),
        ((500.0, 10.0, 560.0, 30.0), config.YOLOE_CONF, "license plate"),
    ]
    assert [d.label for d in _fake_detect(rows)] == ["street sign", "street sign", "license plate"]


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


def _check_detections_on_clip(detector: ObjectDetector, frames: list[np.ndarray]) -> tuple[int, int, list[tuple]]:
    """Checks every box; returns (frames with any detection, frames with a street sign, street sign boxes)."""
    frames_with_detections, frames_with_street_signs, street_signs = 0, 0, []
    max_area = MAX_OBJECT_BOX_SHARE * config.FRAME_WIDTH * config.FRAME_HEIGHT
    for image in frames:
        detections = detector.detect(image)
        ignored = _ignored_boxes(detector, image)
        for detection in detections:
            assert detection.label in OBJECT_CLASSES
            x1, y1, x2, y2 = detection.xyxy
            assert all(isinstance(v, int) for v in detection.xyxy)
            assert 0 <= x1 < x2 <= config.FRAME_WIDTH and 0 <= y1 < y2 <= config.FRAME_HEIGHT
            min_conf = _STREET_SIGN_CONF if detection.label == "street sign" else config.YOLOE_CONF
            assert min_conf <= detection.conf <= 1.0
            assert (x2 - x1) * (y2 - y1) <= max_area
            assert not any(conf > detection.conf and _overlap(detection.xyxy, box) >= 0.5 for box, conf in ignored)
        frames_with_detections += bool(detections)
        street_signs += [d.xyxy for d in detections if d.label == "street sign"]
        frames_with_street_signs += any(d.label == "street sign" for d in detections)
    return frames_with_detections, frames_with_street_signs, street_signs


def _ignored_boxes(detector: ObjectDetector, image: np.ndarray) -> list[tuple[list[float], float]]:
    """The ignored-prompt boxes the model sees on this image down to the lowest cut."""
    result = detector._model.predict(
        image, device=detector._device, imgsz=detector._imgsz, conf=_STREET_SIGN_CONF, verbose=False
    )[0]
    boxes = result.boxes
    return [
        (xyxy, conf)
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        if result.names[int(cls)] in IGNORED_PROMPTS
    ]


def _overlap(a, b) -> float:
    """The larger of IoU and the share of the smaller box inside the other."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return max(inter / (area_a + area_b - inter), inter / min(area_a, area_b))


def _check_street_signs_are_wide_and_rare(street_signs: list[tuple]) -> None:
    """Regression for WP16: no more square stop/regulatory signs reported as street signs.

    A box cut by the left or right frame edge has lost its width, so its shape is not checked.
    """
    assert len(street_signs) <= MAX_STREET_SIGN_BOXES
    whole = [(x1, y1, x2, y2) for x1, y1, x2, y2 in street_signs if 0 < x1 and x2 < config.FRAME_WIDTH]
    wide = [(x2 - x1) / (y2 - y1) >= MIN_STREET_SIGN_ASPECT for x1, y1, x2, y2 in whole]
    assert sum(wide) >= 0.8 * len(wide)


def test_pytorch_detector_finds_street_name_signs_and_plates_on_the_clip(intersection_frames: list[np.ndarray]) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "YOLOE_RUNTIME", "pytorch")
        detector = ObjectDetector()
    frames_with_detections, frames_with_street_signs, street_signs = _check_detections_on_clip(detector, intersection_frames)
    assert frames_with_detections >= PYTORCH_MIN_SHARE_WITH_DETECTIONS * len(intersection_frames)
    assert frames_with_street_signs > WP16_PYTORCH_STREET_SIGN_FRAMES
    _check_street_signs_are_wide_and_rare(street_signs)


def test_ncnn_export_loads_and_reports_only_contract_labels_on_the_clip(intersection_frames: list[np.ndarray]) -> None:
    """Slow-tolerant: NCNN on the Mac CPU. Skipped until scripts/export_ncnn.py has run.

    Street-name blades clear _STREET_SIGN_CONF here; at YOLOE_CONF nothing on this clip did (WP16: 0/33).
    """
    if not Path(config.YOLOE_NCNN_MODEL).is_dir():
        pytest.skip("no NCNN export; run scripts/export_ncnn.py")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "YOLOE_RUNTIME", "ncnn")
        detector = ObjectDetector()
    _, frames_with_street_signs, street_signs = _check_detections_on_clip(detector, intersection_frames)
    assert frames_with_street_signs > 0
    _check_street_signs_are_wide_and_rare(street_signs)


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
