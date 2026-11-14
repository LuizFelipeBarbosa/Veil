"""YOLOE-26n on its own thread (WP5): PyTorch on the Mac, NCNN on the Pi, chosen by config."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

import numpy as np

import config
from contracts import OBJECT_CLASSES, BufferedFrame, Detection

log = logging.getLogger(__name__)

# The text prompts YOLOE is given, and the contract label each one reports as.
# The NCNN export bakes in exactly these prompts, in this order.
#
# "street sign" means a sign that names a street (intersection blade signs, wall plates).
# Measured on tests/data/intersection.mov (WP16): the old "traffic sign" prompt fired on
# stop, one-way and other regulatory signs too, so it is gone. A box goes to its
# best-matching prompt, so the traffic-sign prompts below (label None) pull those signs
# away from the street-name prompts and their boxes are dropped. "traffic sign" itself is
# deliberately not a distractor: it outscores "street name sign" on the real blade signs.
_PROMPT_LABELS: dict[str, str | None] = {
    "license plate": "license plate",
    "street name sign": "street sign",
    "street name plate": "street sign",
    "house number": "house number",
    "storefront sign": "storefront sign",
    "sign with text": "storefront sign",
    "signboard": "storefront sign",
    # Distractors: detected so they are not mistaken for the above, then ignored.
    "stop sign": None,
    "one way sign": None,
    "traffic light": None,
    "speed limit sign": None,
    "parking sign": None,
    "no parking sign": None,
    "yield sign": None,
    "do not enter sign": None,
    "pedestrian crossing sign": None,
}
PROMPTS = tuple(_PROMPT_LABELS)

NCNN_IMGSZ = 320  # the NCNN export is made and run at this size (Brief: Pi, 320 px)
_DETECT_ONLY_CFG = "yoloe-26n.yaml"  # same weights without the mask branch: same boxes, faster
_TEXT_ENCODER = "mobileclip2_b.ts"  # what YOLOE-26's get_text_pe loads, looked up in the working directory
_IDLE_WAIT_S = 0.005


def load_prompted_model():
    """YOLOE-26n (detect-only head) from config.YOLOE_PT_MODEL with PROMPTS set as its classes.

    Shared with scripts/export_ncnn.py so the export bakes in the same prompts.
    """
    from ultralytics import YOLOE

    model = YOLOE(_DETECT_ONLY_CFG).load(str(config.YOLOE_PT_MODEL))
    model.set_classes(list(PROMPTS), _text_embeddings(model, list(PROMPTS)))
    return model


def _text_embeddings(model, prompts: list[str]):
    """get_text_pe, run from config.WEIGHTS_DIR so ultralytics finds the text encoder there instead of downloading it."""
    if not (config.WEIGHTS_DIR / _TEXT_ENCODER).exists():
        raise FileNotFoundError(f"{_TEXT_ENCODER} missing from {config.WEIGHTS_DIR}; refusing to let ultralytics download it")
    previous_dir = os.getcwd()
    os.chdir(config.WEIGHTS_DIR)
    try:
        return model.get_text_pe(prompts)
    finally:
        os.chdir(previous_dir)


class ObjectDetector:
    """Loads the model for config.YOLOE_RUNTIME with the 4 fixed classes."""

    def __init__(self) -> None:
        runtime = config.YOLOE_RUNTIME
        if runtime == "pytorch":
            self._model = load_prompted_model()
            self._device = config.YOLOE_DEVICE
            self._imgsz = config.YOLOE_IMGSZ
        elif runtime == "ncnn":
            from ultralytics import YOLO

            self._model = YOLO(str(config.YOLOE_NCNN_MODEL), task="detect")
            self._device = "cpu"
            self._imgsz = NCNN_IMGSZ
        else:
            raise ValueError(f"unknown YOLOE_RUNTIME {runtime!r}; use pytorch or ncnn")
        self._conf = config.YOLOE_CONF
        # Warm-up: the first call compiles kernels and, for NCNN, reads the model's class names.
        self.detect(np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8))

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(image, device=self._device, imgsz=self._imgsz, conf=self._conf, verbose=False)[0]
        height, width = image.shape[:2]
        boxes = result.boxes
        detections = []
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            label = _label(result.names[int(cls)])
            if label is not None:  # None: a distractor prompt, never blurred
                detections.append(Detection(label=label, xyxy=_clipped_box(xyxy, width, height), conf=float(conf)))
        return detections


def _label(prompt: str) -> str | None:
    """The contract label for a model class name, or None for a distractor.

    An unknown name means a stale export: fail loudly.
    """
    try:
        return _PROMPT_LABELS[prompt]
    except KeyError:
        raise ValueError(f"model class {prompt!r} is not one of the prompts; re-run scripts/export_ncnn.py") from None


def _clipped_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Integer pixels, rounded outwards and clipped to the frame."""
    x1, y1, x2, y2 = xyxy
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(width, int(np.ceil(x2))),
        min(height, int(np.ceil(y2))),
    )


class ObjectsThread(threading.Thread):
    """Runs the detector on the newest buffered frame whenever it is free."""

    def __init__(self, detector: ObjectDetector, newest_frame: Callable[[], BufferedFrame | None]) -> None:
        super().__init__(name="objects", daemon=True)
        self._detector = detector
        self._newest_frame = newest_frame
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[int, float, list[Detection]] | None = None

    def latest(self) -> tuple[int, float, list[Detection]] | None:
        """(frame_index, finished_at, detections) of the last finished detection, or None."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        last_index: int | None = None
        while not self._stopped.is_set():
            index = self._detect_newest(last_index)
            if index is None:
                self._stopped.wait(_IDLE_WAIT_S)
            else:
                last_index = index

    def _detect_newest(self, last_index: int | None) -> int | None:
        """Detect on the newest frame if it is new and return its index; None when there is nothing new.

        The frame is held only inside this call, so the raw image is never kept by the thread.
        """
        frame = self._newest_frame()
        if frame is None or frame.index == last_index:
            return None
        try:
            detections = self._detector.detect(frame.image)
        except Exception:
            # Nothing is stored, so the pipeline's object watchdog fails closed.
            log.exception("object detection failed on frame %d", frame.index)
            return frame.index
        with self._lock:
            self._latest = (frame.index, time.monotonic(), detections)
        return frame.index
