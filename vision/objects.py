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
# WP19: the generic "sign with text" and "signboard" prompts are gone. On the clip their only
# boxes at YOLOE_CONF were a square sign under the traffic lights that the model, asked alone,
# calls "traffic sign" (0.76) rather than "sign with text" (0.38), and a no-parking sign.
_PROMPT_LABELS: dict[str, str | None] = {
    "license plate": "license plate",
    "street name sign": "street sign",
    "street name plate": "street sign",
    "house number": "house number",
    "storefront sign": "storefront sign",
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

# Street-name signs are kept from this confidence, the other labels from config.YOLOE_CONF.
# Measured on the clip (every 4th frame, after suppression): NCNN at 320 scores the blades
# 0.14-0.18 and the far blue sign 0.11-0.13, all wide but one edge-clipped box; 0.05 adds
# boxes no prompt names well. PyTorch at 640 finds the same 8 wide blades at 0.2, 0.15 and 0.1.
_STREET_SIGN_CONF = 0.1
# A kept box is dropped when a more confident ignored-prompt box overlaps it this much
# (IoU, or the share of the smaller box inside the other): "the same object". On the clip
# at conf 0.05 this matched 13 NCNN pairs, every kept box below 0.1, none above 0.25.
_SUPPRESS_IOU = 0.5
_SUPPRESS_INSIDE = 0.8


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
        # Warm-up: the first call compiles kernels and, for NCNN, reads the model's class names.
        self.detect(np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Contract-labelled boxes, minus any box an ignored prompt claims more confidently.

        The model runs at the lowest per-label cut, so a confident ignored box (a stop sign)
        can still suppress a weak kept box on the same object before the per-label cut.
        Ultralytics' NMS is per class, so it never does this itself.
        """
        min_conf = min(config.YOLOE_CONF, _STREET_SIGN_CONF)
        result = self._model.predict(image, device=self._device, imgsz=self._imgsz, conf=min_conf, verbose=False)[0]
        boxes = result.boxes
        rows = [
            (_label(result.names[int(cls)]), xyxy, float(conf))
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]
        ignored = [(xyxy, conf) for label, xyxy, conf in rows if label is None]
        height, width = image.shape[:2]
        detections = []
        for label, xyxy, conf in rows:
            if label is None or conf < _min_conf(label):  # None: a distractor prompt, never blurred
                continue
            if any(ignored_conf > conf and _same_object(xyxy, ignored_xyxy) for ignored_xyxy, ignored_conf in ignored):
                continue
            detections.append(Detection(label=label, xyxy=_clipped_box(xyxy, width, height), conf=conf))
        return detections


def _min_conf(label: str) -> float:
    return _STREET_SIGN_CONF if label == "street sign" else config.YOLOE_CONF


def _same_object(a: list[float], b: list[float]) -> bool:
    """IoU >= _SUPPRESS_IOU, or the smaller box at least _SUPPRESS_INSIDE inside the other."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return False
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) >= _SUPPRESS_IOU or inter / min(area_a, area_b) >= _SUPPRESS_INSIDE


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
