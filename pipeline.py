"""The vision loop: raw frame in, CensoredFrame out. Orchestrator-owned (WP0, WP8).

This is the one place raw frames turn into output, and it fails closed: while a
detector is missing, and whenever censoring raises, the whole frame is pixelated.

Per captured frame: detect on the newest frame (so late detections can be
backfilled), then censor the oldest buffered frame, one second behind, with its
own boxes, and hand it to the preview and the recorder.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

import cv2
import numpy as np

import config
from capture import CaptureThread, FrameSource, RingBuffer
from contracts import EMPTY_SCENE, FACE, Action, BufferedFrame, CensoredFrame, Policy, PolicySnapshot, Scene, TrackedBox
from recorder import Recorder
from vision import policy as policy_rules
from vision import render
from vision.faces import FaceDetector
from vision.identity import OwnerIdentity
from vision.tracking import make_face_tracker

log = logging.getLogger(__name__)

_FRAME_WAIT_S = 0.1


def fail_closed_frame(image: np.ndarray, index: int) -> CensoredFrame:
    """Pixelate the whole frame. The last resort when nothing else can be trusted."""
    height, width = image.shape[:2]
    small_size = (max(1, width // config.PIXEL_BLOCK), max(1, height // config.PIXEL_BLOCK))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return CensoredFrame(image=pixelated, index=index, fail_closed=True)


class Pipeline(threading.Thread):
    """Owns the capture thread and the delay buffer; publishes the latest censored frame."""

    def __init__(self, source: FrameSource, policy: Policy, recorder: Recorder | None = None) -> None:
        super().__init__(name="vision", daemon=True)
        self._policy = policy
        self._recorder = recorder
        self._ring = RingBuffer()
        self._capture = CaptureThread(source, self._ring)
        self._stop_requested = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest: CensoredFrame | None = None
        self._scene: Scene = EMPTY_SCENE

        self._faces = FaceDetector()
        self._face_tracker = make_face_tracker()
        self._identity = OwnerIdentity()
        self._hold_over = render.HoldOver()
        self._watchdog = render.Watchdog(watch_objects=False)  # WP5 turns the object watchdog on
        self._seen_track_ids: set[str] = set()
        self._owner_id: str | None = None
        self._owner_lost_at: dict[str, int] = {}  # track id -> frame index where it stopped being the owner
        self._enroll_requested = threading.Event()

    def latest(self) -> CensoredFrame | None:
        with self._latest_lock:
            return self._latest

    def scene(self) -> Scene:
        """What the agent may see and act on: the newest frame's confident tracks."""
        with self._latest_lock:
            return self._scene

    def enroll_owner(self) -> None:
        """Make the largest confident face in the next frame the owner. Safe from any thread."""
        self._enroll_requested.set()

    @property
    def owner_enrolled(self) -> bool:
        return self._identity.enrolled

    def stop(self) -> None:
        self._stop_requested.set()
        self._capture.stop()

    def run(self) -> None:
        self._capture.start()
        last_index = -1
        while not self._stop_requested.is_set():
            newest = self._capture.wait_for_frame(last_index, _FRAME_WAIT_S)
            if newest is None:
                if not self._capture.is_alive():
                    break
                continue
            last_index = newest.index
            self._detect_or_leave_unchecked(newest)

            oldest = self._ring.pop_oldest()
            if oldest is not None:
                self._publish(self._censor_or_fail_closed(oldest))
        self._capture.stop()

    def _publish(self, censored: CensoredFrame) -> None:
        with self._latest_lock:
            self._latest = censored
        if self._recorder is not None:
            self._recorder.write(censored)

    def _detect_or_leave_unchecked(self, frame: BufferedFrame) -> None:
        """A frame whose detection fails keeps faces_done False, so it leaves the buffer pixelated."""
        try:
            self._detect(frame)
        except Exception:
            log.exception("detection failed on frame %d; it will be pixelated whole", frame.index)

    def _censor_or_fail_closed(self, frame: BufferedFrame) -> CensoredFrame:
        if not frame.faces_done:
            return fail_closed_frame(frame.image, frame.index)
        try:
            return self._censor(frame)
        except Exception:
            log.exception("censoring failed on frame %d; failing closed", frame.index)
            return fail_closed_frame(frame.image, frame.index)

    def _detect(self, frame: BufferedFrame) -> None:
        """Faces on the newest frame: track, identify, hold over, backfill, heartbeat."""
        now = time.monotonic()
        detections = self._faces.detect(frame.image)
        tracked = _without_weak_duplicates(self._face_tracker.update(detections, frame.index, now))
        if self._enroll_requested.is_set():
            self._enroll_largest_face(frame.image, tracked)
        self._identity.update(frame.image, tracked, self._policy, now)
        self._follow_owner(tracked, frame.index)

        frame.boxes.extend(self._hold_over.apply(tracked, now, frame_index=frame.index))
        new_boxes = [box for box in tracked if box.id not in self._seen_track_ids]
        self._seen_track_ids.update(box.id for box in new_boxes)
        render.backfill(self._ring.recent(config.BACKFILL_FRAMES + 1)[:-1], new_boxes)
        frame.faces_done = True
        self._watchdog.beat_faces(now)

        confident = [box for box in tracked if box.conf >= config.SCENE_MIN_CONF]
        scene = policy_rules.build_scene(confident, self._policy.snapshot(), config.FRAME_WIDTH, config.FRAME_HEIGHT)
        with self._latest_lock:
            self._scene = scene

    def _follow_owner(self, tracked: list[TrackedBox], frame_index: int) -> None:
        """Remember when a track that is still in frame stops being the owner (a failed ID re-check)."""
        owner_id = next((box.id for box in tracked if box.is_owner), None)
        previous = self._owner_id
        if previous is not None and previous != owner_id and any(box.id == previous for box in tracked):
            self._owner_lost_at[previous] = frame_index
        if owner_id != previous:
            log.info("owner track: %s -> %s", previous, owner_id)
        self._owner_id = owner_id

    def _enroll_largest_face(self, image: np.ndarray, tracked: list[TrackedBox]) -> None:
        candidates = [
            box for box in tracked
            if box.label == FACE and box.face_row is not None and box.conf >= config.SCENE_MIN_CONF
        ]
        if not candidates:
            return  # keep the request until a confident face shows up
        largest = max(candidates, key=lambda box: (box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1]))
        self._identity.enroll(image, largest.face_row)
        self._enroll_requested.clear()
        log.info("owner enrolled from %s", largest.id)

    def _censor(self, frame: BufferedFrame) -> CensoredFrame:
        return render.censor(
            frame, self._policy.snapshot(), self._decide, self._watchdog.tripped(time.monotonic())
        )

    def _decide(self, box: TrackedBox, policy: PolicySnapshot) -> Action:
        """policy.decide, except that an owner mark is void once its track failed the ID re-check.

        The frame leaving the buffer was marked a second ago. If that track has since
        lost ownership, the frames marked before the loss must not show the face.
        """
        if box.is_owner and box.frame_index <= self._owner_lost_at.get(box.id, -1):
            box = dataclasses.replace(box, is_owner=False)
        return policy_rules.decide(box, policy)


def _without_weak_duplicates(boxes: list[TrackedBox]) -> list[TrackedBox]:
    """Drop a weak box whose centre lies inside a confident box: it is the same face, seen twice.

    Without this, a stray low-score hit on the owner's own face would blur the owner.
    """
    confident = [box for box in boxes if box.conf >= config.SCENE_MIN_CONF]

    def inside_a_confident_box(box: TrackedBox) -> bool:
        centre_x, centre_y = (box.xyxy[0] + box.xyxy[2]) / 2, (box.xyxy[1] + box.xyxy[3]) / 2
        return any(c.xyxy[0] <= centre_x <= c.xyxy[2] and c.xyxy[1] <= centre_y <= c.xyxy[3] for c in confident)

    return [box for box in boxes if box.conf >= config.SCENE_MIN_CONF or not inside_a_confident_box(box)]
