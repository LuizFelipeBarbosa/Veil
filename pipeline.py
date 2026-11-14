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
from contracts import (
    CENSORED_OBJECT_CLASSES, EMPTY_SCENE, FACE, REPLACE, Action, BufferedFrame, CensoredFrame, Highlight, Policy,
    PolicySnapshot, Scene, TrackedBox,
)
from recorder import Recorder
from vision import policy as policy_rules
from vision import render
from vision.faces import FaceDetector
from vision.identity import OwnerIdentity
from vision.objects import ObjectDetector, ObjectsThread
from vision.replace import AdReplacer
from vision.tracking import make_face_tracker, make_object_tracker

log = logging.getLogger(__name__)

_THROUGHPUT_LOG_S = 10.0


class _Throughput:
    """Logs how many frames were published, failed closed or were skipped unseen, every 10 s."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._frames = 0
        self._fail_closed = 0
        self._skipped = 0

    def count_skipped(self) -> None:
        self._skipped += 1

    def count(self, fail_closed: bool, now: float) -> None:
        if self._since is None:
            self._since = now
        self._frames += 1
        self._fail_closed += fail_closed
        elapsed = now - self._since
        if elapsed >= _THROUGHPUT_LOG_S:
            log.info(
                "published %.1f FPS, %d of %d frames failed closed, %d skipped unseen",
                self._frames / elapsed, self._fail_closed, self._frames, self._skipped,
            )
            self._since, self._frames, self._fail_closed, self._skipped = now, 0, 0, 0

_FRAME_WAIT_S = 0.1
_SIGN_INSIDE_AD_SHARE = 0.8  # of the sign box's area; see Pipeline._with_ads


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
        self._watchdog = render.Watchdog()
        self._throughput = _Throughput()

        # Built before any thread starts: loading YOLOE changes the working directory for a moment.
        self._objects = ObjectsThread(ObjectDetector(), self._ring.newest)
        self._object_tracker = make_object_tracker()
        # YOLOE results arrive OBJECT_MIN_PERIOD_S apart at best, so an object must stay blurred that
        # much longer than a face, or it would unblur between two detections.
        self._object_hold_over = render.HoldOver(config.HOLD_OVER_S + config.OBJECT_MIN_PERIOD_S)
        self._ads = AdReplacer()
        self._last_object_result_index = -1
        self._seen_track_ids: set[str] = set()
        self._confirmed_face_ids: set[str] = set()
        self._unconfirmed_boxes: dict[str, list[TrackedBox]] = {}  # track id -> weak boxes waiting in the buffer
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

    def toggle_at(self, x: int, y: int) -> str | None:
        """The operator pointed at frame pixel (x, y): flip blur/show for the track shown there.

        Works on the frame currently on screen. Returns the track id, or None when
        nothing pointable is there (including while the whole frame is pixelated).
        """
        shown = self.latest()
        if shown is None:
            return None
        under_pointer = [h for h in shown.highlights if h.xyxy[0] <= x <= h.xyxy[2] and h.xyxy[1] <= y <= h.xyxy[3]]
        if not under_pointer:
            return None
        target = min(under_pointer, key=lambda h: (h.xyxy[2] - h.xyxy[0]) * (h.xyxy[3] - h.xyxy[1]))
        action: Action = "blur" if target.state == "show" else "show"
        self._policy.set_track_override(target.id, action)
        log.info("pointed at %s (%s): %s", target.id, target.label, action)
        return target.id

    @property
    def owner_enrolled(self) -> bool:
        return self._identity.enrolled

    def stop(self) -> None:
        self._stop_requested.set()
        self._capture.stop()
        self._objects.stop()

    def run(self) -> None:
        self._capture.start()
        self._objects.start()
        last_index = -1
        attempted: set[int] = set()  # buffered frames detection ran on, whether or not it succeeded
        while not self._stop_requested.is_set():
            if self._capture.wait_for_frame(last_index, _FRAME_WAIT_S) is None:
                if not self._capture.is_alive():
                    break
                continue
            # When detection is slower than the camera, frames queue up. Detect the newest one and
            # skip the rest: a skipped frame is never shown, so the preview slows down instead of
            # showing frames that left the buffer undetected (those would fail closed).
            newest = self._ring.newest()
            last_index = newest.index
            attempted.add(newest.index)
            self._detect_or_leave_unchecked(newest)

            while (oldest := self._ring.pop_oldest()) is not None:
                if oldest.index in attempted:
                    self._publish(self._censor_or_fail_closed(oldest))
                else:
                    self._throughput.count_skipped()
                attempted = {index for index in attempted if index > oldest.index}
        self._capture.stop()
        self._objects.stop()

    def _publish(self, censored: CensoredFrame) -> None:
        with self._latest_lock:
            self._latest = censored
        self._throughput.count(censored.fail_closed, time.monotonic())
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
        tracked = self._confirmed_faces(tracked, frame.index)
        if self._enroll_requested.is_set():
            self._enroll_largest_face(frame.image, tracked)
        self._identity.update(frame.image, tracked, self._policy, now)
        self._follow_owner(tracked, frame.index)

        objects = self._new_object_boxes()
        frame.boxes.extend(self._hold_over.apply(tracked, now, frame_index=frame.index))
        frame.boxes.extend(self._object_hold_over.apply(objects, now, frame_index=frame.index))

        new_boxes = [box for box in [*tracked, *objects] if box.id not in self._seen_track_ids]
        self._seen_track_ids.update(box.id for box in new_boxes)
        # Every frame still waiting, not only the last BACKFILL_FRAMES: when detection is slow more
        # frames pile up, and none of them may go out without a box that has just appeared (leak review).
        waiting = [earlier for earlier in self._ring.recent(len(self._ring)) if earlier.index < frame.index]
        render.backfill(waiting, new_boxes)
        frame.faces_done = True
        self._watchdog.beat_faces(now)

        confident_faces = [box for box in tracked if box.conf >= config.SCENE_MIN_CONF]
        visible = [*confident_faces, *(box for box in frame.boxes if box.label != FACE)]
        scene = policy_rules.build_scene(visible, self._policy.snapshot(), config.FRAME_WIDTH, config.FRAME_HEIGHT)
        with self._latest_lock:
            self._scene = scene

    def _confirmed_faces(self, tracked: list[TrackedBox], frame_index: int) -> list[TrackedBox]:
        """Only the boxes of tracks that have scored config.FACE_CONFIRM_SCORE at least once.

        A weak box of a track not confirmed yet waits. If the track is confirmed while
        its frames are still in the delay buffer, the waiting boxes are put on those
        frames, so the face is blurred from its first weak sighting. A track that never
        gets there (a pattern on a wall) is never blurred.
        """
        buffered = {frame.index: frame for frame in self._ring.recent(config.DELAY_BUFFER_FRAMES)}
        confirmed: list[TrackedBox] = []
        for box in tracked:
            if box.conf >= config.FACE_CONFIRM_SCORE and box.id not in self._confirmed_face_ids:
                self._confirmed_face_ids.add(box.id)
                for waiting in self._unconfirmed_boxes.pop(box.id, []):
                    if waiting.frame_index in buffered:
                        buffered[waiting.frame_index].boxes.append(waiting)
            if box.id in self._confirmed_face_ids:
                confirmed.append(box)
            else:
                self._unconfirmed_boxes.setdefault(box.id, []).append(box)

        oldest_buffered = frame_index - config.DELAY_BUFFER_FRAMES
        self._unconfirmed_boxes = {
            track_id: boxes
            for track_id, boxes in self._unconfirmed_boxes.items()
            if boxes[-1].frame_index >= oldest_buffered
        }
        return confirmed

    def _new_object_boxes(self) -> list[TrackedBox]:
        """Tracked boxes from a YOLOE result not handled yet, else nothing (hold-over bridges the gap).

        YOLOE runs on its own thread on whichever frame was newest, so its boxes land
        a few frames late; backfill and padding cover that.
        """
        result = self._objects.latest()
        if result is None:
            return []
        frame_index, finished_at, detections = result
        if frame_index == self._last_object_result_index:
            return []
        self._last_object_result_index = frame_index
        self._watchdog.beat_objects(finished_at)
        return self._object_tracker.update(detections, frame_index, finished_at)

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
        if not self._identity.enroll(image, largest.face_row):
            return  # face too small to learn: keep the request until the owner is closer
        self._enroll_requested.clear()
        log.info("owner enrolled from %s", largest.id)

    def _censor(self, frame: BufferedFrame) -> CensoredFrame:
        """Ads are drawn first, so a face in front of a replaced ad is pixelated on top of it."""
        policy = self._policy.snapshot()
        now = time.monotonic()
        with_ads = self._with_ads(frame, policy, now)
        censored = render.censor(with_ads, policy, self._decide_censoring, self._watchdog.tripped(now))
        if censored.fail_closed:
            return censored
        return dataclasses.replace(censored, highlights=self._highlights(frame, policy))

    def _with_ads(self, frame: BufferedFrame, policy: PolicySnapshot, now: float) -> BufferedFrame:
        """The frame with an ad image over every box decided "replace". Still raw: it goes on to censor().

        The model often boxes one ad panel twice, as an ad and as a sign. The lead chose (15:38) that
        the ad wins: a sign box lying almost wholly inside a replaced ad is dropped, so the ad is not
        pixelated. Faces are never dropped, and neither is a sign the operator chose to blur.
        """
        replaced = [box for box in frame.boxes if self._decide(box, policy) == REPLACE]
        if not replaced or not frame.faces_done:
            return frame
        kept = [
            box for box in frame.boxes
            if not (
                box.label in CENSORED_OBJECT_CLASSES
                and box.id not in policy.track_overrides
                and any(_share_inside(box.xyxy, ad.xyxy) >= _SIGN_INSIDE_AD_SHARE for ad in replaced)
            )
        ]
        image = self._ads.draw(frame.image, replaced, policy.replacements, now)
        return dataclasses.replace(frame, image=image, boxes=kept)

    def _decide_censoring(self, box: TrackedBox, policy: PolicySnapshot) -> Action:
        """What censor() needs to know: a replaced ad is already drawn over, so it is not pixelated."""
        action = self._decide(box, policy)
        return "show" if action == REPLACE else action

    def _highlights(self, frame: BufferedFrame, policy: PolicySnapshot) -> tuple[Highlight, ...]:
        """What the operator can point at: objects and confident faces, one entry per track."""
        by_id = {
            box.id: Highlight(box.id, box.label, box.xyxy, self._decide(box, policy), box.is_owner)
            for box in frame.boxes
            if box.label != FACE or box.conf >= config.SCENE_MIN_CONF
        }
        return tuple(by_id.values())

    def _decide(self, box: TrackedBox, policy: PolicySnapshot) -> Action:
        """policy.decide, except that an owner mark is void once its track failed the ID re-check.

        The frame leaving the buffer was marked a second ago. If that track has since
        lost ownership, the frames marked before the loss must not show the face.
        """
        if box.is_owner and box.frame_index <= self._owner_lost_at.get(box.id, -1):
            box = dataclasses.replace(box, is_owner=False)
        return policy_rules.decide(box, policy)


def _share_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """The share of `inner`'s area that lies inside `outer`."""
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0, width) * max(0, height) / area if area > 0 else 0.0


def _without_weak_duplicates(boxes: list[TrackedBox]) -> list[TrackedBox]:
    """Drop a weak box whose centre lies inside a confident box: it is the same face, seen twice.

    Without this, a stray low-score hit on the owner's own face would blur the owner.
    """
    confident = [box for box in boxes if box.conf >= config.SCENE_MIN_CONF]

    def inside_a_confident_box(box: TrackedBox) -> bool:
        centre_x, centre_y = (box.xyxy[0] + box.xyxy[2]) / 2, (box.xyxy[1] + box.xyxy[3]) / 2
        return any(c.xyxy[0] <= centre_x <= c.xyxy[2] and c.xyxy[1] <= centre_y <= c.xyxy[3] for c in confident)

    return [box for box in boxes if box.conf >= config.SCENE_MIN_CONF or not inside_a_confident_box(box)]
