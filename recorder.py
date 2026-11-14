"""MJPG recorder (WP1). It accepts CensoredFrame only, so a raw frame cannot be written."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from contracts import CensoredFrame

log = logging.getLogger(__name__)

_QUEUE_FRAMES = 2 * config.FPS  # a slow disk drops censored frames instead of stalling the vision loop
_STOP = None  # queue sentinel


class Recorder:
    """Writes censored frames to config.RECORDINGS_DIR on its own thread while recording."""

    def __init__(self, directory: Path = config.RECORDINGS_DIR) -> None:
        self._directory = Path(directory)
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._queue: queue.Queue[CensoredFrame | None] | None = None
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._thread is not None

    @property
    def elapsed_s(self) -> float:
        with self._lock:
            return time.monotonic() - self._started_at if self._thread is not None else 0.0

    def start(self) -> Path:
        """Open a new .avi (MJPG, config size and FPS) and return its path."""
        with self._lock:
            if self._thread is not None:
                return self._path
            self._directory.mkdir(parents=True, exist_ok=True)
            path = _new_file_path(self._directory)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*config.RECORDING_FOURCC),
                config.FPS,
                (config.FRAME_WIDTH, config.FRAME_HEIGHT),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {path}")
            frames: queue.Queue[CensoredFrame | None] = queue.Queue(maxsize=_QUEUE_FRAMES)
            thread = threading.Thread(target=_write_frames, args=(writer, frames), name="recorder", daemon=True)
            thread.start()
            self._path, self._queue, self._thread = path, frames, thread
            self._started_at = time.monotonic()
            return path

    def write(self, frame: CensoredFrame) -> None:
        """Queue one censored frame; ignored when not recording. Raises TypeError for anything else."""
        if not isinstance(frame, CensoredFrame):
            raise TypeError(f"the recorder only accepts CensoredFrame, got {type(frame).__name__}")
        with self._lock:
            if self._queue is None:
                return
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                log.warning("recorder queue full; dropped censored frame %d", frame.index)

    def stop(self) -> Path | None:
        """Flush, close and return the file path, or None if nothing was recording."""
        with self._lock:
            path, frames, thread = self._path, self._queue, self._thread
            self._path, self._queue, self._thread = None, None, None
        if thread is None:
            return None
        frames.put(_STOP)  # blocks only until the writer thread drains a slot
        thread.join()
        return path


def _new_file_path(directory: Path) -> Path:
    stem = datetime.now().strftime("privacy-cam-%Y%m%d-%H%M%S")
    path = directory / f"{stem}{config.RECORDING_EXTENSION}"
    suffix = 1
    while path.exists():
        path = directory / f"{stem}-{suffix}{config.RECORDING_EXTENSION}"
        suffix += 1
    return path


def _write_frames(writer: cv2.VideoWriter, frames: queue.Queue[CensoredFrame | None]) -> None:
    """The writer thread: write queued frames until the stop sentinel, then close the file.

    After a write error it keeps draining the queue, so stop() never blocks on a full queue.
    """
    size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    failed = False
    while (frame := frames.get()) is not _STOP:
        if failed:
            continue
        try:
            image = frame.image
            if (image.shape[1], image.shape[0]) != size:
                image = cv2.resize(image, size)  # VideoWriter silently drops wrong-size frames
            writer.write(image)
        except Exception:
            log.exception("recorder failed on frame %d; dropping the rest of this recording", frame.index)
            failed = True
    writer.release()
