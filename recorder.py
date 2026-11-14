"""MJPG recorder (WP1). It accepts CensoredFrame only, so a raw frame cannot be written."""

from __future__ import annotations

from pathlib import Path

import config
from contracts import CensoredFrame


class Recorder:
    """Writes censored frames to config.RECORDINGS_DIR on its own thread while recording."""

    def __init__(self, directory: Path = config.RECORDINGS_DIR) -> None:
        raise NotImplementedError("WP1")

    @property
    def is_recording(self) -> bool:
        raise NotImplementedError("WP1")

    @property
    def elapsed_s(self) -> float:
        raise NotImplementedError("WP1")

    def start(self) -> Path:
        """Open a new .avi (MJPG, config size and FPS) and return its path."""
        raise NotImplementedError("WP1")

    def write(self, frame: CensoredFrame) -> None:
        """Queue one censored frame; ignored when not recording. Raises TypeError for anything else."""
        raise NotImplementedError("WP1")

    def stop(self) -> Path | None:
        """Flush, close and return the file path, or None if nothing was recording."""
        raise NotImplementedError("WP1")
