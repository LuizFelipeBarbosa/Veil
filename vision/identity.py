"""SFace owner match and ID-swap re-checks (WP6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from contracts import Policy, TrackedBox


class OwnerIdentity:
    """Remembers the owner's embedding and the embedding of every unblurred person."""

    def __init__(self, model_path: Path = config.SFACE_MODEL) -> None:
        raise NotImplementedError("WP6")

    @property
    def enrolled(self) -> bool:
        raise NotImplementedError("WP6")

    def enroll(self, image: np.ndarray, face_row: np.ndarray) -> None:
        """Store the owner's embedding (memory only) from one aligned face."""
        raise NotImplementedError("WP6")

    def update(self, image: np.ndarray, boxes: list[TrackedBox], policy: Policy, now: float) -> None:
        """Set is_owner on at most one face box, and re-check about once a second.

        A "show" override whose face no longer matches its saved embedding is
        removed from the policy, so a swapped ID blurs again (Never-leaks rule 6).
        """
        raise NotImplementedError("WP6")
