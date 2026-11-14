"""decide() and the scene builder (WP6). The Policy shape itself lives in contracts.py."""

from __future__ import annotations

from typing import Sequence

from contracts import Action, PolicySnapshot, Scene, TrackedBox


def decide(box: TrackedBox, policy: PolicySnapshot) -> Action:
    """Order: panic, track override, owner, class rule. Anything unknown blurs."""
    raise NotImplementedError("WP6")


def build_scene(boxes: Sequence[TrackedBox], policy: PolicySnapshot, frame_width: int, frame_height: int) -> Scene:
    """The Scene for get_scene: sides relative to the owner, honoring config.CAMERA_FACES_OPERATOR."""
    raise NotImplementedError("WP6")
