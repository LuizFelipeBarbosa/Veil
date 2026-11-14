"""decide() and the scene builder (WP6). The Policy shape itself lives in contracts.py."""

from __future__ import annotations

from typing import Sequence

import config
from contracts import ACTIONS, AD_CLASSES, REPLACE, Action, Distance, PolicySnapshot, Scene, SceneTrack, Side, TrackedBox


def decide(box: TrackedBox, policy: PolicySnapshot) -> Action:
    """Order: panic, track override, owner, class rule. Anything unknown blurs.

    "replace" holds only for an ad class with an image named for it; any other "replace" blurs.
    """
    if policy.panic:
        return "blur"
    override = policy.track_overrides.get(box.id)
    if override is not None:
        return override if override in ACTIONS else "blur"
    if box.is_owner:
        return "show"
    rule = policy.class_rules.get(box.label)
    if rule == "show":
        return "show"
    if rule == REPLACE and box.label in AD_CLASSES and policy.replacements.get(box.label):
        return REPLACE
    return "blur"


def build_scene(boxes: Sequence[TrackedBox], policy: PolicySnapshot, frame_width: int, frame_height: int) -> Scene:
    """The Scene for get_scene: sides relative to the owner, honoring config.CAMERA_FACES_OPERATOR."""
    owner_box = next((box for box in boxes if box.is_owner), None)
    tracks: dict[str, SceneTrack] = {}
    for box in boxes:
        if box.id in tracks:
            continue
        if owner_box is None or box.id == owner_box.id:
            side = _operator_side(box, frame_width)
        else:
            side = _side_of_owner(box, owner_box)
        tracks[box.id] = SceneTrack(
            id=box.id, label=box.label, side=side, distance=_distance(box, frame_height), state=decide(box, policy)
        )
    return Scene(
        owner=owner_box.id if owner_box is not None else None,
        tracks=tuple(tracks.values()),
        class_rules=dict(policy.class_rules),
        panic=policy.panic,
    )


def _center_x(box: TrackedBox) -> float:
    return (box.xyxy[0] + box.xyxy[2]) / 2


def _distance(box: TrackedBox, frame_height: int) -> Distance:
    height = box.xyxy[3] - box.xyxy[1]
    return "near" if height >= config.NEAR_BOX_HEIGHT_FRACTION * frame_height else "far"


def _side_of_owner(box: TrackedBox, owner_box: TrackedBox) -> Side:
    """From the owner's point of view. Facing the camera, the owner's left is image-right."""
    image_right = _center_x(box) > _center_x(owner_box)
    on_owner_left = image_right if config.CAMERA_FACES_OPERATOR else not image_right
    return "owner_left" if on_owner_left else "owner_right"


def _operator_side(box: TrackedBox, frame_width: int) -> Side:
    """From the operator's point of view, with a centre band of config.CENTER_BAND_FRACTION."""
    half_band = config.CENTER_BAND_FRACTION * frame_width / 2
    offset = _center_x(box) - frame_width / 2
    if abs(offset) <= half_band:
        return "center"
    image_right = offset > 0
    on_operator_left = image_right if config.CAMERA_FACES_OPERATOR else not image_right
    return "left" if on_operator_left else "right"
