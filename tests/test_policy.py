"""WP6: decide() order and the scene builder. Synthetic boxes only."""

from __future__ import annotations

from types import MappingProxyType

import pytest

import config
from contracts import KNOWN_CLASSES, Policy, PolicySnapshot, TrackedBox
from fakes import scenes
from vision.policy import build_scene, decide

WIDTH, HEIGHT = 640, 480


def _box(track_id: str, label: str = "face", xyxy=(300, 100, 340, 160), is_owner: bool = False) -> TrackedBox:
    return TrackedBox(
        id=track_id, label=label, xyxy=xyxy, conf=0.9, frame_index=0, last_seen=0.0, is_owner=is_owner
    )


def _snapshot(overrides=None, panic: bool = False, **class_rules) -> PolicySnapshot:
    rules = {label: "blur" for label in KNOWN_CLASSES}
    rules.update({label.replace("_", " "): action for label, action in class_rules.items()})
    return PolicySnapshot(
        class_rules=MappingProxyType(rules), track_overrides=MappingProxyType(dict(overrides or {})), panic=panic
    )


# --- decide ---


def test_default_policy_blurs_faces_and_objects() -> None:
    snapshot = Policy().snapshot()
    assert decide(_box("face#1"), snapshot) == "blur"
    assert decide(_box("obj#1", "license plate"), snapshot) == "blur"


def test_panic_beats_a_show_override_and_the_owner() -> None:
    snapshot = _snapshot({"face#7": "show"}, panic=True, street_sign="show")
    assert decide(_box("face#7"), snapshot) == "blur"
    assert decide(_box("face#2", is_owner=True), snapshot) == "blur"
    assert decide(_box("obj#1", "street sign"), snapshot) == "blur"


def test_show_override_shows_a_face() -> None:
    assert decide(_box("face#7"), _snapshot({"face#7": "show"})) == "show"


def test_blur_override_beats_the_owner() -> None:
    assert decide(_box("face#2", is_owner=True), _snapshot({"face#2": "blur"})) == "blur"


def test_blur_override_beats_a_show_class_rule() -> None:
    assert decide(_box("obj#3", "street sign"), _snapshot({"obj#3": "blur"}, street_sign="show")) == "blur"


def test_owner_is_shown_without_an_override() -> None:
    assert decide(_box("face#2", is_owner=True), _snapshot()) == "show"


def test_class_rule_applies_without_override_or_owner() -> None:
    snapshot = _snapshot(street_sign="show")
    assert decide(_box("obj#1", "street sign"), snapshot) == "show"
    assert decide(_box("obj#2", "license plate"), snapshot) == "blur"


def test_unknown_label_blurs() -> None:
    assert decide(_box("obj#9", "logo"), _snapshot()) == "blur"


def test_unknown_override_action_blurs() -> None:
    assert decide(_box("obj#1", "street sign"), _snapshot({"obj#1": "maybe"}, street_sign="show")) == "blur"


def test_another_tracks_override_does_not_apply() -> None:
    assert decide(_box("face#8"), _snapshot({"face#7": "show"})) == "blur"


# --- build_scene ---


def _owner_and_friend_left_boxes(camera_faces_operator: bool) -> list[TrackedBox]:
    """Owner in the middle, a friend on the owner's left, a plate far away (like OWNER_AND_FRIEND_LEFT)."""
    image_right, image_left = (480, 100, 560, 220), (80, 100, 160, 220)
    friend, plate = (image_right, image_left) if camera_faces_operator else (image_left, image_right)
    return [
        _box("face#2", xyxy=(280, 80, 360, 220), is_owner=True),
        _box("face#7", xyxy=friend),
        _box("obj#3", "license plate", xyxy=(plate[0], 400, plate[0] + 30, 420)),
    ]


@pytest.mark.parametrize("camera_faces_operator", [True, False])
def test_person_on_my_left_is_owner_left(monkeypatch, camera_faces_operator: bool) -> None:
    monkeypatch.setattr(config, "CAMERA_FACES_OPERATOR", camera_faces_operator)
    scene = build_scene(_owner_and_friend_left_boxes(camera_faces_operator), _snapshot(), WIDTH, HEIGHT)
    fixture = {track.id: track for track in scenes.OWNER_AND_FRIEND_LEFT.tracks}
    tracks = {track.id: track for track in scene.tracks}
    assert scene.owner == "face#2"
    assert tracks["face#2"] == fixture["face#2"]
    assert tracks["face#7"] == fixture["face#7"]
    assert [t.id for t in scene.tracks if t.side == "owner_left"] == ["face#7"]


@pytest.mark.parametrize("camera_faces_operator", [True, False])
def test_same_image_flips_sides_with_camera_orientation(monkeypatch, camera_faces_operator: bool) -> None:
    monkeypatch.setattr(config, "CAMERA_FACES_OPERATOR", camera_faces_operator)
    boxes = [_box("face#2", xyxy=(280, 80, 360, 220), is_owner=True), _box("face#7", xyxy=(480, 100, 560, 220))]
    scene = build_scene(boxes, _snapshot(), WIDTH, HEIGHT)
    assert scene.tracks[1].side == ("owner_left" if camera_faces_operator else "owner_right")


def test_scene_to_dict_has_the_brief_shape() -> None:
    scene = build_scene(_owner_and_friend_left_boxes(config.CAMERA_FACES_OPERATOR), _snapshot(), WIDTH, HEIGHT)
    as_dict = scene.to_dict()
    expected = scenes.OWNER_AND_FRIEND_LEFT.to_dict()
    assert set(as_dict) == {"owner", "tracks", "class_rules", "panic"}
    assert as_dict["owner"] == "face#2"
    assert as_dict["class_rules"] == expected["class_rules"]
    assert as_dict["panic"] is False
    for track in as_dict["tracks"]:
        assert set(track) == {"id", "label", "side", "distance", "state"}
    assert as_dict["tracks"][:2] == expected["tracks"][:2]
    assert as_dict["tracks"][2] == {
        "id": "obj#3", "label": "license plate", "side": "owner_right", "distance": "far", "state": "blur"
    }


@pytest.mark.parametrize(
    ("camera_faces_operator", "x_center", "side"),
    [
        (True, 50, "right"),
        (True, 320, "center"),
        (True, 600, "left"),
        (False, 50, "left"),
        (False, 320, "center"),
        (False, 600, "right"),
    ],
)
def test_operator_sides_without_owner(monkeypatch, camera_faces_operator: bool, x_center: int, side: str) -> None:
    monkeypatch.setattr(config, "CAMERA_FACES_OPERATOR", camera_faces_operator)
    scene = build_scene([_box("obj#1", "street sign", xyxy=(x_center - 10, 0, x_center + 10, 20))], _snapshot(), 640, 480)
    assert scene.owner is None
    assert scene.tracks[0].side == side


def test_center_band_follows_config(monkeypatch) -> None:
    box = _box("obj#1", "street sign", xyxy=(370, 0, 390, 20))  # centre 380, 60 px right of the middle
    monkeypatch.setattr(config, "CENTER_BAND_FRACTION", 0.2)  # band is +-64 px
    assert build_scene([box], _snapshot(), WIDTH, HEIGHT).tracks[0].side == "center"
    monkeypatch.setattr(config, "CENTER_BAND_FRACTION", 0.1)  # band is +-32 px
    assert build_scene([box], _snapshot(), WIDTH, HEIGHT).tracks[0].side != "center"


def test_distance_uses_box_height(monkeypatch) -> None:
    monkeypatch.setattr(config, "NEAR_BOX_HEIGHT_FRACTION", 0.25)
    near = _box("face#1", xyxy=(0, 0, 50, 120))  # exactly 0.25 of 480
    far = _box("face#2", xyxy=(0, 0, 50, 119))
    distances = {t.id: t.distance for t in build_scene([near, far], _snapshot(), WIDTH, HEIGHT).tracks}
    assert distances == {"face#1": "near", "face#2": "far"}


def test_scene_has_one_track_per_id_and_states_from_decide() -> None:
    boxes = [_box("face#1"), _box("face#1"), _box("face#5"), _box("obj#2", "street sign")]
    scene = build_scene(boxes, _snapshot({"face#5": "show"}, street_sign="show"), WIDTH, HEIGHT)
    assert [(t.id, t.state) for t in scene.tracks] == [("face#1", "blur"), ("face#5", "show"), ("obj#2", "show")]


def test_scene_carries_panic_and_class_rules() -> None:
    scene = build_scene([_box("face#2", is_owner=True)], _snapshot(panic=True, house_number="show"), WIDTH, HEIGHT)
    assert scene.panic is True
    assert scene.class_rules["house number"] == "show"
    assert scene.tracks[0].state == "blur"


def test_empty_scene() -> None:
    assert build_scene([], Policy().snapshot(), WIDTH, HEIGHT).to_dict() == scenes.EMPTY.to_dict()
