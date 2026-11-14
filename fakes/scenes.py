"""Fixture scenes for the agent and tool tests. Shapes match contracts.Scene."""

from __future__ import annotations

from contracts import KNOWN_CLASSES, Scene, SceneTrack

_ALL_BLUR = {label: "blur" for label in KNOWN_CLASSES}

EMPTY = Scene(owner=None, tracks=(), class_rules=_ALL_BLUR, panic=False)

# "Unblur the person on my left" -> set_track("face#7", "show")
OWNER_AND_FRIEND_LEFT = Scene(
    owner="face#2",
    tracks=(
        SceneTrack(id="face#2", label="face", side="center", distance="near", state="show"),
        SceneTrack(id="face#7", label="face", side="owner_left", distance="near", state="blur"),
        SceneTrack(id="obj#3", label="license plate", side="right", distance="far", state="blur"),
    ),
    class_rules=_ALL_BLUR,
    panic=False,
)

# "Unblur the guy" -> two candidates, the agent must ask
TWO_CANDIDATES = Scene(
    owner="face#2",
    tracks=(
        SceneTrack(id="face#2", label="face", side="center", distance="near", state="show"),
        SceneTrack(id="face#7", label="face", side="owner_left", distance="near", state="blur"),
        SceneTrack(id="face#9", label="face", side="owner_right", distance="far", state="blur"),
    ),
    class_rules=_ALL_BLUR,
    panic=False,
)

# No owner in frame: sides are the operator's view
SIGNS_AND_PLATE = Scene(
    owner=None,
    tracks=(
        SceneTrack(id="obj#1", label="street sign", side="left", distance="far", state="blur"),
        SceneTrack(id="obj#3", label="license plate", side="right", distance="near", state="blur"),
        SceneTrack(id="obj#4", label="house number", side="center", distance="far", state="blur"),
    ),
    class_rules=_ALL_BLUR,
    panic=False,
)

PANIC = Scene(owner="face#2", tracks=OWNER_AND_FRIEND_LEFT.tracks, class_rules=_ALL_BLUR, panic=True)

ALL = {
    "empty": EMPTY,
    "owner_and_friend_left": OWNER_AND_FRIEND_LEFT,
    "two_candidates": TWO_CANDIDATES,
    "signs_and_plate": SIGNS_AND_PLATE,
    "panic": PANIC,
}
