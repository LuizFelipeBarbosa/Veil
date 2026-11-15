"""Shared shapes and tool schemas. Only the orchestrator edits this file.

Spec: docs/brief.md, section "Interfaces".
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

Action = Literal["blur", "show", "replace"]
Side = Literal["owner_left", "owner_right", "left", "center", "right"]
Distance = Literal["near", "far"]

FACE = "face"
CENSORED_OBJECT_CLASSES = ("license plate", "street sign", "house number", "storefront sign")
# Ads are not private: they start shown, and are the only classes that can be replaced by an image.
AD_CLASSES = ("billboard", "poster", "advertisement")
OBJECT_CLASSES = (*CENSORED_OBJECT_CLASSES, *AD_CLASSES)
KNOWN_CLASSES = (FACE, *OBJECT_CLASSES)
# What a class rule or a track override can be set to directly. "replace" is set
# only through Policy.set_replacement, and only for AD_CLASSES.
ACTIONS: tuple[Action, ...] = ("blur", "show")
REPLACE: Action = "replace"

FACE_ID_PREFIX = "face"  # track IDs look like face#7
OBJECT_ID_PREFIX = "obj"  # and obj#3


# --- vision shapes ---------------------------------------------------------


@dataclass
class Detection:
    """One raw detector hit, before tracking."""

    label: str
    xyxy: tuple[int, int, int, int]
    conf: float
    # The full YuNet output row (box, 5 landmarks, score). SFace needs it to align the crop.
    face_row: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class TrackedBox:
    """One tracked object on one frame (vision -> policy -> render)."""

    id: str  # "face#7", "obj#3"
    label: str  # "face" or one of OBJECT_CLASSES
    xyxy: tuple[int, int, int, int]  # pixels, unpadded
    conf: float
    frame_index: int  # the buffered frame these boxes belong to
    last_seen: float  # time.monotonic() seconds; drives the hold-over
    is_owner: bool = False  # set by SFace, re-checked every second
    # Not part of the Brief's shape: in-memory only, never serialized. See Detection.face_row.
    face_row: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class BufferedFrame:
    """A RAW frame waiting in the delay buffer, with its own box list.

    `image` is raw. It may be read by vision code only and must never reach the
    preview, the recorder, the disk or the network. Only CensoredFrame goes out.
    """

    index: int
    timestamp: float  # time.monotonic() at capture
    image: np.ndarray  # BGR, config.FRAME_HEIGHT x config.FRAME_WIDTH
    boxes: list[TrackedBox] = field(default_factory=list)
    # False until face detection has finished for this frame. A frame that leaves
    # the buffer with faces_done False is pixelated whole.
    faces_done: bool = False


@dataclass(frozen=True)
class Highlight:
    """One track the operator can point at on the preview. Never drawn into a recording."""

    id: str
    label: str
    xyxy: tuple[int, int, int, int]  # frame pixels, unpadded
    state: Action  # what the frame shows for it right now ("replace": an ad image is drawn over it)
    is_owner: bool = False


@dataclass(frozen=True)
class CensoredFrame:
    """The only frame type the preview and the recorder accept.

    Built only by vision/render.py and by the fail-closed path in pipeline.py.
    """

    image: np.ndarray  # BGR, already pixelated
    index: int
    fail_closed: bool  # True when the whole frame was pixelated (watchdog, panic, error)
    # Operator highlights for the preview only (Never-leaks rule 9); the recorder writes `image` alone.
    highlights: tuple[Highlight, ...] = ()


# --- policy ----------------------------------------------------------------

DEFAULT_CLASS_RULES: Mapping[str, Action] = MappingProxyType(
    {label: ("show" if label in AD_CLASSES else "blur") for label in KNOWN_CLASSES}
)


@dataclass(frozen=True)
class PolicySnapshot:
    """An immutable copy of the Policy, safe to read without the lock."""

    class_rules: Mapping[str, Action]
    track_overrides: Mapping[str, Action]
    panic: bool
    # Goes up each time the face class is set to "blur". Whoever remembers
    # unblurred people (vision/identity.py) must forget them all when it changes.
    face_reset_count: int = 0
    # Ad class -> image file name in config.ADS_DIR, for every class whose rule is "replace".
    replacements: Mapping[str, str] = field(default_factory=dict)


class Policy:
    """The only state the agent and the buttons can change. Thread-safe.

    Everything private starts blurred; ads (AD_CLASSES) start shown. The limits live here so no caller can skip them:
    unknown labels and actions are refused, and blurring the face class clears
    every face override and makes every remembered person forgotten.

    Changed by the lead on 2026-11-14 (was refused in the Brief): the face class
    may be set to "show" on an explicit request to keep everyone unblurred.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._class_rules: dict[str, Action] = dict(DEFAULT_CLASS_RULES)
        self._replacements: dict[str, str] = {}
        self._track_overrides: dict[str, Action] = {}
        self._panic = False
        self._face_reset_count = 0

    def snapshot(self) -> PolicySnapshot:
        with self._lock:
            return PolicySnapshot(
                class_rules=MappingProxyType(dict(self._class_rules)),
                track_overrides=MappingProxyType(dict(self._track_overrides)),
                panic=self._panic,
                face_reset_count=self._face_reset_count,
                replacements=MappingProxyType(dict(self._replacements)),
            )

    def to_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "class_rules": dict(snapshot.class_rules),
            "track_overrides": dict(snapshot.track_overrides),
            "panic": snapshot.panic,
            "replacements": dict(snapshot.replacements),
        }

    def set_class_rule(self, label: str, action: Action) -> None:
        """Raises ValueError for an unknown label or action.

        ("face", "show") keeps everyone unblurred, including people who arrive later,
        until ("face", "blur"), which also clears every per-person choice.
        Setting an ad class to "blur" or "show" turns its replacement off.
        """
        _require_action(action)
        if label not in KNOWN_CLASSES:
            raise ValueError(f"unknown label {label!r}; known: {', '.join(KNOWN_CLASSES)}")
        with self._lock:
            self._class_rules[label] = action
            self._replacements.pop(label, None)
            if label == FACE and action == "blur":
                self._clear_overrides(FACE_ID_PREFIX)
                self._face_reset_count += 1

    def set_replacement(self, label: str, asset: str) -> None:
        """Draw the image `asset` over every box of an ad class. Raises ValueError for any other class.

        The caller checks that `asset` is a file in config.ADS_DIR.
        """
        if label not in AD_CLASSES:
            raise ValueError(f"only ads can be replaced ({', '.join(AD_CLASSES)}), not {label!r}")
        if not asset:
            raise ValueError("no ad image named")
        with self._lock:
            self._class_rules[label] = REPLACE
            self._replacements[label] = asset

    def set_track_override(self, track_id: str, action: Action) -> None:
        """The caller checks that track_id is in the live scene."""
        _require_action(action)
        with self._lock:
            self._track_overrides[track_id] = action

    def clear_track_override(self, track_id: str) -> None:
        with self._lock:
            self._track_overrides.pop(track_id, None)

    def set_panic(self, on: bool) -> None:
        with self._lock:
            self._panic = bool(on)

    def show_all(self) -> None:
        """Unblur everything and keep it unblurred: every class, faces included. Turns panic off.

        Per-track "blur" choices stay, so select things can still be hidden. Replaced ads stay replaced.
        """
        with self._lock:
            for label in KNOWN_CLASSES:
                if self._class_rules[label] != REPLACE:
                    self._class_rules[label] = "show"
            self._panic = False

    def _clear_overrides(self, id_prefix: str) -> None:
        for track_id in [t for t in self._track_overrides if t.startswith(f"{id_prefix}#")]:
            del self._track_overrides[track_id]


def _require_action(action: str) -> None:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; use blur or show")


# --- scene -----------------------------------------------------------------


@dataclass(frozen=True)
class SceneTrack:
    id: str
    label: str
    # Relative to the owner when one is in frame (owner_left, owner_right),
    # otherwise the operator's view (left, center, right). See config.CAMERA_FACES_OPERATOR.
    side: Side
    distance: Distance
    state: Action  # what decide() returns for this track right now


@dataclass(frozen=True)
class Scene:
    """What get_scene returns (vision -> agent)."""

    owner: str | None
    tracks: tuple[SceneTrack, ...]
    class_rules: Mapping[str, Action]
    panic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "tracks": [asdict(track) for track in self.tracks],
            "class_rules": dict(self.class_rules),
            "panic": self.panic,
        }

    def track_ids(self) -> set[str]:
        return {track.id for track in self.tracks}


EMPTY_SCENE = Scene(owner=None, tracks=(), class_rules=dict(DEFAULT_CLASS_RULES), panic=False)


# --- agent tools -----------------------------------------------------------
# Deepgram Voice Agent function definitions. No "endpoint" key, so they are
# client-side: Deepgram sends a FunctionCallRequest and the device answers.

_NO_ARGUMENTS = {"type": "object", "properties": {}, "required": []}
_ACTION_PARAM = {"type": "string", "enum": list(ACTIONS), "description": "blur hides it, show reveals it"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_scene",
        "description": "List the owner, every tracked face and object with its side, distance and "
        "current state, the class rules and the panic flag. Call it before any command about "
        "specific people or objects. Changes nothing.",
        "parameters": _NO_ARGUMENTS,
    },
    {
        "name": "set_class_rule",
        "description": "Blur or show a whole class. For an ad class this also turns its replacement "
        "off. Showing the face class unblurs everyone and "
        "keeps people who arrive later unblurred too; blurring it blurs everyone again and forgets "
        "every per-person choice.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(KNOWN_CLASSES)},
                "action": _ACTION_PARAM,
            },
            "required": ["label", "action"],
        },
    },
    {
        "name": "set_track",
        "description": "Blur or show one tracked person or object. track_id must come from the "
        "latest get_scene result.",
        "parameters": {
            "type": "object",
            "properties": {
                "track_id": {"type": "string", "description": "An id from get_scene, such as face#7"},
                "action": _ACTION_PARAM,
            },
            "required": ["track_id", "action"],
        },
    },
    {
        "name": "set_replacement",
        "description": "Replace every billboard, poster or advertisement in view with one of the ad "
        "images on the device. Turn it off with set_class_rule(label, show).",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": list(AD_CLASSES)},
                "asset": {"type": "string", "description": "An ad image file name from the list in the prompt"},
            },
            "required": ["label", "asset"],
        },
    },
    {
        "name": "blur_all",
        "description": "Panic on: pixelate the whole frame.",
        "parameters": _NO_ARGUMENTS,
    },
    {
        "name": "clear_panic",
        "description": "Panic off: go back to the normal blur rules.",
        "parameters": _NO_ARGUMENTS,
    },
    {
        "name": "show_all",
        "description": "Unblur everything and keep it unblurred: every object class and every "
        "person, including people who arrive later. Also turns panic off. Only for an explicit "
        "request to unblur or show everything.",
        "parameters": _NO_ARGUMENTS,
    },
    {
        "name": "start_recording",
        "description": "Start recording the censored video to a file. Same as the Record button. "
        "Safe to call when already recording. Changes nothing about what is blurred.",
        "parameters": _NO_ARGUMENTS,
    },
    {
        "name": "stop_recording",
        "description": "Stop recording and save the file. Same as the Record button. Safe to call "
        "when not recording.",
        "parameters": _NO_ARGUMENTS,
    },
]

TOOL_NAMES = tuple(schema["name"] for schema in TOOL_SCHEMAS)
