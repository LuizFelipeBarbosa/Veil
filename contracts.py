"""Shared shapes and tool schemas. Only the orchestrator edits this file.

Spec: docs/brief.md, section "Interfaces".
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

Action = Literal["blur", "show"]
Side = Literal["owner_left", "owner_right", "left", "center", "right"]
Distance = Literal["near", "far"]

FACE = "face"
OBJECT_CLASSES = ("license plate", "street sign", "house number", "storefront sign")
KNOWN_CLASSES = (FACE, *OBJECT_CLASSES)
ACTIONS: tuple[Action, ...] = ("blur", "show")

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
class CensoredFrame:
    """The only frame type the preview and the recorder accept.

    Built only by vision/render.py and by the fail-closed path in pipeline.py.
    """

    image: np.ndarray  # BGR, already pixelated
    index: int
    fail_closed: bool  # True when the whole frame was pixelated (watchdog, panic, error)


# --- policy ----------------------------------------------------------------


@dataclass(frozen=True)
class PolicySnapshot:
    """An immutable copy of the Policy, safe to read without the lock."""

    class_rules: Mapping[str, Action]
    track_overrides: Mapping[str, Action]
    panic: bool


class Policy:
    """The only state the agent and the buttons can change. Thread-safe.

    The hard limits live here so no caller can skip them: faces can never be
    shown as a class, unknown labels are refused, and blurring faces clears
    every face override.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._class_rules: dict[str, Action] = {label: "blur" for label in KNOWN_CLASSES}
        self._track_overrides: dict[str, Action] = {}
        self._panic = False

    def snapshot(self) -> PolicySnapshot:
        with self._lock:
            return PolicySnapshot(
                class_rules=MappingProxyType(dict(self._class_rules)),
                track_overrides=MappingProxyType(dict(self._track_overrides)),
                panic=self._panic,
            )

    def to_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "class_rules": dict(snapshot.class_rules),
            "track_overrides": dict(snapshot.track_overrides),
            "panic": snapshot.panic,
        }

    def set_class_rule(self, label: str, action: Action) -> None:
        """Raises ValueError for an unknown label or action, or for showing all faces."""
        _require_action(action)
        if label not in KNOWN_CLASSES:
            raise ValueError(f"unknown label {label!r}; known: {', '.join(KNOWN_CLASSES)}")
        if label == FACE and action == "show":
            raise ValueError("faces can only be shown one track at a time")
        with self._lock:
            self._class_rules[label] = action
            if label == FACE:
                self._clear_overrides(FACE_ID_PREFIX)

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


EMPTY_SCENE = Scene(owner=None, tracks=(), class_rules={label: "blur" for label in KNOWN_CLASSES}, panic=False)


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
        "description": "Blur or show every object of one known class. Showing the face class is "
        "refused; blurring it also clears every per-person override.",
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
        "name": "blur_all",
        "description": "Panic on: pixelate the whole frame.",
        "parameters": _NO_ARGUMENTS,
    },
    {
        "name": "clear_panic",
        "description": "Panic off: go back to the normal blur rules.",
        "parameters": _NO_ARGUMENTS,
    },
]

TOOL_NAMES = tuple(schema["name"] for schema in TOOL_SCHEMAS)
