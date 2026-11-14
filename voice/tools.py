"""Tool handlers and the keyword fallback (WP7). Schemas come from contracts.TOOL_SCHEMAS."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from contracts import ACTIONS, FACE, KNOWN_CLASSES, Policy, Scene

_log = logging.getLogger(__name__)


class ToolHandlers:
    """Runs the 6 client-side tools against the local Policy and the live Scene."""

    def __init__(self, policy: Policy, get_scene: Callable[[], Scene]) -> None:
        self._policy = policy
        self._get_scene = get_scene

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool and return its JSON-ready result.

        Never raises: a refusal (unknown tool or label, face/show, track not in
        the live scene) comes back as {"ok": False, "error": "..."}.
        """
        try:
            arguments = _as_dict(arguments)
            if name == "get_scene":
                return {"ok": True, **self._get_scene().to_dict()}
            if name == "set_class_rule":
                return self._set_class_rule(arguments)
            if name == "set_track":
                return self._set_track(arguments)
            if name == "blur_all":
                self._policy.set_panic(True)
                return {"ok": True, "panic": True}
            if name == "clear_panic":
                self._policy.set_panic(False)
                return {"ok": True, "panic": False}
            if name == "show_all":
                return self._show_all()
            return _refuse(f"unknown tool {name!r}")
        except ValueError as error:  # contracts.Policy's hard limits, bad arguments
            return _refuse(str(error))
        except Exception as error:  # a broken scene source must not crash the voice loop
            _log.exception("tool %r failed", name)
            return _refuse(f"{name} failed: {type(error).__name__}")

    def _set_class_rule(self, arguments: dict[str, Any]) -> dict[str, Any]:
        label = _string_argument(arguments, "label").strip().lower()
        action = _string_argument(arguments, "action").strip().lower()
        self._policy.set_class_rule(label, action)  # type: ignore[arg-type]
        result: dict[str, Any] = {"ok": True, "label": label, "action": action}
        if label == FACE and action == "blur":
            result["note"] = "everyone is blurred again; every per-person choice was cleared"
        elif label == FACE:
            result["note"] = "everyone is unblurred, including people who arrive later"
        return result

    def _set_track(self, arguments: dict[str, Any]) -> dict[str, Any]:
        track_id = _string_argument(arguments, "track_id").strip()
        action = _string_argument(arguments, "action").strip().lower()
        if action not in ACTIONS:
            return _refuse(f"unknown action {action!r}; use blur or show")
        if track_id not in self._get_scene().track_ids():
            return _refuse(f"track {track_id!r} is not in the live scene; call get_scene again")
        self._policy.set_track_override(track_id, action)  # type: ignore[arg-type]
        return {"ok": True, "track_id": track_id, "action": action}

    def _show_all(self) -> dict[str, Any]:
        self._policy.show_all()
        return {
            "ok": True,
            "classes_shown": list(KNOWN_CLASSES),
            "panic": False,
            "note": "everything stays unblurred, including people who arrive later; "
            "things blurred one by one stay blurred",
        }


def _refuse(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


def _as_dict(arguments: Any) -> dict[str, Any]:
    """Accept a dict, a JSON object string or nothing; refuse anything else."""
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            raise ValueError("arguments are not valid JSON") from None
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a JSON object")
    return arguments


def _string_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing or invalid argument {key!r}")
    return value


# --- keyword fallback ------------------------------------------------------

# Any of these means the command is about one person: the model has to handle it.
_PERSON_WORDS = {
    "person", "people", "guy", "guys", "girl", "boy", "man", "woman", "lady", "kid", "child",
    "friend", "him", "her", "he", "she", "they", "them", "me", "myself", "owner", "one",
    "left", "right", "middle", "center", "near", "far", "this", "that",
    "track", "id",
}
_BLUR_WORDS = {"blur", "blurred", "hide", "hidden", "pixelate", "censor", "cover", "mask", "block"}
_SHOW_WORDS = {
    "unblur", "unblurred", "show", "shown", "reveal", "unhide", "uncover", "unpixelate", "unmask", "unblock",
}
_CLEAR_PANIC_PHRASES = (
    "clear panic", "clear the panic", "panic off", "stop panic", "stop the panic", "cancel panic",
    "end panic", "unpanic", "back to normal", "turn off panic", "turn panic off",
)
_PANIC_PHRASES = (
    "panic", "blur everything", "hide everything", "pixelate everything", "blur it all",
    "blur all of it", "hide it all", "censor everything",
)
# Phrase -> known label. Checked longest first so "street signs" wins over "signs".
_LABEL_PHRASES = {
    "faces": FACE, "face": FACE, "everyone": FACE, "everybody": FACE,
    "license plates": "license plate", "license plate": "license plate",
    "number plates": "license plate", "number plate": "license plate",
    "plates": "license plate", "plate": "license plate",
    "street signs": "street sign", "street sign": "street sign",
    "road signs": "street sign", "road sign": "street sign",
    "house numbers": "house number", "house number": "house number",
    "addresses": "house number", "address": "house number",
    "storefront signs": "storefront sign", "storefront sign": "storefront sign",
    "storefronts": "storefront sign", "storefront": "storefront sign",
    "store signs": "storefront sign", "shop signs": "storefront sign",
}
# show_all only when the whole command, minus these fillers, is one of the phrases below,
# so "unblur all faces" (the face class) or "show all the signs" never becomes show_all.
_SHOW_ALL_PHRASES = {
    f"{verb} {what}"
    for verb in ("unblur", "show", "reveal", "unhide", "uncover", "unpixelate")
    for what in ("everything", "all", "it all", "all of it", "everything again", "all again")
}
_FILLER_WORDS = {"please", "now", "just", "ok", "okay", "hey", "can", "could", "you", "go", "ahead", "and"}
_NEGATION_WORDS = {"don't", "dont", "not", "never", "no", "stop"}
# "show my face" is about one person; "blur my face" may still blur the whole class (fails closed).
_POSSESSIVE_WORDS = {"my", "his", "your", "their"}


def parse_keywords(transcript: str) -> tuple[str, dict[str, Any]] | None:
    """Map a transcript to (tool name, arguments) without the model, or None if nothing matches.

    Only class-wide, show-everything and panic commands; anything about a specific person
    returns None, and so does a negated show ("don't show the plates") or "show my face".
    """
    if not isinstance(transcript, str):
        return None
    text = " ".join(re.sub(r"[^a-z0-9#' ]+", " ", transcript.lower()).split())
    words = set(text.split())
    if not words or words & _PERSON_WORDS or "#" in text:
        return None

    if _contains_phrase(text, _CLEAR_PANIC_PHRASES):
        return ("clear_panic", {})
    if _contains_phrase(text, _PANIC_PHRASES):
        return ("blur_all", {})
    if words & _SHOW_WORDS and words & (_NEGATION_WORDS | _POSSESSIVE_WORDS):
        return None  # a misheard "don't", or "show my face", must never unblur everyone
    if " ".join(word for word in text.split() if word not in _FILLER_WORDS) in _SHOW_ALL_PHRASES:
        return ("show_all", {})

    labels = _labels_in(text)
    wants_blur, wants_show = bool(words & _BLUR_WORDS), bool(words & _SHOW_WORDS)
    if len(labels) != 1 or wants_blur == wants_show:
        return None  # nothing, or more than one thing, to act on
    action = "show" if wants_show else "blur"
    return ("set_class_rule", {"label": labels.pop(), "action": action})


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _labels_in(text: str) -> set[str]:
    labels = set()
    remaining = f" {text} "
    for phrase in sorted(_LABEL_PHRASES, key=len, reverse=True):
        if f" {phrase} " in remaining:
            labels.add(_LABEL_PHRASES[phrase])
            remaining = remaining.replace(f" {phrase} ", " ")
    return labels
