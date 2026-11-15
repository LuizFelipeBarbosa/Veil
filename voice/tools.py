"""Tool handlers and the keyword fallback (WP7, WP21). Schemas come from contracts.TOOL_SCHEMAS."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Sequence

import config
from contracts import ACTIONS, AD_CLASSES, FACE, KNOWN_CLASSES, Policy, Scene

if TYPE_CHECKING:
    from recorder import Recorder

_log = logging.getLogger(__name__)

# The refusal when there is nothing to replace ads with; the keyword fallback says it out loud.
NO_AD_IMAGES = "There are no ad images on the device."
# What the keyword fallback says after these tools succeed; every other tool gets "Done.".
KEYWORD_REPLIES = {"start_recording": "Recording.", "stop_recording": "Recording stopped."}


class ToolHandlers:
    """Runs the 9 client-side tools against the local Policy, the live Scene and the Recorder."""

    def __init__(
        self,
        policy: Policy,
        get_scene: Callable[[], Scene],
        ad_assets: Sequence[str] = (),
        *,
        recorder: Recorder | None = None,
    ) -> None:
        """ad_assets: the image file names in config.ADS_DIR, the only things an ad can be replaced with.

        recorder: the app's Recorder (thread-safe); without one the recording tools refuse.
        """
        self._policy = policy
        self._get_scene = get_scene
        self._ad_assets = tuple(ad_assets)
        self._recorder = recorder

    @property
    def ad_assets(self) -> tuple[str, ...]:
        return self._ad_assets

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool and return its JSON-ready result.

        Never raises: a refusal (unknown tool, label or ad image, track not in
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
            if name == "set_replacement":
                return self._set_replacement(arguments)
            if name == "blur_all":
                self._policy.set_panic(True)
                return {"ok": True, "panic": True}
            if name == "clear_panic":
                self._policy.set_panic(False)
                return {"ok": True, "panic": False}
            if name == "show_all":
                return self._show_all()
            if name in ("start_recording", "stop_recording"):
                return self._recording_tool(name)
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

    def _set_replacement(self, arguments: dict[str, Any]) -> dict[str, Any]:
        label = _string_argument(arguments, "label").strip().lower()
        requested = _string_argument(arguments, "asset")
        if label not in AD_CLASSES:
            return _refuse(f"only ads can be replaced ({', '.join(AD_CLASSES)}), not {label!r}")
        if not self._ad_assets:
            return _refuse(NO_AD_IMAGES)
        asset = _find_asset(requested, self._ad_assets)
        if asset is None:
            return _refuse(f"no single ad image matches {requested!r}; use one of: {', '.join(self._ad_assets)}")
        self._policy.set_replacement(label, asset)
        return {"ok": True, "label": label, "asset": asset}

    def _show_all(self) -> dict[str, Any]:
        self._policy.show_all()
        class_rules = self._policy.snapshot().class_rules
        return {
            "ok": True,
            "classes_shown": [label for label in KNOWN_CLASSES if class_rules[label] == "show"],
            "panic": False,
            "note": "everything stays unblurred, including people who arrive later; "
            "things blurred one by one stay blurred and replaced ads stay replaced",
        }

    def _recording_tool(self, name: str) -> dict[str, Any]:
        """Start or stop the Recorder; both are safe to repeat. Never touches the Policy."""
        recorder = self._recorder
        if recorder is None:
            return _refuse("there is no recorder on this device")
        try:
            if name == "start_recording":
                already = recorder.is_recording
                recorder.start()
                return {"ok": True, "recording": True, "already": already}
            path = recorder.stop()
            return {"ok": True, "recording": False, "already": path is None, "file": path.name if path else None}
        except Exception as error:  # e.g. the video writer cannot open: report it, change nothing else
            _log.exception("tool %r failed", name)
            return {**_refuse(f"{name} failed: {type(error).__name__}"), **_recording_state(recorder)}


def _recording_state(recorder: Recorder) -> dict[str, Any]:
    """{"recording": bool} when the recorder can still say, else {}."""
    try:
        return {"recording": bool(recorder.is_recording)}
    except Exception:  # noqa: BLE001
        return {}


def _find_asset(requested: str, ad_assets: tuple[str, ...]) -> str | None:
    """The device's file name for a requested ad image, or None when none or several match.

    An exact name (ignoring case) wins; otherwise the extension, case, "-" and "_" don't matter,
    so "Summer Sale" finds summer_sale.png.
    """
    exact = [asset for asset in ad_assets if asset.casefold() == requested.strip().casefold()]
    loose = [asset for asset in ad_assets if _asset_key(asset) == _asset_key(requested)]
    matches = exact or loose
    return matches[0] if len(matches) == 1 else None


def _asset_key(name: str) -> str:
    stem, suffix = os.path.splitext(name.strip())
    if suffix.lower() in config.AD_IMAGE_SUFFIXES:
        name = stem
    return " ".join(re.sub(r"[_-]+", " ", name.casefold()).split())


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
    "billboards": "billboard", "billboard": "billboard",
    "posters": "poster", "poster": "poster",
    "advertisements": "advertisement", "advertisement": "advertisement",
}
# Phrase -> every label it names: one each, and "ads" for all three ad classes.
_LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    **{phrase: (label,) for phrase, label in _LABEL_PHRASES.items()},
    **{phrase: AD_CLASSES for phrase in ("ads", "adverts", "advertising")},
}
_REPLACE_WORDS = {"replace", "swap"}
_STOP_REPLACING_PHRASES = ("stop replacing", "stop swapping", "ads off", "adverts off")
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
# Recording. "stop" alone is not here: it may mean stop panic or stop replacing ads.
_STOP_RECORDING_PHRASES = ("stop recording", "stop the recording", "end recording", "end the recording")
_START_RECORDING_PHRASES = ("start recording", "start the recording", "start a recording", "begin recording", "record")


def parse_keywords(transcript: str, ad_assets: Sequence[str] = ()) -> list[tuple[str, dict[str, Any]]]:
    """Map a transcript to tool calls (name, arguments) without the model; [] if nothing matches.

    Only recording, class-wide, show-everything, panic and ad-replacement commands; anything about a specific
    person returns [], and so does a negated show ("don't show the plates") or "show my face".
    "Replace the ads" uses the ad image the command names, else the first of ad_assets.
    """
    if not isinstance(transcript, str):
        return []
    text = " ".join(re.sub(r"[^a-z0-9#' ]+", " ", transcript.lower()).split())
    words = set(text.split())
    if not words:
        return []
    recording = _recording_call(text, words)
    if recording:
        return [recording]
    if words & _PERSON_WORDS or "#" in text:
        return []

    if _contains_phrase(text, _CLEAR_PANIC_PHRASES):
        return [("clear_panic", {})]
    if _contains_phrase(text, _PANIC_PHRASES):
        return [("blur_all", {})]
    groups = _label_groups_in(text)
    if words & _REPLACE_WORDS or _contains_phrase(text, _STOP_REPLACING_PHRASES):
        return _ad_calls(text, words, groups, tuple(ad_assets))
    if words & _SHOW_WORDS and words & (_NEGATION_WORDS | _POSSESSIVE_WORDS):
        return []  # a misheard "don't", or "show my face", must never unblur everyone
    if " ".join(word for word in text.split() if word not in _FILLER_WORDS) in _SHOW_ALL_PHRASES:
        return [("show_all", {})]

    wants_blur, wants_show = bool(words & _BLUR_WORDS), bool(words & _SHOW_WORDS)
    if len(groups) != 1 or wants_blur == wants_show:
        return []  # nothing, or more than one thing, to act on
    action = "show" if wants_show else "blur"
    return [("set_class_rule", {"label": label, "action": action}) for label in groups.pop()]


def _recording_call(text: str, words: set[str]) -> tuple[str, dict[str, Any]] | None:
    """start_recording or stop_recording for a command that is only about recording, else None.

    Checked before the person words, so "record this" works. A command that also blurs, shows,
    replaces or names a type ("stop recording and blur everything") is left to the rest of the
    parser and the model, so panic and blur commands keep priority; a negation matches nothing.
    """
    if words & (_BLUR_WORDS | _SHOW_WORDS | _REPLACE_WORDS | {"panic", "don't", "dont", "not", "never", "no"}):
        return None
    if _label_groups_in(text):
        return None
    if _contains_phrase(text, _STOP_RECORDING_PHRASES):
        return ("stop_recording", {})
    if _contains_phrase(text, _START_RECORDING_PHRASES):
        return ("start_recording", {})
    return None


def _ad_calls(
    text: str, words: set[str], groups: set[tuple[str, ...]], ad_assets: tuple[str, ...]
) -> list[tuple[str, dict[str, Any]]]:
    """Replace ads, or stop replacing them (show them again). [] for a negation or anything but ads."""
    named = {label for group in groups for label in group}
    if not named <= set(AD_CLASSES):
        return []  # "replace the faces": the model explains why not
    labels = [label for label in AD_CLASSES if label in named] or list(AD_CLASSES)
    if _contains_phrase(text, _STOP_REPLACING_PHRASES):
        return [("set_class_rule", {"label": label, "action": "show"}) for label in labels]
    if not named or words & _NEGATION_WORDS:
        return []
    asset = _asset_named_in(text, ad_assets)
    if asset is None:
        return []
    return [("set_replacement", {"label": label, "asset": asset}) for label in labels]


def _asset_named_in(text: str, ad_assets: tuple[str, ...]) -> str | None:
    """The ad image a command names, else the first one; None when it names one the device lacks.

    "" when there are no images at all, so the handler refuses with NO_AD_IMAGES.
    """
    if not ad_assets:
        return ""
    named = [asset for asset in ad_assets if _contains_phrase(text, (_asset_key(asset),))]
    if named:
        return named[0]
    return None if "with" in text.split() else ad_assets[0]  # "with pepsi" and no pepsi image


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _label_groups_in(text: str) -> set[tuple[str, ...]]:
    groups = set()
    remaining = f" {text} "
    for phrase in sorted(_LABEL_GROUPS, key=len, reverse=True):
        if f" {phrase} " in remaining:
            groups.add(_LABEL_GROUPS[phrase])
            remaining = remaining.replace(f" {phrase} ", " ")
    return groups
