"""WP7: tool handlers, hard limits and the keyword parser. Offline: no network, key, mic or speaker."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

import config
from contracts import AD_CLASSES, FACE, KNOWN_CLASSES, OBJECT_CLASSES, TOOL_NAMES, Policy, Scene
from fakes import scenes
from voice.dg_agent import VoiceAgent, _Turn, build_settings
from voice.tools import NO_AD_IMAGES, ToolHandlers, parse_keywords


def _handlers(scene: Scene = scenes.OWNER_AND_FRIEND_LEFT) -> tuple[ToolHandlers, Policy]:
    policy = Policy()
    return ToolHandlers(policy, lambda: scene), policy


# --- hard limits ---


def test_showing_the_face_class_keeps_everyone_unblurred() -> None:
    handlers, policy = _handlers()
    result = handlers.handle("set_class_rule", {"label": "face", "action": "show"})
    assert result["ok"] is True and "later" in result["note"]
    assert policy.snapshot().class_rules[FACE] == "show"
    assert policy.snapshot().face_reset_count == 0


def test_blurring_the_face_class_after_showing_it_blurs_everyone_again() -> None:
    handlers, policy = _handlers()
    assert handlers.handle("set_class_rule", {"label": "face", "action": "show"})["ok"]
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "blur"})["ok"]
    result = handlers.handle("set_class_rule", {"label": "face", "action": "blur"})
    assert result["ok"] is True and "cleared" in result["note"]
    snapshot = policy.snapshot()
    assert snapshot.class_rules[FACE] == "blur"
    assert snapshot.track_overrides == {}
    assert snapshot.face_reset_count == 1


def test_one_person_can_be_blurred_while_the_face_class_is_shown() -> None:
    handlers, policy = _handlers()
    assert handlers.handle("set_class_rule", {"label": "face", "action": "show"})["ok"]
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "blur"})["ok"]
    assert policy.snapshot().class_rules[FACE] == "show"
    assert policy.snapshot().track_overrides == {"face#7": "blur"}


def test_blurring_faces_clears_face_overrides() -> None:
    handlers, policy = _handlers()
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "show"})["ok"]
    assert handlers.handle("set_track", {"track_id": "obj#3", "action": "show"})["ok"]
    assert handlers.handle("set_class_rule", {"label": "face", "action": "blur"})["ok"]
    assert policy.snapshot().track_overrides == {"obj#3": "show"}


@pytest.mark.parametrize("label", ["logo", "logos", "tattoo", "", "faces"])
def test_unknown_labels_are_refused(label: str) -> None:
    handlers, policy = _handlers()
    before = policy.to_dict()
    assert handlers.handle("set_class_rule", {"label": label, "action": "blur"})["ok"] is False
    assert policy.to_dict() == before


def test_unknown_actions_are_refused() -> None:
    handlers, policy = _handlers()
    before = policy.to_dict()
    assert handlers.handle("set_class_rule", {"label": "street sign", "action": "delete"})["ok"] is False
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "delete"})["ok"] is False
    assert policy.to_dict() == before


def test_set_track_rejects_ids_not_in_the_live_scene() -> None:
    live = {"scene": scenes.OWNER_AND_FRIEND_LEFT}
    policy = Policy()
    handlers = ToolHandlers(policy, lambda: live["scene"])
    assert handlers.handle("set_track", {"track_id": "face#99", "action": "show"})["ok"] is False
    # face#7 was there a moment ago but has left: checked at call time
    live["scene"] = scenes.SIGNS_AND_PLATE
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "show"})["ok"] is False
    assert policy.snapshot().track_overrides == {}


def test_every_known_object_class_can_be_shown_and_blurred() -> None:
    handlers, policy = _handlers()
    for label in OBJECT_CLASSES:
        assert handlers.handle("set_class_rule", {"label": label, "action": "show"})["ok"]
        assert policy.snapshot().class_rules[label] == "show"
        assert handlers.handle("set_class_rule", {"label": label, "action": "blur"})["ok"]
        assert policy.snapshot().class_rules[label] == "blur"


# --- the Brief's command table, at handler level ---


def test_blur_all_faces() -> None:
    handlers, policy = _handlers()
    policy.set_track_override("face#7", "show")
    assert handlers.handle("set_class_rule", {"label": "face", "action": "blur"})["ok"]
    assert policy.snapshot().class_rules[FACE] == "blur"
    assert policy.snapshot().track_overrides == {}


def test_unblur_the_person_on_my_left() -> None:
    handlers, policy = _handlers(scenes.OWNER_AND_FRIEND_LEFT)
    scene = handlers.handle("get_scene", {})
    assert scene["ok"] is True
    assert scene["owner"] == "face#2"
    left = [track["id"] for track in scene["tracks"] if track["side"] == "owner_left"]
    assert left == ["face#7"]
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "show"}) == {
        "ok": True, "track_id": "face#7", "action": "show",
    }
    assert policy.snapshot().track_overrides == {"face#7": "show"}


def test_show_the_street_signs_again() -> None:
    handlers, policy = _handlers(scenes.SIGNS_AND_PLATE)
    assert handlers.handle("set_class_rule", {"label": "street sign", "action": "show"})["ok"]
    assert policy.snapshot().class_rules["street sign"] == "show"


def test_blur_everything_and_clear_panic() -> None:
    handlers, policy = _handlers()
    assert handlers.handle("blur_all", {}) == {"ok": True, "panic": True}
    assert policy.snapshot().panic is True
    assert handlers.handle("clear_panic", {}) == {"ok": True, "panic": False}
    assert policy.snapshot().panic is False


def test_unblur_the_guy_two_candidates_scene_lists_both() -> None:
    handlers, policy = _handlers(scenes.TWO_CANDIDATES)
    scene = handlers.handle("get_scene", {})
    candidates = [t["id"] for t in scene["tracks"] if t["label"] == FACE and t["id"] != scene["owner"]]
    assert candidates == ["face#7", "face#9"]
    assert policy.snapshot().track_overrides == {}  # get_scene changes nothing


def test_hide_the_logos_is_refused() -> None:
    handlers, policy = _handlers()
    before = policy.to_dict()
    assert handlers.handle("set_class_rule", {"label": "logo", "action": "blur"})["ok"] is False
    assert policy.to_dict() == before


def test_get_scene_returns_the_scene_dict() -> None:
    for scene in scenes.ALL.values():
        handlers, _ = _handlers(scene)
        assert handlers.handle("get_scene", {}) == {"ok": True, **scene.to_dict()}


# --- show_all: every class, faces included; per-track blur choices stay ---


def test_show_all_shows_every_class_and_turns_panic_off() -> None:
    handlers, policy = _handlers(scenes.OWNER_AND_FRIEND_LEFT)
    policy.set_panic(True)
    result = handlers.handle("show_all", {})
    assert result["ok"] is True
    assert set(result["classes_shown"]) == set(KNOWN_CLASSES)
    snapshot = policy.snapshot()
    assert all(snapshot.class_rules[label] == "show" for label in KNOWN_CLASSES)
    assert snapshot.track_overrides == {}
    assert snapshot.panic is False


def test_show_all_keeps_things_blurred_one_by_one() -> None:
    handlers, policy = _handlers(scenes.OWNER_AND_FRIEND_LEFT)
    assert handlers.handle("set_track", {"track_id": "obj#3", "action": "blur"})["ok"]
    assert handlers.handle("show_all", {})["ok"]
    assert policy.snapshot().track_overrides == {"obj#3": "blur"}


def test_show_all_does_not_need_the_scene() -> None:
    def broken() -> Scene:
        raise RuntimeError("vision is down")

    policy = Policy()
    result = ToolHandlers(policy, broken).handle("show_all", {})
    assert result["ok"] is True
    assert policy.snapshot().class_rules[FACE] == "show"


def test_blurring_faces_after_show_all_reblurs_every_person() -> None:
    handlers, policy = _handlers()
    assert handlers.handle("show_all", {})["ok"]
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "show"})["ok"]
    assert handlers.handle("set_class_rule", {"label": "face", "action": "blur"})["ok"]
    snapshot = policy.snapshot()
    assert snapshot.class_rules[FACE] == "blur"
    assert snapshot.track_overrides == {}
    assert all(snapshot.class_rules[label] == "show" for label in OBJECT_CLASSES)


# --- handle never raises ---


GARBAGE_CALLS = [
    ("nope", {}),
    ("", {}),
    (None, {}),
    ("set_class_rule", None),
    ("set_class_rule", {}),
    ("set_class_rule", {"label": None, "action": None}),
    ("set_class_rule", {"label": 3, "action": ["blur"]}),
    ("set_class_rule", "not json"),
    ("set_class_rule", "[1, 2]"),
    ("set_class_rule", 42),
    ("set_track", {"track_id": {"x": 1}, "action": "show"}),
    ("set_track", {"track_id": "face#7"}),
    ("set_track", {"action": "show"}),
    ("set_track", ["face#7", "show"]),
]


@pytest.mark.parametrize("name, arguments", GARBAGE_CALLS)
def test_handle_never_raises_on_garbage(name, arguments) -> None:
    handlers, policy = _handlers()
    before = policy.to_dict()
    result = handlers.handle(name, arguments)
    assert result["ok"] is False and isinstance(result["error"], str)
    assert policy.to_dict() == before


def test_handle_accepts_json_string_arguments() -> None:
    handlers, policy = _handlers()
    assert handlers.handle("set_class_rule", '{"label": "house number", "action": "show"}')["ok"]
    assert policy.snapshot().class_rules["house number"] == "show"


def test_a_broken_scene_source_is_a_refusal() -> None:
    def broken() -> Scene:
        raise RuntimeError("vision is down")

    policy = Policy()
    handlers = ToolHandlers(policy, broken)
    assert handlers.handle("get_scene", {})["ok"] is False
    assert handlers.handle("set_track", {"track_id": "face#7", "action": "show"})["ok"] is False
    assert policy.snapshot().track_overrides == {}


# --- keyword parser ---


@pytest.mark.parametrize(
    "transcript, expected",
    [
        ("Blur all faces.", ("set_class_rule", {"label": "face", "action": "blur"})),
        ("blur the faces", ("set_class_rule", {"label": "face", "action": "blur"})),
        ("Show the street signs again.", ("set_class_rule", {"label": "street sign", "action": "show"})),
        ("Hide license plates", ("set_class_rule", {"label": "license plate", "action": "blur"})),
        ("unblur the plates", ("set_class_rule", {"label": "license plate", "action": "show"})),
        ("pixelate house numbers", ("set_class_rule", {"label": "house number", "action": "blur"})),
        ("show storefronts", ("set_class_rule", {"label": "storefront sign", "action": "show"})),
        ("Blur everything.", ("blur_all", {})),
        ("Panic!", ("blur_all", {})),
        ("clear panic", ("clear_panic", {})),
        ("Panic off.", ("clear_panic", {})),
        ("back to normal", ("clear_panic", {})),
        ("Unblur everything.", ("show_all", {})),
        ("show everything", ("show_all", {})),
        ("Unblur all.", ("show_all", {})),
        ("okay, unblur it all please", ("show_all", {})),
        ("show all the plates", ("set_class_rule", {"label": "license plate", "action": "show"})),
        ("unblur everyone", ("set_class_rule", {"label": "face", "action": "show"})),
        ("Unblur all faces.", ("set_class_rule", {"label": "face", "action": "show"})),
        ("keep everyone unblurred", ("set_class_rule", {"label": "face", "action": "show"})),
        ("show all the faces", ("set_class_rule", {"label": "face", "action": "show"})),
        ("blur everyone", ("set_class_rule", {"label": "face", "action": "blur"})),
        ("keep everybody blurred", ("set_class_rule", {"label": "face", "action": "blur"})),
    ],
)
def test_keyword_parser_matches(transcript: str, expected) -> None:
    assert parse_keywords(transcript) == [expected]


@pytest.mark.parametrize(
    "transcript",
    [
        "Unblur the person on my left, she's with me.",
        "Unblur the guy.",
        "show her face",
        "show face#7",
        "blur the one in the back",
        "hide the logos",
        "blur faces and show plates",
        "hide plates and street signs",
        "hello there",
        "",
        "show all the signs",
        "unblur all the logos",
        "unblur everything except me",
        "don't unblur everything",
        "do not show the plates",
        "don't unblur everyone",
        "never show faces",
    ],
)
def test_keyword_parser_declines(transcript: str) -> None:
    assert parse_keywords(transcript) == []


def test_keyword_parser_never_turns_a_face_command_into_show_all() -> None:
    for verb in ("show", "unblur", "reveal", "unhide"):
        for noun in ("face", "faces", "all faces", "everyone", "all the faces"):
            assert parse_keywords(f"{verb} {noun}") == [("set_class_rule", {"label": FACE, "action": "show"})]


def test_keyword_parser_never_shows_one_person() -> None:
    for verb in ("show", "unblur", "reveal"):
        for who in ("her", "him", "the guy", "the person on my left", "face#7", "my face"):
            assert parse_keywords(f"{verb} {who}") == []


def test_keyword_parser_ignores_non_strings() -> None:
    assert parse_keywords(None) == []  # type: ignore[arg-type]


def test_keyword_matches_run_through_the_handlers() -> None:
    handlers, policy = _handlers()
    [(name, arguments)] = parse_keywords("hide license plates")
    assert handlers.handle(name, arguments)["ok"]
    assert policy.snapshot().class_rules["license plate"] == "blur"


def test_keyword_show_all_runs_through_the_handlers() -> None:
    handlers, policy = _handlers(scenes.TWO_CANDIDATES)
    [(name, arguments)] = parse_keywords("unblur everything")
    assert handlers.handle(name, arguments)["ok"]
    assert all(policy.snapshot().class_rules[label] == "show" for label in KNOWN_CLASSES)


# --- ad replacement (WP21) ---

ASSETS = ("summer_sale.png", "Coffee.jpg", "concert-poster.jpeg")


def _ad_handlers(ad_assets: tuple[str, ...] = ASSETS) -> tuple[ToolHandlers, Policy]:
    policy = Policy()
    return ToolHandlers(policy, lambda: scenes.EMPTY, ad_assets=ad_assets), policy


def _replace_all(asset: str) -> list:
    return [("set_replacement", {"label": label, "asset": asset}) for label in AD_CLASSES]


def _show_all_ads() -> list:
    return [("set_class_rule", {"label": label, "action": "show"}) for label in AD_CLASSES]


def test_set_replacement_replaces_an_ad_class() -> None:
    handlers, policy = _ad_handlers()
    result = handlers.handle("set_replacement", {"label": "poster", "asset": "summer_sale.png"})
    assert result == {"ok": True, "label": "poster", "asset": "summer_sale.png"}
    snapshot = policy.snapshot()
    assert snapshot.class_rules["poster"] == "replace"
    assert snapshot.replacements == {"poster": "summer_sale.png"}


@pytest.mark.parametrize(
    "requested, on_device",
    [
        ("Coffee.jpg", "Coffee.jpg"),
        ("coffee.JPG", "Coffee.jpg"),
        ("coffee", "Coffee.jpg"),
        ("Summer Sale", "summer_sale.png"),
        ("summer-sale.png", "summer_sale.png"),
        ("concert poster", "concert-poster.jpeg"),
    ],
)
def test_set_replacement_forgives_case_and_extension_but_stores_the_file_name(requested: str, on_device: str) -> None:
    handlers, policy = _ad_handlers()
    assert handlers.handle("set_replacement", {"label": "billboard", "asset": requested})["ok"]
    assert policy.snapshot().replacements == {"billboard": on_device}


@pytest.mark.parametrize(
    "arguments",
    [
        {"label": "face", "asset": "coffee.jpg"},
        {"label": "license plate", "asset": "coffee.jpg"},
        {"label": "street sign", "asset": "coffee.jpg"},
        {"label": "posters", "asset": "coffee.jpg"},
        {"label": "poster", "asset": "pepsi.png"},
        {"label": "poster", "asset": ""},
        {"label": "poster", "asset": "../secret.png"},
        {"label": "poster"},
        {"asset": "coffee.jpg"},
        {"label": "poster", "asset": 3},
    ],
)
def test_set_replacement_refuses_and_changes_nothing(arguments: dict) -> None:
    handlers, policy = _ad_handlers()
    before = policy.to_dict()
    result = handlers.handle("set_replacement", arguments)
    assert result["ok"] is False and isinstance(result["error"], str)
    assert policy.to_dict() == before


def test_set_replacement_refuses_an_ambiguous_name() -> None:
    handlers, policy = _ad_handlers(("sale.png", "sale.jpg"))
    assert handlers.handle("set_replacement", {"label": "poster", "asset": "sale"})["ok"] is False
    assert handlers.handle("set_replacement", {"label": "poster", "asset": "sale.jpg"})["ok"]
    assert policy.snapshot().replacements == {"poster": "sale.jpg"}


def test_set_replacement_without_assets_says_there_are_none() -> None:
    handlers, policy = _ad_handlers(())
    result = handlers.handle("set_replacement", {"label": "poster", "asset": "coffee.jpg"})
    assert result == {"ok": False, "error": NO_AD_IMAGES}
    assert policy.snapshot().replacements == {}


def test_showing_a_replaced_class_turns_the_replacement_off() -> None:
    handlers, policy = _ad_handlers()
    assert handlers.handle("set_replacement", {"label": "billboard", "asset": "coffee"})["ok"]
    assert handlers.handle("set_class_rule", {"label": "billboard", "action": "show"})["ok"]
    snapshot = policy.snapshot()
    assert snapshot.class_rules["billboard"] == "show" and snapshot.replacements == {}


def test_show_all_keeps_replaced_ads_and_says_so() -> None:
    handlers, policy = _ad_handlers()
    assert handlers.handle("set_replacement", {"label": "poster", "asset": "coffee"})["ok"]
    result = handlers.handle("show_all", {})
    assert "poster" not in result["classes_shown"] and FACE in result["classes_shown"]
    assert policy.snapshot().replacements == {"poster": "Coffee.jpg"}


@pytest.mark.parametrize(
    "transcript, expected",
    [
        ("Replace the ads.", _replace_all("summer_sale.png")),
        ("replace ads", _replace_all("summer_sale.png")),
        ("swap the adverts", _replace_all("summer_sale.png")),
        ("replace the ads with coffee", _replace_all("Coffee.jpg")),
        ("Replace the ads with the concert poster.", _replace_all("concert-poster.jpeg")),
        ("replace the posters", [("set_replacement", {"label": "poster", "asset": "summer_sale.png"})]),
        ("replace the billboards with summer sale",
         [("set_replacement", {"label": "billboard", "asset": "summer_sale.png"})]),
        ("Stop replacing the ads.", _show_all_ads()),
        ("stop replacing", _show_all_ads()),
        ("ads off", _show_all_ads()),
        ("turn the ads off please", _show_all_ads()),
        ("Turn off the ads.", _show_all_ads()),
        ("turn off ads", _show_all_ads()),
        ("disable the ads", _show_all_ads()),
        ("ads on", _replace_all("summer_sale.png")),
        ("Turn on the ads.", _replace_all("summer_sale.png")),
        ("turn on ads", _replace_all("summer_sale.png")),
        ("enable the ads", _replace_all("summer_sale.png")),
        ("enable ads please", _replace_all("summer_sale.png")),
        ("stop replacing the billboards", [("set_class_rule", {"label": "billboard", "action": "show"})]),
        ("Show the ads.", _show_all_ads()),
        ("show the real ads", _show_all_ads()),
        ("blur the posters", [("set_class_rule", {"label": "poster", "action": "blur"})]),
        ("hide the ads", [("set_class_rule", {"label": label, "action": "blur"}) for label in AD_CLASSES]),
        ("show the billboards", [("set_class_rule", {"label": "billboard", "action": "show"})]),
    ],
)
def test_keyword_parser_ad_commands(transcript: str, expected: list) -> None:
    assert parse_keywords(transcript, ASSETS) == expected


@pytest.mark.parametrize(
    "transcript",
    [
        "replace the faces",
        "replace the license plates",
        "replace the ads with pepsi",
        "don't replace the ads",
        "never replace the posters",
        "replace that one",
        "replace",
        "don't show the ads",
        "don't turn on the ads",
        "replace the posters and the plates",
    ],
)
def test_keyword_parser_declines_ad_commands(transcript: str) -> None:
    assert parse_keywords(transcript, ASSETS) == []


def test_keyword_replace_without_assets_is_refused_by_the_handler() -> None:
    handlers, policy = _ad_handlers(())
    calls = parse_keywords("replace the ads", handlers.ad_assets)
    assert [name for name, _ in calls] == ["set_replacement"] * len(AD_CLASSES)
    assert all(handlers.handle(name, arguments)["error"] == NO_AD_IMAGES for name, arguments in calls)
    assert policy.snapshot().replacements == {}


def test_keyword_ad_commands_run_through_the_handlers() -> None:
    handlers, policy = _ad_handlers()
    for name, arguments in parse_keywords("replace the ads", handlers.ad_assets):
        assert handlers.handle(name, arguments)["ok"]
    assert dict(policy.snapshot().replacements) == {label: "summer_sale.png" for label in AD_CLASSES}
    for name, arguments in parse_keywords("stop replacing the ads", handlers.ad_assets):
        assert handlers.handle(name, arguments)["ok"]
    snapshot = policy.snapshot()
    assert snapshot.replacements == {} and all(snapshot.class_rules[label] == "show" for label in AD_CLASSES)


def test_keyword_fallback_replaces_ads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DEEPGRAM_API_KEY_ENV, raising=False)
    handlers, policy = _ad_handlers()
    lines: list[tuple[str, str]] = []
    agent = _offline_agent(handlers, lines)
    agent._turn = _Turn(ended_at=0.0, user_text="replace the ads with coffee")
    agent._run_keyword_fallback()
    assert dict(policy.snapshot().replacements) == {label: "Coffee.jpg" for label in AD_CLASSES}
    assert lines[-1] == ("agent", "Done.")
    agent._loop.close()


def test_keyword_fallback_says_there_are_no_ad_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DEEPGRAM_API_KEY_ENV, raising=False)
    handlers, policy = _ad_handlers(())
    lines: list[tuple[str, str]] = []
    agent = _offline_agent(handlers, lines)
    agent._turn = _Turn(ended_at=0.0, user_text="replace the ads")
    agent._run_keyword_fallback()
    assert lines[-1] == ("agent", NO_AD_IMAGES)
    assert policy.snapshot().replacements == {}
    agent._loop.close()


# --- recording (WP28) ---


class _FakeRecorder:
    """Stands in for recorder.Recorder: no video is written."""

    def __init__(self, fail_start: bool = False) -> None:
        self.is_recording = False
        self._fail_start = fail_start

    def start(self) -> Path:
        if self._fail_start:
            raise RuntimeError("cannot open video writer")
        self.is_recording = True
        return Path("/recordings/privacy-cam-1.avi")

    def stop(self) -> Path | None:
        was_recording, self.is_recording = self.is_recording, False
        return Path("/recordings/privacy-cam-1.avi") if was_recording else None


def _recording_handlers(recorder: object) -> tuple[ToolHandlers, Policy]:
    policy = Policy()
    return ToolHandlers(policy, lambda: scenes.EMPTY, recorder=recorder), policy  # type: ignore[arg-type]


def test_start_recording_twice_is_safe() -> None:
    recorder = _FakeRecorder()
    handlers, policy = _recording_handlers(recorder)
    before = policy.snapshot()
    assert handlers.handle("start_recording", {}) == {"ok": True, "recording": True, "already": False}
    assert handlers.handle("start_recording", {}) == {"ok": True, "recording": True, "already": True}
    assert recorder.is_recording and policy.snapshot() == before


def test_stop_recording_twice_is_safe_and_names_the_file_only() -> None:
    recorder = _FakeRecorder()
    handlers, policy = _recording_handlers(recorder)
    handlers.handle("start_recording", {})
    first = handlers.handle("stop_recording", {})
    second = handlers.handle("stop_recording", {})
    assert first == {"ok": True, "recording": False, "already": False, "file": "privacy-cam-1.avi"}
    assert second == {"ok": True, "recording": False, "already": True, "file": None}
    assert not recorder.is_recording


def test_stop_recording_when_never_started() -> None:
    handlers, _ = _recording_handlers(_FakeRecorder())
    assert handlers.handle("stop_recording", {}) == {"ok": True, "recording": False, "already": True, "file": None}


def test_a_failed_start_is_a_refusal_and_changes_no_policy() -> None:
    handlers, policy = _recording_handlers(_FakeRecorder(fail_start=True))
    policy.set_panic(True)
    before = policy.snapshot()
    result = handlers.handle("start_recording", {})
    assert result["ok"] is False and result["recording"] is False and "error" in result
    assert policy.snapshot() == before


def test_recording_tools_without_a_recorder_refuse() -> None:
    handlers, policy = _handlers()
    before = policy.snapshot()
    for name in ("start_recording", "stop_recording"):
        result = handlers.handle(name, {})
        assert result["ok"] is False and "error" in result
    assert policy.snapshot() == before


@pytest.mark.parametrize(
    "transcript, expected",
    [
        ("Start recording.", "start_recording"),
        ("record", "start_recording"),
        ("Record this.", "start_recording"),
        ("can you start recording now", "start_recording"),
        ("Stop recording.", "stop_recording"),
        ("stop the recording please", "stop_recording"),
        ("That's enough, stop recording.", "stop_recording"),
        ("end the recording", "stop_recording"),
    ],
)
def test_keyword_parser_recording(transcript: str, expected: str) -> None:
    assert parse_keywords(transcript) == [(expected, {})]


@pytest.mark.parametrize(
    "transcript, expected",
    [
        ("stop", []),
        ("stop it", []),
        ("don't record", []),
        ("don't stop recording", []),
        ("is it recording", []),
        ("stop panic", [("clear_panic", {})]),
        ("stop the panic", [("clear_panic", {})]),
        ("panic, start recording", [("blur_all", {})]),
        ("stop recording and blur everything", [("blur_all", {})]),
        ("record and hide the plates", [("set_class_rule", {"label": "license plate", "action": "blur"})]),
        ("stop replacing ads", _show_all_ads()),
        ("record her face", []),  # names a person and a show-free face: left to the model
    ],
)
def test_keyword_recording_does_not_collide(transcript: str, expected: list) -> None:
    assert parse_keywords(transcript) == expected


@pytest.mark.parametrize(
    "transcript, reply, recording",
    [("record this", "Recording.", True), ("stop recording", "Recording stopped.", False)],
)
def test_keyword_fallback_recording_replies(
    monkeypatch: pytest.MonkeyPatch, transcript: str, reply: str, recording: bool
) -> None:
    monkeypatch.delenv(config.DEEPGRAM_API_KEY_ENV, raising=False)
    recorder = _FakeRecorder()
    recorder.is_recording = not recording
    handlers, _ = _recording_handlers(recorder)
    lines: list[tuple[str, str]] = []
    agent = _offline_agent(handlers, lines)
    agent._turn = _Turn(ended_at=0.0, user_text=transcript)
    agent._run_keyword_fallback()
    assert lines[-1] == ("agent", reply) and recorder.is_recording is recording
    agent._loop.close()


def test_keyword_fallback_says_when_recording_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DEEPGRAM_API_KEY_ENV, raising=False)
    handlers, _ = _recording_handlers(_FakeRecorder(fail_start=True))
    lines: list[tuple[str, str]] = []
    agent = _offline_agent(handlers, lines)
    agent._turn = _Turn(ended_at=0.0, user_text="start recording")
    agent._run_keyword_fallback()
    assert lines[-1] == ("agent", "Couldn't do that.")
    agent._loop.close()


# --- settings (offline) ---


def test_settings_shape() -> None:
    settings = build_settings()
    agent = settings["agent"]
    assert settings["type"] == "Settings"
    assert settings["audio"]["input"] == {"encoding": "linear16", "sample_rate": 16_000}
    listen = agent["listen"]["provider"]
    assert (listen["type"], listen["model"], listen["version"]) == ("deepgram", "flux-general-en", "v2")
    assert "unblur" in listen["keyterms"]
    think = agent["think"]
    assert think["provider"]["type"] == "open_ai" and think["provider"]["temperature"] == 0
    assert [f["name"] for f in think["functions"]] == list(TOOL_NAMES) == [
        "get_scene", "set_class_rule", "set_track", "set_replacement", "blur_all", "clear_panic", "show_all",
        "start_recording", "stop_recording",
    ]
    assert all("endpoint" not in f for f in think["functions"])
    assert "get_scene" in think["prompt"]
    assert agent["speak"]["provider"]["model"].startswith("aura-2")
    for label in KNOWN_CLASSES:
        assert label in think["prompt"]
    assert "show_all" in think["prompt"]
    assert "no ad images on this device" in think["prompt"]
    assert "start_recording" in think["prompt"] and "stop_recording" in think["prompt"]
    assert f"under {config.AGENT_REPLY_MAX_WORDS} words" in think["prompt"]
    assert "needs_confirmation" in think["prompt"] and "hey veil" in think["prompt"]


def test_settings_prompt_lists_the_ad_images() -> None:
    prompt = build_settings(ad_assets=("summer_sale.png", "coffee.jpg"))["agent"]["think"]["prompt"]
    assert "set_replacement" in prompt and "summer_sale.png, coffee.jpg" in prompt
    assert 'set_replacement(label, "summer_sale.png")' in prompt  # the default when none is named
    assert prompt.rstrip().endswith("no markdown.")


def test_agent_offers_the_handlers_ad_images() -> None:
    policy = Policy()
    agent = VoiceAgent(ToolHandlers(policy, lambda: scenes.EMPTY, ad_assets=("coffee.jpg",)), lambda r, t: None,
                       use_audio=False, think_model="gpt-4.1-mini")
    think = agent._settings["agent"]["think"]
    assert "coffee.jpg" in think["prompt"] and think["provider"]["model"] == "gpt-4.1-mini"
    agent._loop.close()


# --- VoiceAgent without network, key or audio ---


def _wait_for(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def _offline_agent(handlers: ToolHandlers, lines: list) -> VoiceAgent:
    return VoiceAgent(handlers, lambda role, text: lines.append((role, text)), use_audio=False)


def _sent(agent: VoiceAgent) -> list:
    sent = []
    while not agent._outbox.empty():
        sent.append(json.loads(agent._outbox.get_nowait()))
    return sent


def test_agent_without_key_reports_and_keyword_fallback_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DEEPGRAM_API_KEY_ENV, raising=False)
    handlers, policy = _handlers()
    lines: list[tuple[str, str]] = []
    agent = _offline_agent(handlers, lines)
    agent.start()
    try:
        assert _wait_for(lambda: any("not set" in text for _, text in lines))
        assert policy.to_dict() == Policy().to_dict()  # an error leaves the Policy untouched

        agent.inject_text("unblur the guy")  # about a person: the parser won't touch it
        agent.inject_text("blur everything")
        assert _wait_for(lambda: policy.snapshot().panic)
        assert policy.snapshot().track_overrides == {}
        assert ("you", "blur everything") in lines
        agent.set_talking(True)  # no-op without audio, must not raise
        agent.set_talking(False)
    finally:
        agent.stop()
        agent.join(timeout=3.0)
    assert not agent.is_alive()


def test_function_call_requests_get_function_call_responses() -> None:
    handlers, policy = _handlers()
    agent = _offline_agent(handlers, [])
    agent._dispatch({
        "type": "FunctionCallRequest",
        "functions": [
            {"id": "fc_1", "name": "get_scene", "arguments": "{}", "client_side": True},
            {"id": "fc_2", "name": "set_track", "arguments": '{"track_id": "face#7", "action": "show"}', "client_side": True},
            {"id": "fc_3", "name": "set_class_rule", "arguments": '{"label": "logo", "action": "show"}', "client_side": True},
            {"id": "fc_4", "name": "set_track", "arguments": "not json", "client_side": True},
            {"id": "fc_5", "name": "server_thing", "arguments": "{}", "client_side": False},
            "garbage",
        ],
    })
    responses = _sent(agent)
    assert [r["id"] for r in responses] == ["fc_1", "fc_2", "fc_3", "fc_4"]
    assert all(r["type"] == "FunctionCallResponse" for r in responses)
    results = [json.loads(r["content"]) for r in responses]
    assert results[0]["owner"] == "face#2"
    assert [r["ok"] for r in results] == [True, True, False, False]
    assert policy.snapshot().track_overrides == {"face#7": "show"}
    assert policy.snapshot().class_rules[FACE] == "blur"


def test_conversation_text_reaches_on_transcript() -> None:
    handlers, _ = _handlers()
    lines: list[tuple[str, str]] = []
    agent = _offline_agent(handlers, lines)
    agent._dispatch({"type": "ConversationText", "role": "user", "content": "blur all faces"})
    agent._dispatch({"type": "ConversationText", "role": "assistant", "content": "Faces blurred."})
    assert lines == [("you", "blur all faces"), ("agent", "Faces blurred.")]


class _FakeInputStream:
    """Stands in for sounddevice.RawInputStream: no device is opened."""

    opened: list["_FakeInputStream"] = []

    def __init__(self, *, samplerate, channels, dtype, blocksize, callback) -> None:
        assert (samplerate, channels, dtype) == (config.MIC_SAMPLE_RATE, 1, "int16")
        self.callback = callback
        self.running = False
        _FakeInputStream.opened.append(self)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.running = False


_FORCE_END_TURN = json.dumps({"type": "ForceEndTurn"})
_HELD = b"\x01\x00" * 4
_LATE = b"\x02\x00" * 4


@pytest.fixture
def mic_agent(monkeypatch: pytest.MonkeyPatch):
    """A VoiceAgent with a fake mic and a ready session; the test runs its loop by hand."""
    import types

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=_FakeInputStream))
    _FakeInputStream.opened.clear()
    handlers, _ = _handlers()
    agent = VoiceAgent(handlers, lambda role, text: None, use_audio=True)
    agent._speaker = None  # no playback thread work in these tests
    agent._ready.set()
    yield agent
    agent._loop.close()


def _run_loop(agent: VoiceAgent) -> None:
    agent._loop.run_until_complete(asyncio.sleep(0))


def _queued(agent: VoiceAgent) -> list:
    queued = []
    while not agent._outbox.empty():
        queued.append(agent._outbox.get_nowait())
    return queued


def test_talk_gate_streams_only_while_held_then_force_end_turn(mic_agent: VoiceAgent) -> None:
    mic_agent.set_talking(True)
    _run_loop(mic_agent)
    (mic,) = _FakeInputStream.opened
    assert mic.running
    mic.callback(_HELD, 4, None, None)
    mic_agent.set_talking(False)
    _run_loop(mic_agent)
    mic.callback(_LATE, 4, None, None)  # a callback after the loop closed the mic is dropped too
    _run_loop(mic_agent)

    assert not mic.running and mic_agent._mic is None
    assert _queued(mic_agent) == [_HELD, _FORCE_END_TURN]


def test_a_block_after_release_returns_is_dropped_before_the_loop_runs(mic_agent: VoiceAgent) -> None:
    mic_agent.set_talking(True)
    _run_loop(mic_agent)
    (mic,) = _FakeInputStream.opened
    mic.callback(_HELD, 4, None, None)
    mic_agent.set_talking(False)
    mic.callback(_LATE, 4, None, None)  # the loop has not run since release: the mic is still open
    assert mic.running

    _run_loop(mic_agent)
    assert _queued(mic_agent) == [_HELD, _FORCE_END_TURN]  # held audio kept, ahead of ForceEndTurn


def test_a_press_released_before_the_loop_runs_sends_no_audio(mic_agent: VoiceAgent) -> None:
    mic_agent.set_talking(True)
    mic_agent.set_talking(False)
    _run_loop(mic_agent)  # the loop opens the mic for the press, then closes it for the release
    (mic,) = _FakeInputStream.opened
    mic.callback(_LATE, 4, None, None)
    _run_loop(mic_agent)
    assert not mic.running
    assert _queued(mic_agent) == [_FORCE_END_TURN]


def test_press_release_press_streams_again(mic_agent: VoiceAgent) -> None:
    mic_agent.set_talking(True)
    _run_loop(mic_agent)
    first = _FakeInputStream.opened[0]
    first.callback(_HELD, 4, None, None)
    mic_agent.set_talking(False)
    mic_agent.set_talking(True)  # pressed again before the loop ran
    _run_loop(mic_agent)
    second = _FakeInputStream.opened[-1]
    assert second is not first and second.running and not first.running
    held_again = b"\x03\x00" * 4
    second.callback(held_again, 4, None, None)
    _run_loop(mic_agent)
    assert _queued(mic_agent) == [_HELD, _FORCE_END_TURN, held_again]


def test_an_agent_error_message_ends_the_session_without_touching_policy() -> None:
    handlers, policy = _handlers()
    agent = _offline_agent(handlers, [])
    with pytest.raises(RuntimeError):
        agent._dispatch({"type": "Error", "description": "boom", "code": "X"})
    assert policy.to_dict() == Policy().to_dict()
