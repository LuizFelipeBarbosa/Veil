"""WP7: tool handlers, hard limits and the keyword parser. Offline: no network, key, mic or speaker."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest

import config
from contracts import FACE, KNOWN_CLASSES, OBJECT_CLASSES, Policy, Scene
from fakes import scenes
from voice.dg_agent import VoiceAgent, build_settings
from voice.tools import ToolHandlers, parse_keywords


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
    assert parse_keywords(transcript) == expected


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
    assert parse_keywords(transcript) is None


def test_keyword_parser_never_turns_a_face_command_into_show_all() -> None:
    for verb in ("show", "unblur", "reveal", "unhide"):
        for noun in ("face", "faces", "all faces", "everyone", "all the faces"):
            assert parse_keywords(f"{verb} {noun}") == ("set_class_rule", {"label": FACE, "action": "show"})


def test_keyword_parser_never_shows_one_person() -> None:
    for verb in ("show", "unblur", "reveal"):
        for who in ("her", "him", "the guy", "the person on my left", "face#7", "my face"):
            assert parse_keywords(f"{verb} {who}") is None


def test_keyword_parser_ignores_non_strings() -> None:
    assert parse_keywords(None) is None  # type: ignore[arg-type]


def test_keyword_matches_run_through_the_handlers() -> None:
    handlers, policy = _handlers()
    name, arguments = parse_keywords("hide license plates")
    assert handlers.handle(name, arguments)["ok"]
    assert policy.snapshot().class_rules["license plate"] == "blur"


def test_keyword_show_all_runs_through_the_handlers() -> None:
    handlers, policy = _handlers(scenes.TWO_CANDIDATES)
    name, arguments = parse_keywords("unblur everything")
    assert handlers.handle(name, arguments)["ok"]
    assert all(policy.snapshot().class_rules[label] == "show" for label in KNOWN_CLASSES)


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
    assert [f["name"] for f in think["functions"]] == [
        "get_scene", "set_class_rule", "set_track", "blur_all", "clear_panic", "show_all",
    ]
    assert all("endpoint" not in f for f in think["functions"])
    assert "get_scene" in think["prompt"]
    assert agent["speak"]["provider"]["model"].startswith("aura-2")
    for label in KNOWN_CLASSES:
        assert label in think["prompt"]
    assert "show_all" in think["prompt"]
    assert f"under {config.AGENT_REPLY_MAX_WORDS} words" in think["prompt"]


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


def test_talk_gate_streams_only_while_held_then_force_end_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=_FakeInputStream))
    _FakeInputStream.opened.clear()
    handlers, _ = _handlers()
    agent = VoiceAgent(handlers, lambda role, text: None, use_audio=True)
    agent._speaker = None  # no playback thread work in this test
    agent._ready.set()
    run_loop = lambda: agent._loop.run_until_complete(asyncio.sleep(0))  # noqa: E731

    agent._set_talking(True)
    (mic,) = _FakeInputStream.opened
    assert mic.running
    mic.callback(b"\x01\x00" * 4, 4, None, None)
    agent._set_talking(False)  # closes the mic; the audio scheduled above still goes first
    mic.callback(b"\x02\x00" * 4, 4, None, None)  # a late callback after release is dropped
    run_loop()

    queued = []
    while not agent._outbox.empty():
        queued.append(agent._outbox.get_nowait())
    assert not mic.running and agent._mic is None
    assert queued == [b"\x01\x00" * 4, json.dumps({"type": "ForceEndTurn"})]
    agent._loop.close()


def test_an_agent_error_message_ends_the_session_without_touching_policy() -> None:
    handlers, policy = _handlers()
    agent = _offline_agent(handlers, [])
    with pytest.raises(RuntimeError):
        agent._dispatch({"type": "Error", "description": "boom", "code": "X"})
    assert policy.to_dict() == Policy().to_dict()
