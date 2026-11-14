"""WP7: tool handlers, hard limits and the keyword parser. Offline: no network, key, mic or speaker."""

from __future__ import annotations

import json
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


def test_showing_the_face_class_is_refused() -> None:
    handlers, policy = _handlers()
    before = policy.to_dict()
    result = handlers.handle("set_class_rule", {"label": "face", "action": "show"})
    assert result["ok"] is False and result["error"]
    assert policy.to_dict() == before


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
        "show all faces",
        "unblur the faces",
        "blur faces and show plates",
        "hide plates and street signs",
        "hello there",
        "",
    ],
)
def test_keyword_parser_declines(transcript: str) -> None:
    assert parse_keywords(transcript) is None


def test_keyword_parser_never_shows_faces() -> None:
    for verb in ("show", "unblur", "reveal", "unhide"):
        for noun in ("face", "faces", "all faces", "everyone's faces"):
            assert parse_keywords(f"{verb} {noun}") is None


def test_keyword_parser_ignores_non_strings() -> None:
    assert parse_keywords(None) is None  # type: ignore[arg-type]


def test_keyword_matches_run_through_the_handlers() -> None:
    handlers, policy = _handlers()
    name, arguments = parse_keywords("hide license plates")
    assert handlers.handle(name, arguments)["ok"]
    assert policy.snapshot().class_rules["license plate"] == "blur"


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
        "get_scene", "set_class_rule", "set_track", "blur_all", "clear_panic",
    ]
    assert all("endpoint" not in f for f in think["functions"])
    assert "get_scene" in think["prompt"]
    assert agent["speak"]["provider"]["model"].startswith("aura-2")
    for label in KNOWN_CLASSES:
        assert label in think["prompt"]


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
            {"id": "fc_3", "name": "set_class_rule", "arguments": '{"label": "face", "action": "show"}', "client_side": True},
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


def test_an_agent_error_message_ends_the_session_without_touching_policy() -> None:
    handlers, policy = _handlers()
    agent = _offline_agent(handlers, [])
    with pytest.raises(RuntimeError):
        agent._dispatch({"type": "Error", "description": "boom", "code": "X"})
    assert policy.to_dict() == Policy().to_dict()
