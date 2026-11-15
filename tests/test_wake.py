"""WP29: wake phrase, wake turns, the TTS deaf window and spoken confirmation. No mic, network or speaker."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest

import config
from contracts import FACE, Policy
from fakes import scenes
from voice import dg_agent
from voice.confirm import Confirmations
from voice.dg_agent import VoiceAgent
from voice.tools import ToolHandlers

_FORCE_END_TURN = json.dumps({"type": "ForceEndTurn"})
_BLOCK = b"\x01\x00" * 800  # 50 ms at 16 kHz


class _FakeDetector:
    """Fires on the block equal to `wake_block`; counts feeds and resets."""

    def __init__(self, wake_block: bytes = b"WAKE") -> None:
        self.wake_block = wake_block
        self.fed: list[bytes] = []
        self.resets = 0

    def feed(self, block: bytes) -> bool:
        self.fed.append(block)
        return block == self.wake_block

    def reset(self) -> None:
        self.resets += 1


class _FakeSpeaker:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.until = 0.0

    def play(self, chunk: bytes) -> None:
        self.played.append(chunk)

    def flush(self) -> None:
        pass

    def busy(self, now: float, tail_s: float) -> bool:
        return now < self.until + tail_s


class _FakeInputStream:
    opened: list["_FakeInputStream"] = []

    def __init__(self, *, samplerate, channels, dtype, blocksize, callback) -> None:
        self.callback = callback
        self.running = False
        _FakeInputStream.opened.append(self)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.running = False


def _handlers() -> tuple[ToolHandlers, Policy]:
    policy = Policy()
    return ToolHandlers(policy, lambda: scenes.OWNER_AND_FRIEND_LEFT), policy


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch):
    """A ready agent with a fake detector, fake speaker and fake mic; the test runs the loop."""
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=_FakeInputStream))
    _FakeInputStream.opened.clear()
    handlers, policy = _handlers()
    lives: list[bool] = []
    voice = VoiceAgent(handlers, lambda role, text: None, use_audio=True, on_mic_live=lives.append)
    voice._speaker = _FakeSpeaker()
    voice._wake = _FakeDetector()
    voice._panic = _FakeDetector(b"PANIC")
    voice._ready.set()
    voice.policy, voice.lives = policy, lives  # handy for the tests
    yield voice
    voice._loop.close()


def _run_loop(voice: VoiceAgent) -> None:
    voice._loop.run_until_complete(asyncio.sleep(0))


def _queued(voice: VoiceAgent) -> list:
    out = []
    while not voice._outbox.empty():
        out.append(voice._outbox.get_nowait())
    return out


def _wake(voice: VoiceAgent, now: float = 100.0) -> None:
    voice._on_mic_block(b"WAKE", now)
    _run_loop(voice)


# --- the gate ---


def test_nothing_is_queued_before_the_wake_and_the_wake_block_is_not_sent(agent: VoiceAgent) -> None:
    agent._on_mic_block(_BLOCK, 99.0)
    _run_loop(agent)
    assert _queued(agent) == [] and not agent.mic_live
    assert agent._wake.fed == [_BLOCK]

    _wake(agent)
    assert agent.mic_live and agent.lives == [True]
    assert agent._speaker.played  # the tone
    after = b"\x05\x00" * 800
    agent._on_mic_block(after, 100.05)
    _run_loop(agent)
    assert _queued(agent) == [after]  # not b"WAKE"
    assert agent._wake.fed == [_BLOCK, b"WAKE"]  # blocks in a wake turn don't reach the detector


def test_a_wake_turn_stops_sending_when_deepgram_ends_the_turn(agent: VoiceAgent) -> None:
    _wake(agent)
    agent._on_mic_block(_BLOCK, 100.1)
    _run_loop(agent)
    agent._dispatch({"type": "ConversationText", "role": "user", "content": "blur all faces"})
    agent._on_mic_block(b"late", 100.2)
    _run_loop(agent)
    assert _queued(agent) == [_BLOCK]  # Deepgram ended it: no ForceEndTurn, nothing after
    assert not agent.mic_live and agent.lives == [True, False]
    assert agent._turn is not None and agent._turn.user_text == "blur all faces"


@pytest.mark.parametrize("kind", ["AgentThinking", "AgentStartedSpeaking"])
def test_agent_messages_end_a_wake_turn(agent: VoiceAgent, kind: str) -> None:
    _wake(agent)
    agent._dispatch({"type": kind})
    agent._on_mic_block(_BLOCK, 100.1)
    _run_loop(agent)
    assert _queued(agent) == []


def test_a_wake_turn_times_out_with_force_end_turn(agent: VoiceAgent) -> None:
    _wake(agent, now=100.0)
    agent._on_mic_block(_BLOCK, 100.0 + config.WAKE_TURN_SILENCE_S - 0.05)
    agent._on_mic_block(b"late", 100.0 + config.WAKE_TURN_SILENCE_S)
    agent._on_mic_block(b"later", 100.0 + config.WAKE_TURN_SILENCE_S + 0.05)
    _run_loop(agent)
    assert _queued(agent) == [_BLOCK, _FORCE_END_TURN]
    assert not agent.mic_live
    assert agent._wake.fed[-1] == b"later"  # back to listening for the wake phrase


def test_the_detector_is_deaf_while_the_agent_speaks(agent: VoiceAgent) -> None:
    agent._speaker.until = 100.0
    agent._on_mic_block(b"WAKE", 100.2)  # within the 0.3 s tail
    _run_loop(agent)
    assert agent._wake.fed == [] and agent._wake.resets == 1
    assert not agent.mic_live and _queued(agent) == []
    agent._on_mic_block(b"WAKE", 100.4)
    assert agent.mic_live


def test_talk_takes_over_a_wake_turn_and_release_stops_sending(agent: VoiceAgent) -> None:
    _wake(agent)
    agent.set_talking(True)
    _run_loop(agent)
    agent._on_mic_block(_BLOCK, 100.1)
    agent.set_talking(False)
    agent._on_mic_block(b"late", 100.2)
    _run_loop(agent)
    assert _queued(agent) == [_BLOCK, _FORCE_END_TURN]
    assert not agent.mic_live
    assert agent._turn_began_with_wake is False


def test_talk_uses_the_running_wake_stream_instead_of_a_second_one(agent: VoiceAgent) -> None:
    agent._wake_mic = object()
    agent.set_talking(True)
    _run_loop(agent)
    assert _FakeInputStream.opened == []
    agent._on_mic_block(_BLOCK, 1.0)
    agent.set_talking(False)
    _run_loop(agent)
    assert _queued(agent) == [_BLOCK, _FORCE_END_TURN]
    assert agent._wake.fed == []  # held blocks never reach the detector


def test_wake_disabled_opens_no_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=_FakeInputStream))
    monkeypatch.setattr(config, "WAKE_PHRASE_ENABLED", False)
    monkeypatch.delenv(config.DEEPGRAM_API_KEY_ENV, raising=False)
    _FakeInputStream.opened.clear()
    handlers, _ = _handlers()
    voice = VoiceAgent(handlers, lambda role, text: None, use_audio=True)
    voice._speaker = None
    voice.start()
    voice.stop()
    voice.join(timeout=3.0)
    assert _FakeInputStream.opened == [] and voice._wake is None


def test_a_missing_model_means_no_listening_not_a_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(RawInputStream=_FakeInputStream))
    from voice import wake

    with pytest.raises(wake.WakeUnavailable):
        wake.WakeDetector(model_path=tmp_path / "missing")

    def missing():
        raise wake.WakeUnavailable("Wake phrase off: no Vosk model")

    monkeypatch.setattr(dg_agent, "WakeDetector", missing)
    _FakeInputStream.opened.clear()
    lines: list = []
    handlers, _ = _handlers()
    voice = VoiceAgent(handlers, lambda role, text: lines.append(text), use_audio=True)
    voice._speaker = None
    voice._start_wake_listening()
    assert voice._wake is None and _FakeInputStream.opened == []
    assert lines == ["Wake phrase off: no Vosk model"]
    voice._loop.close()


def test_offline_wake_then_panic_turns_panic_on_without_the_network(agent: VoiceAgent) -> None:
    agent._ready.clear()
    _wake(agent)
    agent._on_mic_block(_BLOCK, 100.1)
    agent._on_mic_block(b"PANIC", 100.2)
    _run_loop(agent)
    assert agent.policy.snapshot().panic
    assert _queued(agent) == [] and not agent.mic_live


# --- confirmation ---


def _call(agent: VoiceAgent, name: str, arguments: dict, call_id: str = "fc_1") -> dict:
    agent._dispatch({"type": "FunctionCallRequest", "functions": [
        {"id": call_id, "name": name, "arguments": json.dumps(arguments), "client_side": True}]})
    return json.loads(json.loads(_queued(agent)[-1])["content"])


def _user_says(agent: VoiceAgent, text: str) -> None:
    agent._dispatch({"type": "ConversationText", "role": "user", "content": text})


def test_show_all_after_a_wake_waits_for_a_yes(agent: VoiceAgent) -> None:
    _wake(agent)
    result = _call(agent, "show_all", {})
    assert result == {"ok": False, "needs_confirmation": True, "say": "Say yes to confirm."}
    assert agent.policy.snapshot().class_rules[FACE] == "blur"
    _wake(agent, now=105.0)
    _user_says(agent, "yes")
    assert agent.policy.snapshot().class_rules[FACE] == "show"


def test_the_model_reissuing_the_call_after_yes_runs_it(agent: VoiceAgent) -> None:
    _wake(agent)
    _call(agent, "set_class_rule", {"label": FACE, "action": "show"})
    _wake(agent, now=105.0)
    _user_says(agent, "yeah do it")
    assert _call(agent, "set_class_rule", {"label": FACE, "action": "show"}, "fc_2")["ok"]


def test_anything_but_yes_drops_the_held_reveal(agent: VoiceAgent) -> None:
    _wake(agent)
    _call(agent, "show_all", {})
    _user_says(agent, "no, never mind")
    _user_says(agent, "yes")
    assert agent.policy.snapshot().class_rules[FACE] == "blur"


def test_a_talk_turn_reveals_directly(agent: VoiceAgent) -> None:
    _wake(agent)
    agent.set_talking(True)
    agent.set_talking(False)
    _run_loop(agent)
    assert _call(agent, "show_all", {})["ok"]


@pytest.mark.parametrize("name, arguments", [
    ("blur_all", {}),
    ("set_class_rule", {"label": FACE, "action": "blur"}),
    ("set_class_rule", {"label": "license plate", "action": "show"}),
    ("set_track", {"track_id": "face#7", "action": "show"}),
])
def test_blur_panic_and_others_never_wait(agent: VoiceAgent, name: str, arguments: dict) -> None:
    _wake(agent)
    assert _call(agent, name, arguments)["ok"]


def test_confirmations_pure() -> None:
    confirmations = Confirmations(window_s=15.0)
    assert not confirmations.needs_confirmation("show_all", {}, began_with_wake=False)
    assert confirmations.needs_confirmation("show_all", {}, began_with_wake=True)
    assert not confirmations.needs_confirmation("blur_all", {}, began_with_wake=True)
    confirmations.hold("show_all", {}, now=0.0)
    assert confirmations.confirm("yes", now=16.0) is None  # expired
    assert confirmations.pending is None
    confirmations.hold("show_all", {}, now=20.0)
    assert confirmations.confirm("Yes.", now=21.0) == ("show_all", {})
    assert not confirmations.needs_confirmation("show_all", {}, began_with_wake=True)
    assert confirmations.confirm("something else", now=22.0) is None
    assert confirmations.needs_confirmation("show_all", {}, began_with_wake=True)


# --- the real detector on generated speech ---


def _say_wav(tmp_path: Path, text: str, voice: str) -> bytes:
    aiff, wav_path = tmp_path / "t.aiff", tmp_path / "t.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav_path)], check=True)
    with wave.open(str(wav_path)) as w:
        return w.readframes(w.getnframes()) + b"\0" * 32000


def _fires(detector, audio: bytes) -> int:
    return sum(detector.feed(audio[i:i + 1600]) for i in range(0, len(audio), 1600))


@pytest.mark.skipif(
    not (shutil.which("say") and shutil.which("afconvert") and config.VOSK_MODEL.is_dir()),
    reason="needs macOS say/afconvert and the Vosk model",
)
def test_real_detector_fires_once_on_the_phrase_and_not_on_commands(tmp_path: Path) -> None:
    pytest.importorskip("vosk")
    from voice.wake import WakeDetector, panic_spotter

    detector = WakeDetector()
    assert _fires(detector, _say_wav(tmp_path, "hey veil, unblur everyone", "Samantha")) == 1
    for text in ("unblur everyone", "blur everything", "panic", "replace the ads", "stop recording"):
        detector.reset()
        assert _fires(detector, _say_wav(tmp_path, text, "Samantha")) == 0, text
    assert _fires(panic_spotter(), _say_wav(tmp_path, "panic", "Samantha")) >= 1
