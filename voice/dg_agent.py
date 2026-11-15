"""Deepgram Voice Agent socket, Settings, mic gate and playback (WP7).

Mic audio is sent only while Talk is held, or between a wake phrase detected on the device
(voice/wake.py) and the end of that turn. Before the wake phrase, audio is only fed to the
on-device detector in memory: never stored, sent or logged.

Speaks the Voice Agent websocket protocol directly (plain JSON over the
`websockets` package that deepgram-sdk already depends on), because the SDK's
async client passes headers with a keyword the installed websockets no longer takes.
sounddevice is imported only when use_audio=True, so the text-only path works
on a machine with no audio device or no PortAudio at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import config
from contracts import AD_CLASSES, KNOWN_CLASSES, TOOL_SCHEMAS
from voice.confirm import NEEDS_CONFIRMATION, Confirmations
from voice.tools import KEYWORD_REPLIES, NO_AD_IMAGES, ToolHandlers, parse_keywords
from voice.wake import WakeDetector, WakeUnavailable, panic_spotter

# role is "you" or "agent"
OnTranscript = Callable[[str, str], None]
OnMicLive = Callable[[bool], None]

_log = logging.getLogger(__name__)

AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
_KEEPALIVE_S = 5.0
_MIC_BLOCK = config.MIC_SAMPLE_RATE // 20  # 50 ms of audio per send
_TRANSCRIPT_WAIT_S = 10.0  # after release, give up waiting for the user transcript
_RECONNECT_DELAYS_S = (1.0, 2.0, 5.0, 10.0)
_TTS_DEAF_S = 0.3  # the wake detector stays deaf this long after the agent's own audio ends
# Deepgram messages that mean the user's turn is over: a wake turn stops sending here.
_END_OF_USER_TURN = ("AgentThinking", "AgentStartedSpeaking", "EndOfTurn")

_COMMANDS_PROMPT = f"""\
You are the voice control of a handheld privacy camera. You change what it blurs only by
calling tools; you cannot see the video. Everything private starts blurred except the camera
owner; ads ({", ".join(AD_CLASSES)}) start shown.

The KNOWN TYPES are exactly: {", ".join(KNOWN_CLASSES)}.
("plates" means license plate, "storefronts" means storefront sign, "signs" on a street means street sign.)

Pick one path for every command:

A. A whole known type ("all faces", "everyone", "the house numbers", "license plates"):
   call set_class_rule right away, no get_scene. Examples:
   "blur all faces", "blur everyone", "blur people again" -> set_class_rule(label="face", action="blur")
   "unblur everyone", "unblur all faces", "keep everyone unblurred", "people start unblurred"
     -> set_class_rule(label="face", action="show"). That keeps new people unblurred too;
     don't ask about it and don't say new people stay blurred.
   "pixelate the house numbers" -> set_class_rule(label="house number", action="blur")
   "hide all license plates" -> set_class_rule(label="license plate", action="blur")
   "unblur the license plates" -> set_class_rule(label="license plate", action="show")
   "show street signs" -> set_class_rule(label="street sign", action="show")

B. Specific people or objects ("the person on my left", "the guy", "her", "my face",
   "the plate on the right", "the sign behind me", "the two people on my left"), to blur OR show,
   also while a whole type is shown:
   1. Call get_scene first. Use only track ids from that latest scene.
   2. Find the tracks that fit. "my left"/"my right" = owner_left/owner_right. Without an owner,
      left, center, right are the operator's view. "the back", "behind me" = far, "close" = near.
      The scene's "owner" is the user's own face, shown by default.
      "me", "my face" = the owner's track: "blur my face" -> set_track(owner id, "blur").
      "The guy", "her", "that person" with no side or distance fit EVERY face except the owner's.
      "Again" keeps only tracks whose state is the opposite of the action. Example: face A has
      state "show", face B has state "blur"; "blur her again" -> set_track(A, "blur").
   3. Singular ("the guy", "her", "the plate", "the person on my right"): exactly one fit ->
      set_track(its id, action). Two or more fit -> never guess, call no set_track, and your
      whole reply is one question ending in "?", like "Which one, left or right?" or
      "The one on your left or the one in the back?"
      Plural ("the two people on my left", "both plates", "them") -> one set_track per fit.
   4. A short answer to your question ("closer", "the left one", "both") picks from the
      candidates you asked about: act on it with set_track right away, never ask again.
   5. set_track says the id is not in the live scene: call get_scene once more and retry
      once with the new id at the same side and distance. Only then say it failed.

C. Everything at once:
   "Unblur everything", "show everything", "unblur all" -> call show_all, no get_scene.
   It shows every type, people included, and keeps new ones unblurred.
   "Blur everything", "panic", "hide it all" -> call blur_all.
   "Stop panic", "clear panic" -> call clear_panic.
   "Back to normal", "reset" -> call clear_panic, then set_class_rule(label, "blur") for EVERY
   known type, face included (that also re-blurs every person).
   "Everyone" and "all faces" are path A, not this path.

D. A type that is NOT a known type (logos, tattoos, cars, screens): call no tool and say
   "I can hide faces, plates, street signs, house numbers, storefronts."
"""

_RECORDING_PROMPT = """
F. Recording, never ask first: "start recording", "record this" -> start_recording; "stop recording",
   "stop" meaning the recording -> stop_recording. Reply says whether it is recording now.
"""

_REPLY_PROMPT = f"""
A command may start with "hey veil", the wake phrase: it is not part of the command, ignore it.
If a tool returns needs_confirmation, your whole reply is exactly its "say" text. If the user's
next turn is a yes, call that same tool again with the same arguments; anything else, drop it.
If a tool returns ok false otherwise, say briefly that it didn't work.
Reply in plain words, under {config.AGENT_REPLY_MAX_WORDS} words, no ids, no lists, no markdown.
"""


def _ads_prompt(ad_assets: Sequence[str]) -> str:
    """Path E: replacing ads with an image. Ads are not private, so this is not blur or unblur."""
    if not ad_assets:
        return """
E. Replacing ads: there are no ad images on this device. Call no tool and say so.
"""
    return f"""
E. Replacing ads with an image. Ads are not private; this is not blurring. Only the ad types
   can be replaced, only with these ad images (exact file names): {", ".join(ad_assets)}.
   "Replace the ads" with no image named -> set_replacement(label, "{ad_assets[0]}") for EVERY
   ad type, no get_scene.
   "Put the X ad on the posters" -> set_replacement(label="poster", asset=X's file name). No
   image fits X -> call no tool and say which images there are.
   "Stop replacing the ads", "show the real ads" -> set_class_rule(label, "show") for every ad
   type, or only the type named.
   The Ads button means replacement. "Ads on", "turn on / enable / start the ads", "ad mode"
   -> replacement ON: set_replacement(label, "{ad_assets[0]}") for billboard, poster and
   advertisement. "Ads off", "turn off / disable the ads" -> replacement OFF:
   set_class_rule(label, "show") for those three. Never blur for these phrases.
   "Toggle the ads" -> call get_scene; if any ad type's rule is "replace", turn replacement
   off, otherwise on. Reply "Ads replaced." or "Ads back to normal."
   Only an explicit "blur/hide the ads" blurs them (path A): "blur the posters" -> set_class_rule(label="poster", action="blur").
"""


def build_settings(
    *,
    think_model: str | None = None,
    think_provider: str | None = None,
    temperature: float | None = config.DG_THINK_TEMPERATURE,
    speak_model: str | None = None,
    reasoning_mode: str | None = None,
    ad_assets: Sequence[str] = (),
) -> dict[str, Any]:
    """The Voice Agent Settings message: Flux, hosted think model, Aura-2, the 9 tools, the prompt.

    The keyword arguments let scripts/voice_check.py try other hosted models;
    temperature=None leaves it out, for models that only accept their default;
    reasoning_mode is for OpenAI reasoning models (gpt-5 family and later).
    ad_assets are the ad image file names the prompt offers for set_replacement.
    """
    think_provider_settings: dict[str, Any] = {
        "type": think_provider or config.DG_THINK_PROVIDER,
        "model": think_model or config.DG_THINK_MODEL,
    }
    if temperature is not None:
        think_provider_settings["temperature"] = temperature
    if reasoning_mode is not None:
        think_provider_settings["reasoning_mode"] = reasoning_mode
    return {
        "type": "Settings",
        "mip_opt_out": True,  # privacy camera: don't hand bystanders' audio to model training
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": config.MIC_SAMPLE_RATE},
            "output": {"encoding": "linear16", "sample_rate": config.SPEAKER_SAMPLE_RATE, "container": "none"},
        },
        "agent": {
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "version": "v2",  # Flux; ForceEndTurn needs it
                    "model": config.DG_LISTEN_MODEL,
                    "keyterms": list(config.DEEPGRAM_KEYTERMS),
                }
            },
            "think": {
                "provider": think_provider_settings,
                "prompt": _COMMANDS_PROMPT + _ads_prompt(ad_assets) + _RECORDING_PROMPT + _REPLY_PROMPT,
                "functions": [dict(schema) for schema in TOOL_SCHEMAS],
            },
            "speak": {"provider": {"type": "deepgram", "model": speak_model or config.DG_SPEAK_MODEL}},
        },
    }


class _AgentError(RuntimeError):
    """A fatal Error message from Deepgram: end the session and reconnect."""


@dataclass
class _Turn:
    """One user turn, from end of speech (or injection) to the first tool call."""

    ended_at: float
    user_text: str = ""
    tool_called: bool = False


class VoiceAgent(threading.Thread):
    """Owns the asyncio loop, the socket and the audio devices."""

    def __init__(
        self,
        handlers: ToolHandlers,
        on_transcript: OnTranscript,
        use_audio: bool = True,
        *,
        keyword_fallback_s: float | None = config.KEYWORD_FALLBACK_S,
        on_mic_live: OnMicLive | None = None,
        **settings_overrides: Any,
    ) -> None:
        """settings_overrides go to build_settings (think_model, think_provider, temperature,
        speak_model, reasoning_mode).

        keyword_fallback_s=None turns the keyword fallback off (scripts/voice_check.py
        uses that to measure the model alone).
        on_mic_live(bool) is called, from any thread, when mic audio starts or stops leaving
        the device (Talk held or a wake turn open).
        """
        super().__init__(name="voice", daemon=True)
        self._handlers = handlers
        self._on_transcript = on_transcript
        self._use_audio = use_audio
        self._keyword_fallback_s = keyword_fallback_s
        self._settings = build_settings(ad_assets=handlers.ad_assets, **settings_overrides)
        self._api_key = os.environ.get(config.DEEPGRAM_API_KEY_ENV, "")

        self._loop = asyncio.new_event_loop()
        self._outbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self._stopping = asyncio.Event()
        self._ready = threading.Event()  # set while a session is up (after SettingsApplied)
        self._talking = False  # the loop's view: mic open, turn in progress
        # The caller's view, set synchronously by set_talking. The audio callback sends a block
        # only while this is True, so nothing captured after release returns can go out.
        # The lock keeps "check and schedule the block" atomic with "release and schedule
        # ForceEndTurn", so every accepted block reaches the outbox before ForceEndTurn.
        self._talk_held = False
        self._talk_lock = threading.Lock()
        self._mic: Any = None  # sounddevice.RawInputStream while Talk is held
        self._turn: _Turn | None = None
        self._speaker = _Speaker() if use_audio else None

        # Wake phrase. The always-on stream feeds the detector while idle; a wake opens a
        # turn that sends like a Talk hold. All of it is guarded by _talk_lock.
        self._wake: WakeDetector | None = None
        self._wake_mic: Any = None  # the continuous stream, only when the detector loaded
        self._panic: WakeDetector | None = None  # on-device panic words for an offline wake turn
        self._wake_turn_open = False
        self._wake_turn_offline = False
        self._wake_opened_at = 0.0
        self._on_mic_live = on_mic_live
        self._mic_live_shown = False
        # Loop-side: the current user turn began with the wake phrase, so reveals need a yes.
        self._turn_began_with_wake = False
        self._confirmations = Confirmations()

    # --- public, safe from any thread -----------------------------------------

    @property
    def mic_live(self) -> bool:
        """True while mic audio is leaving the device: Talk held or a wake turn open."""
        with self._talk_lock:
            return self._talk_held or self._wake_turn_open

    def set_talking(self, pressed: bool) -> None:
        """Press opens the mic gate; release closes it and sends ForceEndTurn. Safe from any thread.

        The gate closes before this returns, without waiting for the loop: a mic block that
        arrives after release is dropped. Blocks captured while held still go out, ahead of
        ForceEndTurn, so the end of the utterance is not cut. A press during a wake turn takes
        the turn over.
        """
        with self._talk_lock:
            self._talk_held = bool(pressed)
            if pressed:
                self._wake_turn_open = False
            self._call_on_loop(self._set_talking, bool(pressed))
        self._notify_mic_live()

    def inject_text(self, text: str) -> None:
        """Send a typed command as InjectUserMessage (tests and the no-mic path)."""
        self._call_on_loop(self._inject_text, str(text))

    def stop(self) -> None:
        self._call_on_loop(self._shutdown)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the agent session is up. For scripts and the UI status."""
        return self._ready.wait(timeout)

    # --- thread and connection --------------------------------------------------

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            if self._use_audio and config.WAKE_PHRASE_ENABLED:
                self._start_wake_listening()
            self._loop.run_until_complete(self._run_forever())
        finally:
            self._stop_wake_listening()
            self._close_mic()
            if self._speaker:
                self._speaker.close()
            self._loop.close()

    def _call_on_loop(self, callback: Callable[..., None], *args: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:  # the loop is already closed: nothing left to do
            pass

    def _say(self, role: str, text: str) -> None:
        try:
            self._on_transcript(role, text)
        except Exception:  # noqa: BLE001 - a UI bug must not take the voice thread down
            _log.exception("on_transcript failed")

    def _shutdown(self) -> None:
        self._close_mic()
        self._talking = False
        self._stopping.set()

    async def _run_forever(self) -> None:
        """Connect, run a session until it drops, report the error, retry with backoff."""
        failures = 0
        last_error = ""
        while not self._stopping.is_set():
            try:
                await self._session()
                failures = 0
            except Exception as error:  # noqa: BLE001 - any failure: report, keep Policy as is, retry
                message = _short_error(error)
                _log.warning("voice session ended: %s", message)
                if message != last_error:  # don't flood the 3-line transcript while offline
                    self._say("agent", message)
                last_error = message
                failures += 1
            finally:
                self._ready.clear()
                self._close_mic()
                self._talking = False
            if self._stopping.is_set():
                break
            delay = _RECONNECT_DELAYS_S[min(failures, len(_RECONNECT_DELAYS_S) - 1)]
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _session(self) -> None:
        from websockets.asyncio.client import connect  # deepgram-sdk's own dependency

        if not self._api_key:
            raise RuntimeError(f"Voice off: {config.DEEPGRAM_API_KEY_ENV} not set")
        headers = {"Authorization": f"Token {self._api_key}"}
        async with connect(AGENT_URL, additional_headers=headers, max_size=None, open_timeout=10) as socket:
            await socket.send(json.dumps(self._settings))
            _drain(self._outbox)  # nothing from before this session goes out
            tasks = [
                asyncio.create_task(self._receive(socket)),
                asyncio.create_task(self._send(socket)),
                asyncio.create_task(self._keepalive()),
                asyncio.create_task(self._fallback_watch()),
                asyncio.create_task(self._stopping.wait()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()  # re-raise a socket error so _run_forever reports it

    async def _send(self, socket: Any) -> None:
        while True:
            message = await self._outbox.get()
            await socket.send(message)

    async def _keepalive(self) -> None:
        while True:
            await asyncio.sleep(_KEEPALIVE_S)
            if not self._talking:
                self._outbox.put_nowait(json.dumps({"type": "KeepAlive"}))

    # --- incoming messages --------------------------------------------------------

    async def _receive(self, socket: Any) -> None:
        async for raw in socket:
            if isinstance(raw, bytes):
                if self._speaker:
                    self._speaker.play(raw)
                continue
            try:
                self._dispatch(json.loads(raw))
            except _AgentError:
                raise
            except Exception as error:  # noqa: BLE001 - a bad message never touches the Policy
                _log.warning("bad agent message: %s", error)
                self._say("agent", "Voice: bad message from agent")
        raise ConnectionError("Voice: connection closed")

    def _dispatch(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind in _END_OF_USER_TURN:
            self._close_wake_turn()
        if kind == "SettingsApplied":
            self._ready.set()
        elif kind == "FunctionCallRequest":
            self._answer_function_calls(message.get("functions") or [])
        elif kind == "ConversationText":
            self._on_conversation_text(message.get("role"), str(message.get("content", "")))
        elif kind == "UserStartedSpeaking":
            if self._speaker:
                self._speaker.flush()  # barge-in: stop the old reply
        elif kind == "InjectionRefused":
            self._say("agent", "Busy, try again.")
        elif kind == "Error":
            raise _AgentError(f"Voice error: {message.get('description') or message.get('code')}")
        elif kind == "Warning":
            _log.warning("agent warning: %s %s", message.get("code"), message.get("description"))
            if "FAILED" in str(message.get("code", "")):  # e.g. THINK_REQUEST_FAILED: the turn is lost
                self._say("agent", "Voice: the model failed, try again")

    def _answer_function_calls(self, functions: list[dict[str, Any]]) -> None:
        for call in functions:
            if not isinstance(call, dict) or not call.get("client_side", True):
                continue  # server-side: Deepgram answers it itself
            if self._turn:
                self._turn.tool_called = True
            name = str(call.get("name", ""))
            result = self._handle_tool(name, _parse_arguments(call.get("arguments")))
            response = {"type": "FunctionCallResponse", "id": call.get("id"), "name": name, "content": json.dumps(result)}
            self._outbox.put_nowait(json.dumps(response))

    def _on_conversation_text(self, role: Any, content: str) -> None:
        if role == "user":
            if self._turn and self._turn.user_text == content:
                return  # the echo of our own InjectUserMessage, already shown
            self._close_wake_turn()  # the transcript means the user's turn ended
            if self._turn:
                self._turn.user_text = content
            self._say("you", content)
            self._run_confirmed_call(content)
        elif role == "assistant":
            self._say("agent", content)

    # --- talk gate and typed commands (run on the loop) ---------------------------

    def _set_talking(self, pressed: bool) -> None:
        if not self._use_audio or pressed == self._talking:
            return
        if pressed:
            if not self._ready.is_set():
                self._say("agent", "Voice offline")
                return
            self._talking = True
            self._turn = None
            self._turn_began_with_wake = False
            if self._speaker:
                self._speaker.flush()
            if self._wake_mic is not None:
                return  # the wake stream is already running; the gate in _on_mic_block sends
            try:
                self._mic = _open_input_stream(self._on_mic_block)
            except Exception as error:  # noqa: BLE001 - no mic: say so, stay closed
                self._talking = False
                self._say("agent", f"Mic error: {type(error).__name__}")
            return
        self._talking = False
        self._close_mic()
        # Every block the gate accepted was scheduled before this call (see set_talking),
        # so the last audio is already in the outbox, ahead of ForceEndTurn.
        self._end_turn()

    def _end_turn(self) -> None:
        self._outbox.put_nowait(json.dumps({"type": "ForceEndTurn"}))
        self._turn = _Turn(ended_at=time.monotonic())

    def _inject_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._say("you", text)
        self._turn = _Turn(ended_at=time.monotonic(), user_text=text)
        self._turn_began_with_wake = False
        self._run_confirmed_call(text)
        if not self._ready.is_set():
            self._run_keyword_fallback()  # offline: class and panic commands still work
            return
        self._outbox.put_nowait(json.dumps({"type": "InjectUserMessage", "content": text}))

    def _close_mic(self) -> None:
        mic, self._mic = self._mic, None
        _close_stream(mic)

    # --- mic gate and wake phrase -------------------------------------------------------

    def _on_mic_block(self, block: bytes, now: float) -> None:
        """Every mic block, on PortAudio's thread (tests call it directly).

        Sends only while Talk is held or a wake turn is open. Otherwise the block goes to the
        on-device wake detector and is dropped: never stored, sent or logged.
        """
        with self._talk_lock:  # the gate; see set_talking
            if self._talk_held:
                self._call_on_loop(self._outbox.put_nowait, block)
            elif self._wake_turn_open:
                self._wake_turn_block(block, now)
            elif self._wake is not None:
                self._listen_for_wake(block, now)
        self._notify_mic_live()

    def _wake_turn_block(self, block: bytes, now: float) -> None:
        """Holds _talk_lock. Sends the block, or spots panic words on the device when offline."""
        if now - self._wake_opened_at >= config.WAKE_TURN_SILENCE_S:
            self._wake_turn_open = False
            self._call_on_loop(self._wake_turn_timed_out, self._wake_turn_offline)
        elif not self._wake_turn_offline:
            self._call_on_loop(self._outbox.put_nowait, block)
        elif self._panic is not None and self._panic.feed(block):
            self._wake_turn_open = False
            self._call_on_loop(self._offline_panic)

    def _listen_for_wake(self, block: bytes, now: float) -> None:
        """Holds _talk_lock. The agent's own voice must not wake it: deaf while it plays."""
        assert self._wake is not None
        if self._speaker and self._speaker.busy(now, _TTS_DEAF_S):
            self._wake.reset()
            return
        if not self._wake.feed(block):
            return
        # The block that completed the phrase is not sent: only blocks after it.
        self._wake_turn_open = True
        self._wake_opened_at = now
        self._wake_turn_offline = not self._ready.is_set()
        if self._wake_turn_offline and self._panic is not None:
            self._panic.reset()
        self._call_on_loop(self._on_wake, self._wake_turn_offline)

    def _on_wake(self, offline: bool) -> None:
        self._turn = None
        self._turn_began_with_wake = True
        if self._speaker:
            self._speaker.flush()
            self._speaker.play(_wake_tone())
        if offline:
            self._say("agent", "Voice offline: only panic works")

    def _close_wake_turn(self) -> None:
        """Deepgram says the user's turn is over: stop sending. No ForceEndTurn needed."""
        with self._talk_lock:
            was_open, self._wake_turn_open = self._wake_turn_open, False
        if was_open:
            self._turn = _Turn(ended_at=time.monotonic())
            self._notify_mic_live()

    def _wake_turn_timed_out(self, offline: bool) -> None:
        if offline:
            self._say("agent", "Voice offline")
        else:
            self._end_turn()  # the blocks already queued go out ahead of it

    def _offline_panic(self) -> None:
        result = self._handlers.handle("blur_all", {})
        _log.info("offline wake panic -> %s", result)
        self._say("agent", "Done." if result.get("ok") else "Couldn't do that.")

    def _notify_mic_live(self) -> None:
        with self._talk_lock:
            live = self._talk_held or self._wake_turn_open
            changed, self._mic_live_shown = live != self._mic_live_shown, live
        if changed and self._on_mic_live is not None:
            try:
                self._on_mic_live(live)
            except Exception:  # noqa: BLE001 - a UI bug must not take the mic thread down
                _log.exception("on_mic_live failed")

    def _start_wake_listening(self) -> None:
        """Load the detector and open the always-on stream. Any failure: no listening, Talk works."""
        try:
            self._wake = WakeDetector()
            self._panic = panic_spotter()
            self._wake_mic = _open_input_stream(self._on_mic_block)
        except Exception as error:  # noqa: BLE001 - WakeUnavailable, no mic: fail closed
            self._wake = None
            self._panic = None
            message = str(error) if isinstance(error, WakeUnavailable) else f"Wake phrase off ({type(error).__name__})"
            _log.warning("%s", message)
            self._say("agent", message)

    def _stop_wake_listening(self) -> None:
        stream, self._wake_mic = self._wake_mic, None
        _close_stream(stream)
        with self._talk_lock:
            self._wake = None
            self._wake_turn_open = False

    # --- confirmation of reveals after a wake -------------------------------------------

    def _handle_tool(self, name: str, arguments: Any) -> dict[str, Any]:
        if self._confirmations.needs_confirmation(name, arguments, self._turn_began_with_wake):
            self._confirmations.hold(name, arguments, time.monotonic())
            return NEEDS_CONFIRMATION
        return self._handlers.handle(name, arguments)

    def _run_confirmed_call(self, transcript: str) -> None:
        """A user turn after a held reveal: run it if the turn is a yes, else drop it."""
        call = self._confirmations.confirm(transcript, time.monotonic())
        if call is None:
            return
        if self._turn:
            self._turn.tool_called = True
        result = self._handlers.handle(*call)
        _log.info("confirmed %s -> %s", call[0], result)

    # --- keyword fallback -------------------------------------------------------------

    async def _fallback_watch(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            turn = self._turn
            if turn is None or turn.tool_called:
                continue
            waited = time.monotonic() - turn.ended_at
            fallback_s = self._keyword_fallback_s
            if fallback_s is not None and waited >= fallback_s and turn.user_text:
                self._run_keyword_fallback()
            elif waited >= _TRANSCRIPT_WAIT_S:
                self._turn = None  # no transcript ever came: nothing to act on

    def _run_keyword_fallback(self) -> None:
        turn, self._turn = self._turn, None
        calls = parse_keywords(turn.user_text, self._handlers.ad_assets) if turn else []
        if not calls:
            if not self._ready.is_set():
                self._say("agent", "Voice offline")
            return
        errors = []
        for name, arguments in calls:
            result = self._handle_tool(name, arguments)  # a reveal after a wake waits for a yes
            _log.info("keyword fallback %s %s -> %s", name, arguments, result)
            if not result.get("ok"):
                errors.append(result.get("error"))
        if not errors:
            self._say("agent", KEYWORD_REPLIES.get(calls[-1][0], "Done."))
        elif self._confirmations.pending is not None:
            self._say("agent", NEEDS_CONFIRMATION["say"])
        else:
            self._say("agent", NO_AD_IMAGES if NO_AD_IMAGES in errors else "Couldn't do that.")


class _Speaker:
    """Plays agent audio on its own thread, so a blocking write never stalls the socket."""

    def __init__(self) -> None:
        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="voice-speaker", daemon=True)
        self._thread.start()
        self._playing_until = 0.0  # monotonic time the queued audio should finish playing

    def play(self, chunk: bytes) -> None:
        now = time.monotonic()
        self._playing_until = max(now, self._playing_until) + len(chunk) / (2 * config.SPEAKER_SAMPLE_RATE)
        self._chunks.put(chunk)

    def busy(self, now: float, tail_s: float) -> bool:
        """True while agent audio plays and for tail_s after (the wake detector stays deaf)."""
        return now < self._playing_until + tail_s

    def flush(self) -> None:
        _drain(self._chunks)
        self._playing_until = min(self._playing_until, time.monotonic())

    def close(self) -> None:
        self.flush()
        self._chunks.put(None)

    def _run(self) -> None:
        stream = None
        leftover = b""  # int16 frames are 2 bytes; a chunk may split one
        try:
            while (chunk := self._chunks.get()) is not None:
                if stream is None:
                    import sounddevice

                    stream = sounddevice.RawOutputStream(
                        samplerate=config.SPEAKER_SAMPLE_RATE, channels=1, dtype="int16"
                    )
                    stream.start()
                data = leftover + chunk
                whole = len(data) - len(data) % 2
                leftover = data[whole:]
                if whole:
                    stream.write(data[:whole])
        except Exception as error:  # noqa: BLE001 - no speaker: replies stay text-only
            _log.warning("speaker failed, text-only replies: %s", error)
            while self._chunks.get() is not None:
                pass
        finally:
            if stream is not None:
                stream.close()


def _open_input_stream(on_block: Callable[[bytes, float], None]) -> Any:
    """A started 16 kHz mono int16 mic stream, 50 ms blocks, each passed to on_block(block, now)."""
    import sounddevice  # only here: no audio device is needed for the text path

    def on_audio(data: Any, frames: int, time_info: Any, status: Any) -> None:
        on_block(bytes(data), time.monotonic())

    stream = sounddevice.RawInputStream(
        samplerate=config.MIC_SAMPLE_RATE, channels=1, dtype="int16", blocksize=_MIC_BLOCK, callback=on_audio
    )
    stream.start()
    return stream


def _close_stream(stream: Any) -> None:
    if stream is None:
        return
    try:
        stream.stop()
        stream.close()
    except Exception as error:  # noqa: BLE001
        _log.warning("closing the mic failed: %s", error)


def _wake_tone() -> bytes:
    """A short sine beep at the speaker rate, int16, with 10 ms fades so it doesn't click."""
    rate = config.SPEAKER_SAMPLE_RATE
    count = int(rate * config.WAKE_TONE_S)
    fade = max(1, rate // 100)
    samples = bytearray()
    for i in range(count):
        envelope = min(1.0, i / fade, (count - 1 - i) / fade)
        value = int(0.3 * 32767 * envelope * math.sin(2 * math.pi * config.WAKE_TONE_HZ * i / rate))
        samples += value.to_bytes(2, "little", signed=True)
    return bytes(samples)


def _parse_arguments(arguments: Any) -> Any:
    """Deepgram sends arguments as a JSON string; ToolHandlers refuses anything that isn't an object."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return arguments
    return arguments


def _drain(pending: asyncio.Queue[Any] | queue.Queue[Any]) -> None:
    while True:
        try:
            pending.get_nowait()
        except (asyncio.QueueEmpty, queue.Empty):
            return


def _short_error(error: Exception) -> str:
    text = str(error)
    if text.startswith(("Voice", "Mic")):
        return text[:80]
    return f"Voice offline ({type(error).__name__})"
