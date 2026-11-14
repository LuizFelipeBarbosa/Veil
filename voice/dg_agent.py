"""Deepgram Voice Agent socket, Settings, mic gate and playback (WP7).

Mic audio is sent only while Talk is held.

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
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import config
from contracts import KNOWN_CLASSES, TOOL_SCHEMAS
from voice.tools import ToolHandlers, parse_keywords

# role is "you" or "agent"
OnTranscript = Callable[[str, str], None]

_log = logging.getLogger(__name__)

AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
_KEEPALIVE_S = 5.0
_MIC_BLOCK = config.MIC_SAMPLE_RATE // 20  # 50 ms of audio per send
_TRANSCRIPT_WAIT_S = 10.0  # after release, give up waiting for the user transcript
_RECONNECT_DELAYS_S = (1.0, 2.0, 5.0, 10.0)

_PROMPT = f"""\
You are the voice control of a handheld privacy camera. You change what it blurs only by
calling tools; you cannot see the video. Everything starts blurred except the camera owner.

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

If a tool returns ok false, say briefly that it didn't work.
Reply in plain words, under {config.AGENT_REPLY_MAX_WORDS} words, no ids, no lists, no markdown.
"""


def build_settings(
    *,
    think_model: str | None = None,
    think_provider: str | None = None,
    temperature: float | None = config.DG_THINK_TEMPERATURE,
    speak_model: str | None = None,
    reasoning_mode: str | None = None,
) -> dict[str, Any]:
    """The Voice Agent Settings message: Flux, hosted think model, Aura-2, the 6 tools, the prompt.

    The keyword arguments let scripts/voice_check.py try other hosted models;
    temperature=None leaves it out, for models that only accept their default;
    reasoning_mode is for OpenAI reasoning models (gpt-5 family and later).
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
                "prompt": _PROMPT,
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
        **settings_overrides: Any,
    ) -> None:
        """settings_overrides go to build_settings (think_model, think_provider, temperature,
        speak_model, reasoning_mode).

        keyword_fallback_s=None turns the keyword fallback off (scripts/voice_check.py
        uses that to measure the model alone).
        """
        super().__init__(name="voice", daemon=True)
        self._handlers = handlers
        self._on_transcript = on_transcript
        self._use_audio = use_audio
        self._keyword_fallback_s = keyword_fallback_s
        self._settings = build_settings(**settings_overrides)
        self._api_key = os.environ.get(config.DEEPGRAM_API_KEY_ENV, "")

        self._loop = asyncio.new_event_loop()
        self._outbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self._stopping = asyncio.Event()
        self._ready = threading.Event()  # set while a session is up (after SettingsApplied)
        self._talking = False
        self._mic: Any = None  # sounddevice.RawInputStream while Talk is held
        self._turn: _Turn | None = None
        self._speaker = _Speaker() if use_audio else None

    # --- public, safe from any thread -----------------------------------------

    def set_talking(self, pressed: bool) -> None:
        """Press opens the mic gate; release closes it and sends ForceEndTurn. Safe from any thread."""
        self._call_on_loop(self._set_talking, bool(pressed))

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
            self._loop.run_until_complete(self._run_forever())
        finally:
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
            result = self._handlers.handle(name, _parse_arguments(call.get("arguments")))
            response = {"type": "FunctionCallResponse", "id": call.get("id"), "name": name, "content": json.dumps(result)}
            self._outbox.put_nowait(json.dumps(response))

    def _on_conversation_text(self, role: Any, content: str) -> None:
        if role == "user":
            if self._turn and self._turn.user_text == content:
                return  # the echo of our own InjectUserMessage, already shown
            if self._turn:
                self._turn.user_text = content
            self._say("you", content)
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
            if self._speaker:
                self._speaker.flush()
            try:
                self._open_mic()
            except Exception as error:  # noqa: BLE001 - no mic: say so, stay closed
                self._talking = False
                self._say("agent", f"Mic error: {type(error).__name__}")
            return
        self._talking = False
        self._close_mic()
        # call_soon queues behind the mic callbacks already scheduled, so the last audio
        # still goes out before ForceEndTurn.
        self._loop.call_soon(self._end_turn)

    def _end_turn(self) -> None:
        self._outbox.put_nowait(json.dumps({"type": "ForceEndTurn"}))
        self._turn = _Turn(ended_at=time.monotonic())

    def _inject_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._say("you", text)
        self._turn = _Turn(ended_at=time.monotonic(), user_text=text)
        if not self._ready.is_set():
            self._run_keyword_fallback()  # offline: class and panic commands still work
            return
        self._outbox.put_nowait(json.dumps({"type": "InjectUserMessage", "content": text}))

    def _open_mic(self) -> None:
        import sounddevice  # only here: no audio device is needed for the text path

        def on_audio(data: Any, frames: int, time_info: Any, status: Any) -> None:
            if self._talking:  # the gate, checked again on PortAudio's thread
                self._call_on_loop(self._outbox.put_nowait, bytes(data))

        self._mic = sounddevice.RawInputStream(
            samplerate=config.MIC_SAMPLE_RATE, channels=1, dtype="int16", blocksize=_MIC_BLOCK, callback=on_audio
        )
        self._mic.start()

    def _close_mic(self) -> None:
        mic, self._mic = self._mic, None
        if mic is None:
            return
        try:
            mic.stop()
            mic.close()
        except Exception as error:  # noqa: BLE001
            _log.warning("closing the mic failed: %s", error)

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
        match = parse_keywords(turn.user_text) if turn else None
        if match is None:
            if not self._ready.is_set():
                self._say("agent", "Voice offline")
            return
        name, arguments = match
        result = self._handlers.handle(name, arguments)
        _log.info("keyword fallback %s %s -> %s", name, arguments, result)
        self._say("agent", "Done." if result.get("ok") else "Couldn't do that.")


class _Speaker:
    """Plays agent audio on its own thread, so a blocking write never stalls the socket."""

    def __init__(self) -> None:
        self._chunks: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="voice-speaker", daemon=True)
        self._thread.start()

    def play(self, chunk: bytes) -> None:
        self._chunks.put(chunk)

    def flush(self) -> None:
        _drain(self._chunks)

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
