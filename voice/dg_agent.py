"""Deepgram Voice Agent socket, Settings, mic gate and playback (WP7).

Mic audio is sent only while Talk is held.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from voice.tools import ToolHandlers

# role is "you" or "agent"
OnTranscript = Callable[[str, str], None]


def build_settings() -> dict[str, Any]:
    """The Voice Agent Settings message: Flux, hosted think model, Aura-2, the 5 tools, the prompt."""
    raise NotImplementedError("WP7")


class VoiceAgent(threading.Thread):
    """Owns the asyncio loop, the socket and the audio devices."""

    def __init__(self, handlers: ToolHandlers, on_transcript: OnTranscript, use_audio: bool = True) -> None:
        raise NotImplementedError("WP7")

    def set_talking(self, pressed: bool) -> None:
        """Press opens the mic gate; release closes it and sends ForceEndTurn. Safe from any thread."""
        raise NotImplementedError("WP7")

    def inject_text(self, text: str) -> None:
        """Send a typed command as InjectUserMessage (tests and the no-mic path)."""
        raise NotImplementedError("WP7")

    def stop(self) -> None:
        raise NotImplementedError("WP7")
