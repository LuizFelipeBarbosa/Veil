"""Spoken confirmation for reveals that follow a wake phrase.

A false wake must not reveal anything: after a wake, show_all and "show every face" are held
until the next user turn is a spoken yes. Talk-button turns never need it.
config.WAKE_REVEAL_NEEDS_YES turns the whole check off: then nothing is ever held.
"""

from __future__ import annotations

import re
from typing import Any

import config
from contracts import FACE

CONFIRM_WINDOW_S = 15.0
NEEDS_CONFIRMATION = {"ok": False, "needs_confirmation": True, "say": "Say yes to confirm."}
_YES = re.compile(r"\b(yes|yeah|yep|confirm|confirmed|do it)\b")

Call = tuple[str, dict[str, Any]]


def _reveals(name: str, args: Any) -> bool:
    if name == "show_all":
        return True
    return (
        name == "set_class_rule"
        and isinstance(args, dict)
        and args.get("label") == FACE
        and args.get("action") == "show"
    )


class Confirmations:
    def __init__(self, window_s: float = CONFIRM_WINDOW_S) -> None:
        self._window_s = window_s
        self.pending: Call | None = None
        self._held_at = 0.0
        self._approved: Call | None = None  # the call the user just said yes to

    def needs_confirmation(self, name: str, args: Any, began_with_wake: bool) -> bool:
        if not config.WAKE_REVEAL_NEEDS_YES:  # read per call so tests can switch it
            return False
        if not began_with_wake or not _reveals(name, args):
            return False
        return (name, args) != self._approved  # the model re-issuing it after a yes runs

    def hold(self, name: str, args: dict[str, Any], now: float) -> None:
        self.pending = (name, dict(args))
        self._held_at = now
        self._approved = None

    def confirm(self, transcript: str, now: float) -> Call | None:
        """Called with each user turn's text: the pending call if it is a yes in time, else None.

        Either way the pending call is gone afterwards: it lasts one following turn.
        """
        call, self.pending = self.pending, None
        self._approved = None
        if call is None or now - self._held_at > self._window_s:
            return None
        if not _YES.search(transcript.lower()):
            return None
        self._approved = call
        return call
